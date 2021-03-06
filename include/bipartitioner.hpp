// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef INCLUDE_SPPART_BIPARTITIONER_HPP
#define INCLUDE_SPPART_BIPARTITIONER_HPP
#include<cstdint>
#include<vector>
#include<random>
#include<array>
#include<queue>
#include<memory>
#include<algorithm>
#include<omp.h>

#include<graph.hpp>
#include<boundary_list.hpp>
#include<priority_queue.hpp>
#include<bfs.hpp>
#include<linear_algebra.hpp>
#include<util.hpp>
#include<connected_component.hpp>
#include<params.hpp>
#include<sssp.hpp>
#include<maxflow/src/lib/common_types.h>
#include<maxflow/src/lib/algorithms/sequential/push_relabel_highest.h>
// #include<maxflow/src/lib/algorithms/sequential/edmonds_karp.h>
#include<unordered_map>

namespace Sppart {
    // Partitioner class
    template<class XADJ_INT, class FLOAT>
    class SpectralBipartitioner {
    protected:
        const InputParams& params;
        OutputInfo& info;
        const Graph<XADJ_INT>& g;
        int64_t cut;
        std::unique_ptr<int[]> bipartition;
        BoundaryList blist;
        std::unique_ptr<int[]> external_degrees;
        std::unique_ptr<int[]> internal_degrees;
        std::array<int, 2> part_weights;
        std::array<double, 2> target_weights_ratio;
        std::mt19937& rand_engine;

    public:

        SpectralBipartitioner(const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info)
        : g(g),        
        blist(g.nv),
        external_degrees(new int[g.nv]),
        internal_degrees(new int[g.nv]),
        bipartition(new int[g.nv]),
        rand_engine(rand_engine), info(info), params(params)
        {
            assert(params.n_dims > 0);
            assert(params.fm_max_pass >= 0);
            target_weights_ratio[0] = in_target_weights_ratio[0];
            target_weights_ratio[1] = in_target_weights_ratio[1];
        }

        static SpectralBipartitioner<XADJ_INT,FLOAT> run_partitioning(const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            SpectralBipartitioner<XADJ_INT,FLOAT> spb = spectral_bipartition(g, params, in_target_weights_ratio, rand_engine, info);
            // SpectralBipartitioner<XADJ_INT,FLOAT> spb = spectral_bipartition_experimental_subgraph_selecting(g, params, in_target_weights_ratio, rand_engine, info);

            info.spectral_cut = spb.cut;
            info.spectral_maxbal = spb.get_maxbal();

            spb.set_boundary();
            info.time_balance += timeit([&]{
                spb.balance(params.ubfactor); });
            info.balance_cut = spb.cut;
            info.balance_maxbal = spb.get_maxbal();

            info.time_fm += timeit([&]{
                info.fm_pass_count = spb.fm_refinement(params.fm_max_pass); });
            info.cut = spb.cut;
            info.maxbal = spb.get_maxbal();

            return spb;
        }

        std::unique_ptr<int[]> get_partition(){
            return std::move(bipartition);
        }

        const BoundaryList& get_blist() const {
            return blist;
        }

        protected:

        static SpectralBipartitioner<XADJ_INT,FLOAT> spectral_bipartition(const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            const int n_dims = params.n_dims;
            const size_t nv = g.nv; // prevent possible overflow when allocating unique_ptrs.

            // auto X = std::make_unique<FLOAT[]>(nv*n_dims);
            auto X = create_up_array<FLOAT>(nv*n_dims);
            auto Y = create_up_array<FLOAT>(nv*n_dims);

            compute_ssspd_basis(g, X.get(), params, rand_engine, info);

            compute_fiedler_rayleigh_ritz(params.n_dims, g, params, X.get(), Y.get(), info);

            Timer timer_round; timer_round.start();
            SpectralBipartitioner<XADJ_INT,FLOAT> spb = fiedler_partition(Y.get(), g, params, in_target_weights_ratio, rand_engine, info);
            timer_round.stop(); info.time_spectral_round += timer_round.get_time();

            return spb;
        }
        
        // This funciton implements a method presented in Lemma A.1. of [Spielman et al., Spectral partitioning works: Planar graphs and finite element meshes, 2007]
        static SpectralBipartitioner<XADJ_INT,FLOAT> spectral_bipartition_experimental_subgraph_selecting(const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            const int n_dims = params.n_dims;
            const size_t nv = g.nv; // prevent possible overflow when allocating unique_ptrs.

            auto X = create_up_array<FLOAT>(nv*n_dims);
            auto Y = create_up_array<FLOAT>(nv*n_dims);
            compute_ssspd_basis(g, X.get(), params, rand_engine, info);
            compute_fiedler_rayleigh_ritz(params.n_dims, g, X.get(), Y.get(), info);
            SpectralBipartitioner<XADJ_INT,FLOAT> spb = fiedler_partition(Y.get(), g, params, in_target_weights_ratio, rand_engine, info);
            X.release();
            Y.release();

            // printf("org cut %d, maxbal %f\n", spb.cut, spb.get_maxbal());
            spb.set_boundary();
            spb.fm_refinement(params.fm_max_pass);
            // printf("org fm cut %d, maxbal %f\n", spb.cut, spb.get_maxbal());

            fflush(stdout);
            if ( spb.get_maxbal() < params.ubfactor ){
                return spb;
            }
            int smaller_part = spb.part_weights[0] <= spb.part_weights[1] ? 1 : 0;
            int larger_part = other(smaller_part);
            int smaller_count = 0;
            int larger_count = spb.part_weights[larger_part];
            
            // auto sub_g = std::make_unique<Graph<XADJ_INT>>(g.create_subgraph([&](int i)->bool{return spb.bipartition[i]==smaller_part;}));
            std::unique_ptr<Graph<XADJ_INT>> sub_g(g.create_subgraph([&](int i)->bool{return spb.bipartition[i]==smaller_part;}));
            for (int i = 0; i < g.nv; ++i){
                spb.bipartition[i] = spb.bipartition[i] == larger_part ? larger_part : -1;
            }
            auto org_idx = create_up_array<int>(sub_g->nv);
            int cnt = 0;
            for (int i = 0; i < nv; ++i){
                if ( spb.bipartition[i] == -1 ){
                    org_idx[cnt] = i;
                    cnt++;
                }
            }
            // printf("nv small large sum %d %d %d %d %d %d\n", g.nv, smaller_part, smaller_count, larger_part, larger_count, smaller_count + larger_count);
            while( 1 ){
                const size_t sub_nv = sub_g->nv;
                auto X = create_up_array<FLOAT>(sub_nv*n_dims);
                auto Y = create_up_array<FLOAT>(sub_nv*n_dims);

                // printf("connected ? %d\n", check_connected(sub_g->nv, sub_g->xadj, sub_g->adjncy));
                compute_ssspd_basis(*sub_g, X.get(), params, rand_engine, info);
                compute_fiedler_rayleigh_ritz(params.n_dims, *sub_g, X.get(), Y.get(), info);
                SpectralBipartitioner<XADJ_INT,FLOAT> sub_spb = fiedler_min_conductance_rounding_partition(Y.get(), *sub_g, params, in_target_weights_ratio, rand_engine, info);

                // printf("sub nv %d\n", sub_nv);
                // printf("sub cut %d, maxbal %f\n", sub_spb.cut, sub_spb.get_maxbal());
                sub_spb.set_boundary();
                sub_spb.fm_refinement(params.fm_max_pass);
                // printf("sub fm cut %d, maxbal %f\n", sub_spb.cut, sub_spb.get_maxbal());

                int sub_smaller_part = sub_spb.part_weights[0] <= sub_spb.part_weights[1] ? 0 : 1;
                int sub_larger_part = other(sub_smaller_part);
                for (int i = 0; i < sub_nv; ++i){
                    if ( sub_spb.bipartition[i] == sub_smaller_part ){
                        if ( spb.bipartition[org_idx[i]] != -1 ){
                            printf("!!!! Error %d\n", i);
                            std::terminate();
                        }
                        spb.bipartition[org_idx[i]] = smaller_part;
                    }
                }
                smaller_count += sub_spb.part_weights[sub_smaller_part];
                if ( smaller_count > larger_count){
                    std::swap(smaller_part, larger_part);
                    std::swap(smaller_count, larger_count);
                }
                // printf("nv small large sum %d %d %d %d %d %d\n", g.nv, smaller_part, smaller_count, larger_part, larger_count, smaller_count + larger_count);
                // printf("org maxbal %f\n", ((double)(g.nv - smaller_count)) / (0.5*g.nv));
                if ( ((double)(g.nv - smaller_count)) / (0.5*g.nv) < params.ubfactor ){
                    for (int i = 0; i < sub_nv; ++i){
                        if ( sub_spb.bipartition[i] == sub_larger_part ){
                            if ( spb.bipartition[org_idx[i]] != -1 ){
                                printf("!!!! Error %d\n", i);
                                std::terminate();
                            }
                            spb.bipartition[org_idx[i]] = larger_part;
                        }
                    }                                        
                    break;
                }
                // sub_g = std::make_unique<Graph<XADJ_INT>>(sub_g->create_subgraph([&](int i)->bool{return sub_spb.bipartition[i]==sub_larger_part;}));
                sub_g = std::unique_ptr<Graph<XADJ_INT>>(sub_g->create_subgraph([&](int i)->bool{return sub_spb.bipartition[i]==sub_larger_part;}));
                auto new_org_idx = create_up_array<int>(sub_g->nv);
                int cnt = 0;
                for (int i = 0; i < sub_nv; ++i){
                    if ( sub_spb.bipartition[i] == sub_larger_part ){
                        new_org_idx[cnt] = org_idx[i];
                        cnt++;
                    }
                }
                org_idx = std::move(new_org_idx);
            }

            for (int i = 0; i < g.nv; ++i){
                if ( spb.bipartition[i] == -1 ){
                    printf("!!!! Error %d\n", i);
                    std::terminate();
                }
            }
            spb.setup_partitioning_data();
            // printf("org cut %d, maxbal %f\n", spb.cut, spb.get_maxbal());
            return spb;
        }

        // compute fiedler vector by Rayleigh-Ritz procedure
        // X: g.nv x n_dims matrix. column-major. contains basis for Rayleigh-Ritz in its columns.
        // Y: g.nv x n_dims matrix. column-major. Fiedler vector is returned in the first g.nv elements of Y
        static void compute_fiedler_rayleigh_ritz(const int n_dims, const Graph<XADJ_INT>& g, const InputParams& params, FLOAT* const X, FLOAT* const Y, OutputInfo& info){
            info.time_spectral_sumzero += timeit([&]{
                make_sum_to_zero(g.nv, n_dims, X); });

            if ( params.orth_alg == 0 ){
                info.time_spectral_orth += timeit([&]{
                    orthonormalize(g.nv, n_dims, X); });
            } else if ( params.orth_alg == 1 ){
                // do nothing
            } else {
                printf("Error: No such orthonormalization method\n");
                std::terminate();
            }

            info.time_spectral_spmm += timeit([&]{
                mult_laplacian_naive(g.nv, n_dims, g.xadj, g.adjncy, X, Y); });

            std::vector<FLOAT> c(n_dims*n_dims), c2(n_dims*n_dims);

            info.time_spectral_XtY += timeit([&]{
                calc_XtY(params, g.nv, n_dims, X, Y, c.data()); });

            if ( params.orth_alg == 0 ){
                info.time_spectral_eig += timeit([&]{
                    calc_eigvecs_std(n_dims, c.data()); });
            } else if ( params.orth_alg == 1 ){
                info.time_spectral_XtY += timeit([&]{
                    calc_XtY(params, g.nv, n_dims, X, X, c2.data()); });
                info.time_spectral_eig += timeit([&]{
                    calc_eigvecs_gen(n_dims, c.data(), c2.data()); });
            }

            info.time_spectral_back += timeit([&]{
                back_transform(g.nv, 1, n_dims, X, c.data(), Y); });

            fix_sign(g.nv, Y);
        }

        static void calc_XtY(const InputParams& params, const int m, const int n, const FLOAT* const X, const FLOAT* const Y, FLOAT* const ret){
            if ( params.xty_alg == 0) {
                calc_XtY_gemm(m, n, X, Y, ret);
            } else if ( params.xty_alg == 1){
                calc_XtY_org(m, n, X, Y, ret);
            } else{
                printf("Error: No such XtY implementation\n");
                std::terminate();
            }
        }

        // Determine partition using the fiedler vector
        static SpectralBipartitioner<XADJ_INT,FLOAT> fiedler_partition(const FLOAT* const fv, const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            if ( params.round_alg == 0 ){
                return fiedler_balanced_min_cut_rounding_partition(fv, g, params, in_target_weights_ratio, rand_engine, info);
            } else if ( params.round_alg == 1 ){
                return fiedler_sign_rounding_partition(fv, g, params, in_target_weights_ratio, rand_engine, info);
            } else if ( params.round_alg == 2 ){
                return fiedler_min_conductance_rounding_partition(fv, g, params, in_target_weights_ratio, rand_engine, info);
            } else if ( params.round_alg == 3 ){
                return fiedler_midflow_partition(fv, g, params, in_target_weights_ratio, rand_engine, info);
            } else {
                printf("Error: No such rounding method\n");
                std::terminate();
            }
        }

        static SpectralBipartitioner<XADJ_INT,FLOAT> fiedler_sign_rounding_partition(const FLOAT* const fv, const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            SpectralBipartitioner<XADJ_INT,FLOAT> spb(g, params, in_target_weights_ratio, rand_engine, info);
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                spb.bipartition[i] = fv[i] < 0.0 ? 0 : 1;
            }
            spb.setup_partitioning_data();
            return spb;
        }

        static SpectralBipartitioner<XADJ_INT,FLOAT> fiedler_balanced_min_cut_rounding_partition(const FLOAT* const fv, const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            SpectralBipartitioner<XADJ_INT,FLOAT> spb(g, params, in_target_weights_ratio, rand_engine, info);
            int from_part = 0;
            int to_part = 1;

            std::vector<int> perm(g.nv);
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                perm[i] = i;            
            }

            std::sort(perm.begin(), perm.end(), [&fv](size_t i1, size_t i2) {
                return fv[i1] < fv[i2];
            });

            const int i_start = g.nv*(1.0 - 0.5*params.ubfactor);
            const int i_end = g.nv - i_start;
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                spb.bipartition[perm[i]] = i < i_start ? to_part : from_part;
            }

            spb.setup_partitioning_data();

            int i_best = i_start;
            int min_cut = spb.cut;
            for (int i = i_start; i < i_end; ++i){
                const int vertex_id = perm[i];
                spb.part_weights[from_part] -= 1;
                spb.part_weights[to_part] += 1;
                spb.bipartition[vertex_id] = to_part;
                spb.cut -= spb.external_degrees[vertex_id] - spb.internal_degrees[vertex_id];
                std::swap(spb.external_degrees[vertex_id], spb.internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = to_part == spb.bipartition[j] ? 1 : -1;
                    spb.internal_degrees[j] += w;
                    spb.external_degrees[j] -= w;
                }
                if ( spb.cut < min_cut ){
                    min_cut = spb.cut;
                    i_best = i;
                }
            }

            spb.cut = min_cut;
            for (int i = i_end - 1; i > i_best; --i){
                const int vertex_id = perm[i];
                spb.part_weights[from_part] += 1;
                spb.part_weights[to_part] -= 1;
                spb.bipartition[vertex_id] = from_part;
                std::swap(spb.external_degrees[vertex_id], spb.internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = from_part == spb.bipartition[j] ? 1 : -1;
                    spb.internal_degrees[j] += w;
                    spb.external_degrees[j] -= w;
                }
            }

            return spb;
        }

        static SpectralBipartitioner<XADJ_INT,FLOAT> fiedler_min_conductance_rounding_partition(const FLOAT* const fv, const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            SpectralBipartitioner<XADJ_INT,FLOAT> spb(g, params, in_target_weights_ratio, rand_engine, info);
            int from_part = 0;
            int to_part = 1;

            std::vector<int> perm(g.nv);
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                perm[i] = i;            
            }

            std::sort(perm.begin(), perm.end(), [&fv](size_t i1, size_t i2) {
                return fv[i1] < fv[i2];
            });

            const int i_start = 1;
            const int i_end = g.nv - i_start;
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                spb.bipartition[perm[i]] = i < i_start ? to_part : from_part;
            }

            spb.setup_partitioning_data();

            int i_best = i_start;
            int min_cut = spb.cut;
            double ratio_cut = spb.cut / std::sqrt((double)std::min(spb.part_weights[0], spb.part_weights[1]));
            double min_ratio_cut = ratio_cut;

            for (int i = i_start; i < i_end; ++i){
                const int vertex_id = perm[i];
                spb.part_weights[from_part] -= 1;
                spb.part_weights[to_part] += 1;
                spb.bipartition[vertex_id] = to_part;
                spb.cut -= spb.external_degrees[vertex_id] - spb.internal_degrees[vertex_id];
                std::swap(spb.external_degrees[vertex_id], spb.internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = to_part == spb.bipartition[j] ? 1 : -1;
                    spb.internal_degrees[j] += w;
                    spb.external_degrees[j] -= w;
                }

                ratio_cut = spb.cut / (double)std::min(spb.part_weights[0], spb.part_weights[1]);

                if ( ratio_cut < min_ratio_cut ){
                    min_ratio_cut = ratio_cut;
                    i_best = i;
                    min_cut = spb.cut;
                }
            }

            spb.cut = min_cut;
            for (int i = i_end - 1; i > i_best; --i){
                const int vertex_id = perm[i];
                spb.part_weights[from_part] += 1;
                spb.part_weights[to_part] -= 1;
                spb.bipartition[vertex_id] = from_part;
                std::swap(spb.external_degrees[vertex_id], spb.internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = from_part == spb.bipartition[j] ? 1 : -1;
                    spb.internal_degrees[j] += w;
                    spb.external_degrees[j] -= w;
                }
            }

            return spb;
        }

        template<class T>
        using my_vector = std::vector<T>;

        static SpectralBipartitioner<XADJ_INT,FLOAT> fiedler_midflow_partition(const FLOAT* const fv, const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            SpectralBipartitioner<XADJ_INT,FLOAT> spb(g, params, in_target_weights_ratio, rand_engine, info);

            std::vector<int> perm(g.nv);
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                perm[i] = i;            
            }

            Timer tt;
            tt.start();
            std::sort(perm.begin(), perm.end(), [&fv](size_t i1, size_t i2) {
                return fv[i1] < fv[i2];
            });
            tt.stop();
            printf("time maxflow sort: %f\n", tt.get_time());
            tt.clear();

            const double beta = 1.0 - params.ubfactor/2.0;
            const int nF = g.nv*beta;
            const int nL = nF;
            const int nU = g.nv - nF - nL;

            const int meta_source = g.nv;
            const int meta_target = g.nv+1;

            tt.start();
            // std::vector<std::vector<basic_edge<int,int>>> maxflow_graph(g.nv+2);
            std::vector<std::vector<cached_edge<int,int>>> maxflow_graph(g.nv+2);

            for (int i = 0; i < g.nv; ++i){
                maxflow_graph[i].reserve(g.xadj[i+1] - g.xadj[i]+1); // +1 for arcs to meta-source or meta-target
            }
            maxflow_graph[meta_source].reserve(nF);
            maxflow_graph[meta_target].reserve(nL);

            std::vector<std::unordered_map<int,int>> map_array(g.nv+2);
            const int unit_capacity = 1;
            // const int infinite_capacity = 100000000;

            // Form digraph for the maxflow algorithm
            int total_meta_source_cap = 0;
            for (int i = 0; i < g.nv; ++i){
                const int i2 = perm[i];
                const int undefined = -1;
                for (XADJ_INT k = g.xadj[i2]; k < g.xadj[i2+1]; ++k){
                    const int j = g.adjncy[k];
                    map_array[i2][j] = maxflow_graph[i2].size();
                    maxflow_graph[i2].emplace_back(j, unit_capacity, undefined);
                }
                if ( i < nF) { // connect with meta source vertex
                    map_array[meta_source][i2] = maxflow_graph[meta_source].size();
                    const int cap = g.xadj[i2+1] - g.xadj[i2];
                    total_meta_source_cap += cap;
                    // maxflow_graph[meta_source].emplace_back(i2, infinite_capacity, undefined);
                    maxflow_graph[meta_source].emplace_back(i2, cap, undefined);
                    map_array[i2][meta_source] = maxflow_graph[i2].size();
                    maxflow_graph[i2].emplace_back(meta_source, 0, undefined);                    
                }else if ( i >= nF + nU ){ // connect with meta target vertex
                    map_array[i2][meta_target] = maxflow_graph[i2].size();
                    // maxflow_graph[i2].emplace_back(meta_target, infinite_capacity, undefined);
                    maxflow_graph[i2].emplace_back(meta_target, total_meta_source_cap, undefined);
                    map_array[meta_target][i2] = maxflow_graph[meta_target].size();
                    maxflow_graph[meta_target].emplace_back(i2, 0, undefined);                    
                }
            }

            // Set reverse_edge_index
            for (int i = 0; i < g.nv+2; ++i){
                for ( auto & edge : maxflow_graph[i] ){
                    edge.reverse_edge_index = map_array[edge.dst_vertex][i];
                    // edge.reverse_edge_index = map_array[edge.dst_vertex].at(i); // can be used for debug
                }
            }
            // set reverse_edge_capacity for push-relabel related algorithms cf. include/maxflow/src/graph_loader.h
            for ( auto & vec : maxflow_graph ){
                for ( auto & edge : vec ){
                    edge.reverse_r_capacity = maxflow_graph[edge.dst_vertex][edge.reverse_edge_index].r_capacity;
                }
            }
            tt.stop();
            printf("time maxflow build graph: %f\n", tt.get_time());
            tt.clear();

            tt.start();
            push_relabel_highest::max_flow_instance<my_vector, int, int> mfi(maxflow_graph, meta_source, meta_target);
            // edmonds_karp::max_flow_instance<my_vector, int, int> mfi(maxflow_graph, meta_source, meta_target);
            int flow = mfi.find_max_flow();
            printf("flow = %d\n", flow);
            tt.stop();
            printf("time maxflow run: %f\n", tt.get_time());
            tt.clear();

            maxflow_graph = mfi.steal_network();
            auto heights = create_up_array<int>(g.nv+2);
            for (int i = 0; i < g.nv+2; ++i) heights[i] = mfi.get_label(i);
            
            for (int i = 0; i < g.nv; ++i) spb.bipartition[i] = 1;
            
            auto visited = create_up_array<bool>(g.nv+2);
            for (int i = 0; i < g.nv+2; ++i) visited[i] = false;
            std::queue<int> que;
            que.push(meta_source);
            visited[meta_source] = true;

            tt.start();
            while ( !que.empty() ){
                const int v = que.front();
                que.pop();
                if (v < g.nv) spb.bipartition[v] = 0;
                for (int k = 0; k < maxflow_graph[v].size(); ++k){                    
                    auto & edge = maxflow_graph[v][k];
                    // height info should be used for push-relabel related algorithms to find s-t min-cut
                    // https://stackoverflow.com/questions/36216258/implement-push-relabel-algorithm-s-t-min-cut-edges-for-undirected-unweighted-gra
                    if ( !visited[edge.dst_vertex] && heights[edge.dst_vertex] >= heights[v] - 1 ){
                    // if ( !visited[edge.dst_vertex] && edge.r_capacity > 0 ){ // works for maxflow algorithms other than push-relabel-type algorithms
                        que.push(edge.dst_vertex);
                        visited[edge.dst_vertex] = true;
                    }
                }
            }
            tt.stop();
            printf("time maxflow bfs: %f\n", tt.get_time());
            tt.clear();

            tt.start();
            spb.setup_partitioning_data();
            tt.stop();
            printf("time maxflow setup pd: %f\n", tt.get_time());
            tt.clear();
            return spb;
        }

        void set_boundary(){
            blist.clear();
            for (int i = 0; i < g.nv; ++i){
                if ( external_degrees[i] > 0 ){
                    blist.insert(i);
                }
            }
        }

        void setup_partitioning_data(){
            int64_t cut_tmp = 0; // Use a temp variable because Intel compiler cannot correctly handle "omp parallel for reduction" with a class member variable (this->cut)

            #pragma omp parallel for reduction (+:cut_tmp)
            for (int i = 0; i < g.nv; ++i){
                internal_degrees[i] = 0;
                external_degrees[i] = 0;
                for (XADJ_INT k = g.xadj[i]; k < g.xadj[i+1]; ++k){
                    if ( bipartition[g.adjncy[k]] == bipartition[i] ){
                        internal_degrees[i] += 1;
                    }else{
                        external_degrees[i] += 1;
                    }
                }
                cut_tmp += external_degrees[i];
            }

            part_weights[0] = 0;
            part_weights[1] = 0;
            for (int i = 0; i < g.nv; ++i){
                part_weights[bipartition[i]] += 1;
            }

            assert(cut_tmp % 2 == 0);
            cut_tmp /= 2;
            assert(part_weights[0] + part_weights[1] == g.nv);
            this->cut = cut_tmp;
        }

        double get_maxbal(){
            int ave;
            if ( g.nv % 2 == 0 ){
                ave = g.nv / 2;
            }else{
                ave = g.nv / 2 + 1;
            }
            int max_part_weight = std::max(part_weights[0], part_weights[1]);
            return static_cast<double>(max_part_weight) / static_cast<double>(ave);
        }

        static SpectralBipartitioner<XADJ_INT,FLOAT> random_partition(const Graph<XADJ_INT>& g, const InputParams& params, const double* const in_target_weights_ratio, std::mt19937& rand_engine, OutputInfo& info){
            SpectralBipartitioner<XADJ_INT,FLOAT> spb(g, params, in_target_weights_ratio, rand_engine, info);

            std::uniform_int_distribution<> ud(0, 1);
            int cnt = 0;
            for (int i = 0; i < g.nv; ++i){
                if ( cnt < g.nv/2 ) {
                    const int part = ud(rand_engine);
                    spb.bipartition[i] = part;
                    if ( part == 0 ){
                        cnt++;
                    }
                }else{
                    spb.bipartition[i] = 1;
                }
            }
            return spb;
        }

        static inline int other(int i){
            return (i + 1) % 2;
        }

        void balance(const double ubfactor){
            std::array<int, 2> target_weights;
            target_weights[0] = g.nv*target_weights_ratio[0];
            target_weights[1] = g.nv - target_weights[0];

            const int from_part = part_weights[0] < target_weights[0] ? 1 : 0;
            const int to_part = other(from_part);

            PriorityQueue gain_queue(g.nv);
            std::vector<int> perm(blist.get_length());
            std::iota(perm.begin(), perm.end(), 0); // perm = [0, 1, ...]
            std::shuffle(perm.begin(), perm.end(), rand_engine);
            for (int i = 0; i < blist.get_length(); ++i){
                const int j = blist.get_boundary_at(perm[i]);
                if ( bipartition[j] == from_part ){
                    const int gain = external_degrees[j] - internal_degrees[j];
                    gain_queue.insert(j, gain);
                }
            }

            auto locked = create_up_array<bool>(g.nv);
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                locked[i] = false;
            }
            
            for (int i = 0; i < g.nv; ++i){
                if ( gain_queue.empty() ){
                    break;
                }
                const int vertex_id = gain_queue.get_top();

                // if ( part_weights[to_part] + 1 > target_weights[to_part]) {
                if ( part_weights[to_part] + 1 > g.nv - target_weights[from_part]*ubfactor) {
                    break;
                }

                cut -= external_degrees[vertex_id] - internal_degrees[vertex_id];
                part_weights[from_part] -= 1;
                part_weights[to_part] += 1;

                bipartition[vertex_id] = to_part;
                locked[vertex_id] = true;

                std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = to_part == bipartition[j] ? 1 : -1;
                    internal_degrees[j] += w;
                    external_degrees[j] -= w;

                    if (blist.is_boundary(j)){
                        if (external_degrees[j] == 0){
                            blist.remove(j);
                            if ( !locked[j] && bipartition[j] == from_part ){
                                gain_queue.remove(j);
                            }
                        }else{
                            if (!locked[j] && bipartition[j] == from_part ){
                                const int gain = external_degrees[j] - internal_degrees[j];
                                gain_queue.update(j, gain);
                            }
                        }
                    }else{
                        if ( external_degrees[j] > 0 ){
                            blist.insert(j);
                            if ( !locked[j] && bipartition[j] == from_part ){
                                const int gain = external_degrees[j] - internal_degrees[j];
                                gain_queue.insert(j, gain);
                            }
                        }
                    }
                }
            }               
        }

        int fm_refinement(const int max_pass){
            // const int max_pass = params.fm_max_pass;
            const bool no_limit = params.fm_no_limit;
            int limit;
            if ( params.fm_limit >= 0 ){
                limit = params.fm_limit;
            }else{
                limit = std::min(std::max(static_cast<int>(0.01*g.nv), 15), 100); // from Metis
            }

            std::array<int, 2> target_weights;
            target_weights[0] = g.nv*target_weights_ratio[0];
            target_weights[1] = g.nv - target_weights[0];

            int average_weight = std::min(g.nv/20, 2); // from Metis
            int org_diff = std::abs(target_weights[0] - part_weights[0]);// from Metis

            // Create vertex gain priority queue for each part
            // Key is vertex_id and Value is gain value
            std::array<PriorityQueue, 2> gain_queues = {PriorityQueue(g.nv), PriorityQueue(g.nv)};

            auto locked = create_up_array<bool>(g.nv);
            auto vertex_id_history = create_up_array<int>(g.nv);
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                locked[i] = false;
                // vertex_id_history[i] = -1;
            }
    
            int pass_count = 0;
            for (int i_pass = 0; i_pass < max_pass; ++i_pass){
                pass_count++;

                gain_queues[0].reset();
                gain_queues[1].reset();

                std::vector<int> perm(blist.get_length());
                std::iota(perm.begin(), perm.end(), 0); // perm = [0, 1, ...]
                std::shuffle(perm.begin(), perm.end(), rand_engine);
                for (int i = 0; i < blist.get_length(); ++i){
                    const int j = blist.get_boundary_at(perm[i]);
                    const int gain = external_degrees[j] - internal_degrees[j];
                    gain_queues[bipartition[j]].insert(j, gain);
                }
    
                const int init_cut = cut;
                int minimum_cut = cut;
                int minimum_diff = std::abs(target_weights[0] - part_weights[0]);
                int i_mincut = -1;
                int try_count = 0;

                const int max_move = params.fm_max_move >= 0 ? std::min(params.fm_max_move, g.nv) : g.nv;
                for (int i = 0; i < max_move; ++i){
                    const int from_part = target_weights[0] - part_weights[0] < target_weights[1] - part_weights[1] ? 0 : 1;
                    const int to_part = other(from_part);
                    if ( gain_queues[from_part].empty() ) {
                        break;
                    }
                    try_count++;
                    const int vertex_id = gain_queues[from_part].get_top();

                    // assert(blist.is_boundary(vertex_id));
                    const int gain = external_degrees[vertex_id] - internal_degrees[vertex_id];
                    cut -= gain;
            
                    part_weights[from_part] -= 1;
                    part_weights[to_part] += 1;
                    const int new_diff = std::abs(target_weights[0] - part_weights[0]);

                    if ( (cut < minimum_cut && new_diff <= org_diff + average_weight)
                        || (cut == minimum_cut && new_diff < minimum_diff ) ){ // from Metis
                    // if ( (cut < minimum_cut && std::max(part_weights[0], part_weights[1]) < g.nv*0.5*params.ubfactor )
                    //      || (cut == minimum_cut && new_diff < minimum_diff ) ){ // from Metis
                        minimum_cut = cut;
                        minimum_diff = new_diff;
                        i_mincut = i;
                    }else if ( !no_limit && i - i_mincut > limit ){
                        cut += gain;
                        part_weights[from_part] += 1;
                        part_weights[to_part] -= 1;
                        try_count--;
                        break;
                    }

                    bipartition[vertex_id] = to_part;
                    locked[vertex_id] = true;
                    vertex_id_history[i] = vertex_id;

                    // Swap ed id
                    std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);

                    if ( external_degrees[vertex_id] == 0 && g.xadj[vertex_id] < g.xadj[vertex_id+1] ){
                        blist.remove(vertex_id);
                    }
                
                    for ( XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                        const int j = g.adjncy[k];
                        const int w = to_part == bipartition[j] ? 1 : -1;
                        internal_degrees[j] += w;
                        external_degrees[j] -= w;

                        if ( blist.is_boundary(j) ){
                            if ( external_degrees[j] == 0 ){
                                blist.remove(j);
                                if ( !locked[j] ){
                                    gain_queues[bipartition[j]].remove(j); // just remove j from queue
                                }
                            }else{
                                if ( !locked[j] ){
                                    const int gain = external_degrees[j] - internal_degrees[j];
                                    gain_queues[bipartition[j]].update(j, gain);
                                }
                            }                    
                        }else{
                            if ( external_degrees[j] > 0 ){
                                blist.insert(j);
                                if (!locked[j]){
                                    const int gain = external_degrees[j] - internal_degrees[j];
                                    gain_queues[bipartition[j]].insert(j, gain);
                                }
                            }
                        }
                    }
                }

                // Clear locked array
                #pragma omp parallel for
                for (int i = 0; i < try_count; ++i){
                    locked[vertex_id_history[i]] = false;
                }

                for (try_count--; try_count > i_mincut; --try_count){
                    const int vertex_id = vertex_id_history[try_count];
                    const int from_part = bipartition[vertex_id];
                    const int to_part = other(from_part);
                    bipartition[vertex_id] = to_part;

                    // Swap ed id
                    std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);

                    if ( external_degrees[vertex_id] == 0 && blist.is_boundary(vertex_id) && g.xadj[vertex_id] < g.xadj[vertex_id+1] ){
                        blist.remove(vertex_id);
                    }else if (external_degrees[vertex_id] > 0 && !blist.is_boundary(vertex_id)){
                        blist.insert(vertex_id);
                    }
                    part_weights[from_part] -= 1;
                    part_weights[to_part] += 1;

                    for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                        const int j = g.adjncy[k];
                        const int w = (to_part == bipartition[j]) ? 1 : -1;
                        internal_degrees[j] += w;
                        external_degrees[j] -= w;           
                        if ( blist.is_boundary(j) && external_degrees[j] == 0){
                            blist.remove(j);
                        }
                        if (!blist.is_boundary(j) && external_degrees[j] > 0){
                            blist.insert(j);
                        }
                    }
                }
                cut = minimum_cut;

                if ( i_mincut <= 0 || cut == init_cut ){
                    break; // break from i_pass loop
                }
            }
            return pass_count;
        }

        int fm_refinement_hill_climb(const int max_pass){
            printf("fm_refinement_hill_climb is under development. do not use it.\n");
            std::terminate();

            std::array<int, 2> target_weights;
            target_weights[0] = g.nv*target_weights_ratio[0];
            target_weights[1] = g.nv - target_weights[0];

            // Create vertex gain priority queue for each part
            // Key is vertex_id and Value is gain value
            std::array<PriorityQueue, 2> gain_queues = {PriorityQueue(g.nv), PriorityQueue(g.nv)};

            auto locked = create_up_array<bool>(g.nv);
            auto is_in_hill = create_up_array<bool>(g.nv);
            auto touched_hill_search = create_up_array<bool>(g.nv);
            auto vertex_id_history = create_up_array<int>(g.nv);
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                locked[i] = false;
                is_in_hill[i] = false;
                touched_hill_search[i] = false;
            }
    
            int pass_count = 0;
            for (int i_pass = 0; i_pass < max_pass; ++i_pass){
                printf("ipass = %d\n", i_pass);
                pass_count++;

                gain_queues[0].reset();
                gain_queues[1].reset();

                std::vector<int> perm(blist.get_length());
                std::iota(perm.begin(), perm.end(), 0); // perm = [0, 1, ...]
                std::shuffle(perm.begin(), perm.end(), rand_engine);
                for (int i = 0; i < blist.get_length(); ++i){
                    const int j = blist.get_boundary_at(perm[i]);
                    const int gain = external_degrees[j] - internal_degrees[j];
                    gain_queues[bipartition[j]].insert(j, gain);
                }
    
                const int init_cut = cut;
                int move_count = 0;

                auto traversed = create_up_array<bool>(g.xadj[g.nv]);
                #pragma omp parallel for
                for (int i = 0; i < g.xadj[g.nv]; ++i){
                    traversed[i] = false;
                }

                for (int i = 0; i < g.nv; ++i){
                    int from_part;
                    int gain0 = gain_queues[0].see_top_val();
                    int gain1 = gain_queues[1].see_top_val();                    
                    if ( !gain_queues[0].empty() && (std::max(part_weights[0]-1, part_weights[1]+1) < g.nv*0.5*params.ubfactor) ){
                        if ( !gain_queues[1].empty() && (std::max(part_weights[1]-1, part_weights[0]+1) < g.nv*0.5*params.ubfactor) ) {
                            // both moves are feasible
                            from_part = gain0 > gain1 ? 0 : 1;
                        }else{
                            // only 0 -> 1 move is feasible
                            from_part = 0;
                        }
                    }else{
                        if ( !gain_queues[1].empty() && (std::max(part_weights[1]-1, part_weights[0]+1) < g.nv*0.5*params.ubfactor) ) {
                            // only 1 -> 0 move is feasible
                            from_part = 1;
                        }else{
                            // both moves are not feasible, make balance better
                            from_part = target_weights[0] - part_weights[0] < target_weights[1] - part_weights[1] ? 0 : 1;
                            if ( gain_queues[from_part].empty() ) break;
                        }
                    }
                    // const int from_part = target_weights[0] - part_weights[0] < target_weights[1] - part_weights[1] ? 0 : 1;
                    const int to_part = other(from_part);

                    if ( gain_queues[from_part].empty() ) {
                        break;
                    }

                    int v = gain_queues[from_part].get_top();
                    int gain = external_degrees[v] - internal_degrees[v];
                    std::vector<int> vertices_to_move;
                    std::vector<int> vertices_in_queue_non_boundary;
                    std::vector<int> vertices_touched;
                    vertices_to_move.push_back(v);
                    vertices_touched.push_back(v);
                    int move_size = 1;
                    is_in_hill[v] = true;
                    touched_hill_search[v] = true;

                    int hill_ed = external_degrees[v]; // hill external degree
                    int hill_id = internal_degrees[v]; // hill internal degree
                    while ( gain < 0 && move_size < 100 && std::max(part_weights[from_part]-move_size, part_weights[to_part]+move_size) < g.nv*0.5*params.ubfactor ) { // hill climing loop
                        // Add the neighbors of candidate to the queue even if they are not boundary vertices
                        for ( XADJ_INT k = g.xadj[v]; k < g.xadj[v+1]; ++k){
                            int u = g.adjncy[k];
                            if ( traversed[k] || locked[u] || is_in_hill[v] || touched_hill_search[u] || bipartition[u] != from_part ) continue; 
                            if ( !gain_queues[from_part].contains(u) ) gain_queues[from_part].insert(u, external_degrees[u] - internal_degrees[u]);
                            traversed[k] = true;
                            touched_hill_search[u] = true;
                            vertices_touched.push_back(u);
                            if ( !blist.is_boundary(u) ) vertices_in_queue_non_boundary.push_back(u);
                        }
                        if ( gain_queues[from_part].empty() ) break;
                        v = gain_queues[from_part].get_top();
                        is_in_hill[v] = true;
                        vertices_to_move.push_back(v);
                        move_size++;
                        hill_ed += external_degrees[v];
                        hill_id += internal_degrees[v];

                        gain = hill_ed - hill_id;
                        printf("%d: v %d gain %d\n", move_size, v, gain);
                    }

                    for ( int u : vertices_touched ) touched_hill_search[u] = false;
                    
                    for ( int u : vertices_in_queue_non_boundary ){
                        if ( !is_in_hill[u] ) {
                            gain_queues[from_part].remove(u);
                        }
                    }

                    if ( gain < 0 ){
                        for ( int u : vertices_to_move ){
                            is_in_hill[u] = false; // for next iteration
                        }
                        continue;
                    }

                    cut -= gain;
            
                    part_weights[from_part] -= move_size;
                    part_weights[to_part] += move_size;

                    if ( move_size > 1 && gain >= 0 ){
                        printf("hill building finish gain = %d size = %d\n", gain, move_size);
                    }

                    for ( int u : vertices_to_move ){
                        locked[u] = true;
                        bipartition[u] = to_part;
                        is_in_hill[u] = false;
                        vertex_id_history[move_count++] = u;
                        // Swap ed id
                        std::swap(external_degrees[u], internal_degrees[u]);
                        if ( blist.is_boundary(u) && external_degrees[u] == 0 && g.xadj[u] < g.xadj[u+1] ){
                            blist.remove(u);
                        }
                        if ( !blist.is_boundary(u) && external_degrees[u] > 0 ){
                            blist.insert(u);
                        }
                        for ( XADJ_INT k = g.xadj[u]; k < g.xadj[u+1]; ++k){
                            const int j = g.adjncy[k];
                            if ( is_in_hill[j] ) continue;
                            const int w = to_part == bipartition[j] ? 1 : -1;
                            internal_degrees[j] += w;
                            external_degrees[j] -= w;

                            if ( blist.is_boundary(j) ){
                                if ( external_degrees[j] == 0 ){
                                    blist.remove(j);
                                    if ( !locked[j] ){
                                        gain_queues[bipartition[j]].remove(j); // just remove j from queue
                                    }
                                }else{
                                    if ( !locked[j] ){
                                        const int new_gain = external_degrees[j] - internal_degrees[j];
                                        if ( gain_queues[bipartition[j]].contains(j) ) //!!!!!!!!!!!
                                            gain_queues[bipartition[j]].update(j, new_gain);
                                    }
                                }                    
                            }else{
                                if ( external_degrees[j] > 0 ){
                                    blist.insert(j);
                                    if (!locked[j]){
                                        const int new_gain = external_degrees[j] - internal_degrees[j];
                                        gain_queues[bipartition[j]].insert(j, new_gain);
                                    }
                                }
                            }
                        }
                    }

                    printf("cut %d maxbal %lf\n", cut, get_maxbal());
                }

                // Clear locked array
                #pragma omp parallel for
                for (int i = 0; i < move_count; ++i){
                    locked[vertex_id_history[i]] = false;
                }

                if ( cut == init_cut ){
                    break; // break from i_pass loop
                }
            }
            return pass_count;
        }


        int fm_refinement_allow_worse_balance(const int max_pass, const double ubfactor){
            // const int max_pass = params.fm_max_pass;
            const bool no_limit = params.fm_no_limit;
            int limit;
            if ( params.fm_limit >= 0 ){
                limit = params.fm_limit;
            }else{
                limit = std::min(std::max(static_cast<int>(0.01*g.nv), 15), 100); // from Metis
            }

            std::array<int, 2> target_weights;
            target_weights[0] = g.nv*target_weights_ratio[0];
            target_weights[1] = g.nv - target_weights[0];

            int average_weight = std::min(g.nv/20, 2); // from Metis
            int org_diff = std::abs(target_weights[0] - part_weights[0]);// from Metis

            // Create vertex gain priority queue
            // Key is vertex_id and Value is gain value
            PriorityQueue gain_queue(g.nv);

            auto locked = create_up_array<bool>(g.nv);
            auto vertex_id_history = create_up_array<int>(g.nv);
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                locked[i] = false;
            }
    
            int pass_count = 0;
            for (int i_pass = 0; i_pass < max_pass; ++i_pass){
                pass_count++;

                gain_queue.reset();

                std::vector<int> perm(blist.get_length());
                std::iota(perm.begin(), perm.end(), 0); // perm = [0, 1, ...]
                std::shuffle(perm.begin(), perm.end(), rand_engine);
                for (int i = 0; i < blist.get_length(); ++i){
                    const int j = blist.get_boundary_at(perm[i]);
                    const int gain = external_degrees[j] - internal_degrees[j];
                    gain_queue.insert(j, gain);
                }
    
                const int init_cut = cut;
                int minimum_cut = cut;
                int minimum_diff = std::abs(target_weights[0] - part_weights[0]);
                int i_mincut = -1;
                int try_count = 0;

                for (int i = 0; i < g.nv; ++i){
                    int vertex_id;
                    int from_part;
                    int to_part;
                    while ( !gain_queue.empty() ) {
                        vertex_id = gain_queue.get_top();
                        from_part = bipartition[vertex_id];
                        to_part = other(from_part);
                        if ( std::max(part_weights[from_part]-1, part_weights[to_part]+1) < g.nv*0.5*ubfactor ){
                            break;
                        }
                    }
                    if ( gain_queue.empty() ) {
                        break;
                    }                    
                    try_count++;

                    const int gain = external_degrees[vertex_id] - internal_degrees[vertex_id];
                    // printf("gain %d\n", gain);
                    cut -= gain;
            
                    part_weights[from_part] -= 1;
                    part_weights[to_part] += 1;
                    const int new_diff = std::abs(target_weights[0] - part_weights[0]);

                    // if ( (cut < minimum_cut && new_diff <= org_diff + average_weight)
                    //     || (cut == minimum_cut && new_diff < minimum_diff ) ){ // from Metis
                    if ( cut < minimum_cut ){
                        minimum_cut = cut;
                        minimum_diff = new_diff;
                        i_mincut = i;
                    }else if ( !no_limit && i - i_mincut > limit ){
                        cut += gain;
                        part_weights[from_part] += 1;
                        part_weights[to_part] -= 1;
                        try_count--;
                        break;
                    }

                    bipartition[vertex_id] = to_part;
                    locked[vertex_id] = true;
                    vertex_id_history[i] = vertex_id;

                    // Swap ed id
                    std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);

                    if ( external_degrees[vertex_id] == 0 && g.xadj[vertex_id] < g.xadj[vertex_id+1] ){
                        blist.remove(vertex_id);
                    }
                
                    for ( XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                        const int j = g.adjncy[k];
                        const int w = to_part == bipartition[j] ? 1 : -1;
                        internal_degrees[j] += w;
                        external_degrees[j] -= w;

                        if ( blist.is_boundary(j) ){
                            if ( external_degrees[j] == 0 ){
                                blist.remove(j);
                                if ( !locked[j] ){
                                    gain_queue.remove(j); // just remove j from queue
                                }
                            }else{
                                if ( !locked[j] ){
                                    const int gain = external_degrees[j] - internal_degrees[j];
                                    gain_queue.update(j, gain);
                                }
                            }                    
                        }else{
                            if ( external_degrees[j] > 0 ){
                                blist.insert(j);
                                if (!locked[j]){
                                    const int gain = external_degrees[j] - internal_degrees[j];
                                    gain_queue.insert(j, gain);
                                }
                            }
                        }
                    }
                }

                // Clear locked array
                #pragma omp parallel for
                for (int i = 0; i < try_count; ++i){
                    locked[vertex_id_history[i]] = false;
                }

                for (try_count--; try_count > i_mincut; --try_count){
                    const int vertex_id = vertex_id_history[try_count];
                    const int from_part = bipartition[vertex_id];
                    const int to_part = other(from_part);
                    bipartition[vertex_id] = to_part;

                    // Swap ed id
                    std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);

                    if ( external_degrees[vertex_id] == 0 && blist.is_boundary(vertex_id) && g.xadj[vertex_id] < g.xadj[vertex_id+1] ){
                        blist.remove(vertex_id);
                    }else if (external_degrees[vertex_id] > 0 && !blist.is_boundary(vertex_id)){
                        blist.insert(vertex_id);
                    }
                    part_weights[from_part] -= 1;
                    part_weights[to_part] += 1;

                    for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                        const int j = g.adjncy[k];
                        const int w = (to_part == bipartition[j]) ? 1 : -1;
                        internal_degrees[j] += w;
                        external_degrees[j] -= w;           
                        if ( blist.is_boundary(j) && external_degrees[j] == 0){
                            blist.remove(j);
                        }
                        if (!blist.is_boundary(j) && external_degrees[j] > 0){
                            blist.insert(j);
                        }
                    }
                }
                cut = minimum_cut;

                if ( i_mincut <= 0 || cut == init_cut ){
                    break; // break from i_pass loop
                }
            }
            return pass_count;
        }

        SpectralBipartitioner(const SpectralBipartitioner&) = delete;
        SpectralBipartitioner& operator=(const SpectralBipartitioner&) = delete;

        public:
        SpectralBipartitioner(SpectralBipartitioner&&) = default;
        SpectralBipartitioner& operator=(SpectralBipartitioner&&) = delete;
    };
}

#endif // INCLUDE_SPPART_BIPARTITIONER_HPP
