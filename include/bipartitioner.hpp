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
#include<timer.hpp>
#include<connected_component.hpp>
#include<params.hpp>
#include<sssp.hpp>

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
        params(params),
        blist(g.nv),
        external_degrees(new int[g.nv]),
        internal_degrees(new int[g.nv]),
        rand_engine(rand_engine),
        info(info)
        {
            assert(params.n_dims > 0);
            assert(params.max_fm_pass >= 0);
            target_weights_ratio[0] = in_target_weights_ratio[0];
            target_weights_ratio[1] = in_target_weights_ratio[1];
        }

        OutputInfo run_partitioning(bool assume_connected){
            bipartition = std::make_unique<int[]>(g.nv+1); // // +1 is for pseudo vertex

            if ( assume_connected ){
                info.time_spectral += timeit([&]{
                    spectral_bipartition(g); 
                });
            }else{
                bool is_connected;
                Timer timer_connect; timer_connect.start();            
                // const Graph<XADJ_INT> connected_g = make_connected(g.nv, g.xadj, g.adjncy, is_connected);
                const Graph<XADJ_INT> connected_g = make_connected_by_func(g.nv, g.xadj, g.adjncy, is_connected);
                timer_connect.stop(); info.time_connect += timer_connect.get_time();
                // if (!is_connected) printf("Disconnected graph found make connected!!!\n");
                info.time_spectral += timeit([&]{
                    if ( is_connected ){
                        spectral_bipartition(g); 
                    }else{
                        spectral_bipartition(connected_g); 
                    }
                });
            }

            info.spectral_cut = cut;
            info.spectral_maxbal = get_maxbal();

            set_boundary();
            info.time_balance += timeit([&]{
                balance(params.ubfactor); });
            info.balance_cut = cut;
            info.balance_maxbal = get_maxbal();

            info.time_fm += timeit([&]{
                info.fm_pass_count = fm_refinement(params.fm_max_pass); });
            info.cut = cut;
            info.maxbal = get_maxbal();

            return info;
        }

        std::unique_ptr<int[]> get_partition(){
            return std::move(bipartition);
        }

        const BoundaryList& get_blist() const {
            return blist;
        }

        protected:


        void spectral_bipartition(const Graph<XADJ_INT>& g){
            const int n_dims = params.n_dims;
            const size_t nv = g.nv; // prevent possible overflow when allocating unique_ptrs.

            auto X = std::make_unique<FLOAT[]>(nv*n_dims);
            auto Y = std::make_unique<FLOAT[]>(nv*n_dims);

            compute_ssspd_basis(g, X.get(), params, rand_engine, info);

            compute_fiedler_rayleigh_ritz(g, X.get(), Y.get());

            fiedler_partition(Y.get());
        }
        
        // compute fiedler vector by Rayleigh-Ritz procedure
        // X: g.nv x n_dims matrix. column-major. contains basis for Rayleigh-Ritz in its columns.
        // Y: g.nv x n_dims matrix. column-major. Fiedler vector is returned in the first g.nv elements of Y
        void compute_fiedler_rayleigh_ritz(const Graph<XADJ_INT>& g, FLOAT* const X, FLOAT* const Y){
            const int n_dims = params.n_dims;
            info.time_spectral_sumzero += timeit([&]{
                make_sum_to_zero(g.nv, n_dims, X); });

            info.time_spectral_orth += timeit([&]{
                orthonormalize(g.nv, n_dims, X); });

            info.time_spectral_spmm += timeit([&]{
                mult_laplacian_naive(g.nv, n_dims, g.xadj, g.adjncy, X, Y); });

            std::vector<FLOAT> c(n_dims*n_dims);

            info.time_spectral_XtY += timeit([&]{
                calc_XtY(g.nv, n_dims, X, Y, c.data()); });

            info.time_spectral_eig += timeit([&]{
                calc_eigvecs(n_dims, c.data()); });

            info.time_spectral_back += timeit([&]{
                back_transform(g.nv, 1, n_dims, X, c.data(), Y); });

            fix_sign(g.nv, Y);
        }

        // Determine partition using the fiedler vector
        void fiedler_partition(const FLOAT* const fv){
            info.time_spectral_round += timeit([&]{
                if ( params.round_alg == 0 ){
                    fiedler_balanced_min_cut_rounding_partition(fv);
                } else if ( params.round_alg == 1 ){
                    fiedler_sign_rounding_partition(fv);
                } else if ( params.round_alg == 2 ){
                    fiedler_min_conductance_rounding_partition(fv);
                } else {
                    printf("Error: No such rounding method\n");
                    std::terminate();
                }
            });
        }

        void fiedler_sign_rounding_partition(const FLOAT* const fv){
            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i){
                bipartition[i] = fv[i] < 0.0 ? 0 : 1;
            }
            setup_partitioning_data();
        }

        void fiedler_balanced_min_cut_rounding_partition(const FLOAT* const fv){
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
                bipartition[perm[i]] = i < i_start ? to_part : from_part;
            }

            setup_partitioning_data();

            int i_best = i_start;
            int min_cut = cut;
            for (int i = i_start; i < i_end; ++i){
                const int vertex_id = perm[i];
                part_weights[from_part] -= 1;
                part_weights[to_part] += 1;
                bipartition[vertex_id] = to_part;
                cut -= external_degrees[vertex_id] - internal_degrees[vertex_id];
                std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = to_part == bipartition[j] ? 1 : -1;
                    internal_degrees[j] += w;
                    external_degrees[j] -= w;
                }
                if ( cut < min_cut ){
                    min_cut = cut;
                    i_best = i;
                }
            }

            cut = min_cut;
            for (int i = i_end - 1; i > i_best; --i){
                const int vertex_id = perm[i];
                part_weights[from_part] += 1;
                part_weights[to_part] -= 1;
                bipartition[vertex_id] = from_part;
                std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = from_part == bipartition[j] ? 1 : -1;
                    internal_degrees[j] += w;
                    external_degrees[j] -= w;
                }
            }
        }

        void fiedler_min_conductance_rounding_partition(const FLOAT* const fv){
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
                bipartition[perm[i]] = i < i_start ? to_part : from_part;
            }

            setup_partitioning_data();

            int i_best = i_start;
            int min_cut = cut;
            double ratio_cut = cut / std::sqrt((double)std::min(part_weights[0], part_weights[1]));
            double min_ratio_cut = ratio_cut;

            for (int i = i_start; i < i_end; ++i){
                const int vertex_id = perm[i];
                part_weights[from_part] -= 1;
                part_weights[to_part] += 1;
                bipartition[vertex_id] = to_part;
                cut -= external_degrees[vertex_id] - internal_degrees[vertex_id];
                std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = to_part == bipartition[j] ? 1 : -1;
                    internal_degrees[j] += w;
                    external_degrees[j] -= w;
                }

                ratio_cut = cut / (double)std::min(part_weights[0], part_weights[1]);

                if ( ratio_cut < min_ratio_cut ){
                    min_ratio_cut = ratio_cut;
                    i_best = i;
                    min_cut = cut;
                }
            }

            cut = min_cut;
            for (int i = i_end - 1; i > i_best; --i){
                const int vertex_id = perm[i];
                part_weights[from_part] += 1;
                part_weights[to_part] -= 1;
                bipartition[vertex_id] = from_part;
                std::swap(external_degrees[vertex_id], internal_degrees[vertex_id]);
                for (XADJ_INT k = g.xadj[vertex_id]; k < g.xadj[vertex_id+1]; ++k){
                    const int j = g.adjncy[k];
                    const int w = from_part == bipartition[j] ? 1 : -1;
                    internal_degrees[j] += w;
                    external_degrees[j] -= w;
                }
            }
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
            cut = 0;

            #pragma omp parallel for reduction (+:cut)
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
                cut += external_degrees[i];
            }

            part_weights[0] = 0;
            part_weights[1] = 0;
            for (int i = 0; i < g.nv; ++i){
                part_weights[bipartition[i]] += 1;
            }

            assert(cut % 2 == 0);
            cut /= 2;
            assert(part_weights[0] + part_weights[1] == g.nv);
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

        void random_partition(int seed){
            std::uniform_int_distribution<> ud(0, 1);
            int cnt = 0;
            for (int i = 0; i < g.nv; ++i){
                if ( cnt < g.nv/2 ) {
                    const int part = ud(rand_engine);
                    bipartition[i] = part;
                    if ( part == 0 ){
                        cnt++;
                    }
                }else{
                    bipartition[i] = 1;
                }
            }
            return;
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

            auto locked = std::make_unique<bool[]>(g.nv);
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

            auto locked = std::make_unique<bool[]>(g.nv);
            auto vertex_id_history = std::make_unique<int[]>(g.nv);
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

                for (int i = 0; i < g.nv; ++i){
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

        SpectralBipartitioner(const SpectralBipartitioner&) = delete;
        SpectralBipartitioner& operator=(const SpectralBipartitioner&) = delete;
        SpectralBipartitioner(SpectralBipartitioner&&) = delete;
        SpectralBipartitioner& operator=(SpectralBipartitioner&&) = delete;
    };
}

#endif // INCLUDE_SPPART_BIPARTITIONER_HPP
