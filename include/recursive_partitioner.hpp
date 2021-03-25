// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#pragma once
#include<bipartitioner.hpp>

namespace Sppart {

    template<class XADJ_INT, class FLOAT>
    class RecursivePartitioner {
    protected:
        const InputParams& params;
        OutputInfo info;
        const Graph<XADJ_INT>& g;
        std::unique_ptr<int[]> partition;
        std::mt19937 rand_engine;

    public:

        RecursivePartitioner(const Graph<XADJ_INT>& g, const InputParams& params)
        : g(g), params(params), rand_engine(params.rand_seed)
        {
        }

        OutputInfo run_partitioning(int n_parts){
            std::vector<double> target_weights_ratio(n_parts);
            for (int i = 0; i < n_parts; ++i) target_weights_ratio[i] = 1.0 / n_parts;

            std::array<double,2> target_bipart_weights_ratio;
            target_bipart_weights_ratio[0] = 0.0;
            for (int i = 0; i < n_parts / 2; ++i){
                target_bipart_weights_ratio[0] += target_weights_ratio[i];
            }
            target_bipart_weights_ratio[1] = 1.0 - target_bipart_weights_ratio[0];

            Timer timer_spectral; timer_spectral.start();
            SpectralBipartitioner<XADJ_INT, FLOAT> bipartitioner = SpectralBipartitioner<XADJ_INT, FLOAT>::run_partitioning(g, params, target_bipart_weights_ratio.data(), rand_engine, info);
            timer_spectral.stop(); info.time_spectral += timer_spectral.get_time();

            partition = bipartitioner.get_partition();

            int64_t cut = info.cut;

            if ( n_parts > 2 ){
                auto label = std::make_unique<int[]>(g.nv);
                #pragma omp parallel for
                for (int i = 0; i < g.nv; ++i){
                    label[i] = i;
                }

                double wsum = 0.0;
                for (int i = 0; i < n_parts / 2; ++i){
                    wsum += target_weights_ratio[i];
                }
                for (int i = 0; i < n_parts / 2; ++i){
                    target_weights_ratio[i] *= 1.0 / wsum;
                    target_weights_ratio[n_parts / 2 + i] *= 1.0 / (1.0 - wsum);
                }

                std::array<std::unique_ptr<int[]>, 2> slabel;
                Timer timer_split;
                timer_split.start();
                std::array<Graph<XADJ_INT>, 2> children = split_graph(g, label.get(), partition.get(), bipartitioner.get_blist(), slabel);
                timer_split.stop();
                info.time_split += timer_split.get_time();
                
                const int fpart = 0;
                cut += recursive_partitioning(children[0], n_parts/2, target_weights_ratio.data(), fpart, slabel[0].get());
                children[0].release();
                cut += recursive_partitioning(children[1], n_parts - n_parts/2, &target_weights_ratio[n_parts/2], fpart + n_parts/2, slabel[1].get());
            }
            info.cut = cut;
            return info;
        }

        std::unique_ptr<int[]> get_partition(){
            return std::move(partition);
        }

    protected:

        int64_t recursive_partitioning(Graph<XADJ_INT>& ag, const int n_parts, double* const target_weights_ratio, const int fpart, const int* const label){
            if ( n_parts <= 1 ){
                return 0;
            }
            if ( ag.nv == 0 ){
                printf("Error: No vertex in the graph\n");
                std::terminate();
            }

            std::array<double,2> target_bipart_weights_ratio;
            target_bipart_weights_ratio[0] = 0.0;
            for (int i = 0; i < n_parts / 2; ++i){
                target_bipart_weights_ratio[0] += target_weights_ratio[i];
            }
            target_bipart_weights_ratio[1] = 1.0 - target_bipart_weights_ratio[0];

            bool is_connected;
            Timer timer_connect; timer_connect.start();            
            const Graph<XADJ_INT> connected_ag = make_connected_by_func(ag.nv, ag.xadj, ag.adjncy, is_connected);
            timer_connect.stop(); info.time_connect += timer_connect.get_time();
            // if (!is_connected) printf("Disconnected graph found make connected!!!\n");

            const Graph<XADJ_INT> *ag_ptr = &ag;
            if ( !is_connected ){
                ag_ptr = &connected_ag;
            }

            Timer timer_spectral; timer_spectral.start();
            SpectralBipartitioner<XADJ_INT, FLOAT> bipartitioner = SpectralBipartitioner<XADJ_INT, FLOAT>::run_partitioning(*ag_ptr, params, target_bipart_weights_ratio.data(), rand_engine, info);
            timer_spectral.stop(); info.time_spectral += timer_spectral.get_time();

            int64_t cut = info.cut;
            std::unique_ptr<int[]> bipartition = bipartitioner.get_partition();
            // printf("cut %lld\n", info.cut);

            #pragma omp parallel for
            for (int i = 0; i < ag.nv; ++i){
                partition[label[i]] = bipartition[i] + fpart;
            }

            if ( n_parts > 2 ){
                double wsum = 0.0;
                for (int i = 0; i < n_parts / 2; ++i){
                    wsum += target_weights_ratio[i];
                }
                for (int i = 0; i < n_parts / 2; ++i){
                    target_weights_ratio[i] *= 1.0 / wsum;
                    target_weights_ratio[n_parts / 2 + i] *= 1.0 / (1.0 - wsum);
                }

                std::array<std::unique_ptr<int[]>, 2> slabel;

                Timer timer_split;
                timer_split.start();
                std::array<Graph<XADJ_INT>, 2> children = split_graph(ag, label, bipartition.get(), bipartitioner.get_blist(), slabel);
                timer_split.stop();
                info.time_split += timer_split.get_time();
                ag.release();

                cut += recursive_partitioning(children[0], n_parts/2, target_weights_ratio, fpart, slabel[0].get());
                children[0].release();
                cut += recursive_partitioning(children[1], n_parts - n_parts/2, &target_weights_ratio[n_parts/2], fpart + n_parts/2, slabel[1].get());
            }
            return cut;
        }

        // SplitGraphPart of metis-5.1.0/libmetis/pmetis.c
        static std::array<Graph<XADJ_INT>, 2> split_graph(const Graph<XADJ_INT> &g, const int* const label, const int* const bipartition, const BoundaryList &blist, std::array<std::unique_ptr<int[]>, 2> &slabel_out){
            int snvtxs[2], snedges[2];
            auto rename = std::make_unique<int[]>(g.nv);

            snvtxs[0] = snvtxs[1] = snedges[0] = snedges[1] = 0;
            for (int i = 0; i < g.nv; ++i) {
                const int p = bipartition[i];
                rename[i] = snvtxs[p]++;
                snedges[p] += g.xadj[i+1] - g.xadj[i];
            }

            std::array<std::unique_ptr<XADJ_INT[]>, 2> sxadj = { std::make_unique<XADJ_INT[]>(snvtxs[0]+1), std::make_unique<XADJ_INT[]>(snvtxs[1]+1) };
            std::array<std::unique_ptr<int[]>, 2> sadjncy = { std::make_unique<int[]>(snedges[0]), std::make_unique<int[]>(snedges[1]) };
            slabel_out[0] = std::make_unique<int[]>(snvtxs[0]);
            slabel_out[1] = std::make_unique<int[]>(snvtxs[1]);

            #pragma omp parallel
            for (int p = 0; p < 2; ++p){
                #pragma omp for
                for (int i = 1; i < snvtxs[p]+1; ++i){
                    sxadj[p][i] = 0;
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < g.nv; ++i) {
                const int p = bipartition[i];
                const int i_p = rename[i];
                for ( XADJ_INT k = g.xadj[i]; k < g.xadj[i+1]; ++k){
                    const int j = g.adjncy[k];
                    if ( bipartition[j] == p ) {
                        sxadj[p][i_p+1]++; // first count edges for each vertex
                    }
                }
            }

            // finish making xadj
            for (int p = 0; p < 2; ++p){
                sxadj[p][0] = 0;
                for (int i = 0; i < snvtxs[p]; ++i){
                    sxadj[p][i+1] += sxadj[p][i];
                }            
            }

            #pragma omp parallel for
            for ( int i = 0; i < g.nv; ++i) {
                const int p = bipartition[i];
                const int i_p = rename[i];
                if ( !blist.is_boundary(i) ) { /* This is an interior vertex */
                    XADJ_INT l = sxadj[p][i_p];
                    for (XADJ_INT k= g.xadj[i]; k < g.xadj[i+1]; ++k) {
                        sadjncy[p][l] = rename[g.adjncy[k]];
                        l++;
                    }
                } else {
                    XADJ_INT l = sxadj[p][i_p];
                    for (XADJ_INT k = g.xadj[i]; k < g.xadj[i+1]; ++k) {
                        const int j = g.adjncy[k];
                        if ( bipartition[j] == p ) {
                            sadjncy[p][l] = rename[j];
                            l++;
                        }
                    }
                }

                slabel_out[p][i_p] = label[i];
            }

            Graph<XADJ_INT> lgraph(snvtxs[0], std::move(sxadj[0]), std::move(sadjncy[0]));
            Graph<XADJ_INT> rgraph(snvtxs[1], std::move(sxadj[1]), std::move(sadjncy[1]));
            return {std::move(lgraph), std::move(rgraph)};
        }

    // // SplitGraphPart of metis-5.1.0/libmetis/pmetis.c
    // template<class XADJ_INT>
    // std::pair<Graph<XADJ_INT>, Graph<XADJ_INT>> split_graph_old(const Graph<XADJ_INT> &g, const int* const label, const int* const bipartition, std::vector<int> &label0, std::vector<int> &label1){
    //     int snvtxs[2], snedges[2];
    //     auto rename = std::make_unique<int[]>(g.nv);

    //     snvtxs[0] = snvtxs[1] = snedges[0] = snedges[1] = 0;
    //     for (int i=0; i < g.nv; ++i) {
    //         const int p = bipartition[i];
    //         rename[i] = snvtxs[p]++;
    //         snedges[p] += g.xadj[i+1] - g.xadj[i];
    //     }

    //     std::array<std::unique_ptr<int[]>, 2> sxadj = { std::make_unique<int[]>(snvtxs[0]+1), std::make_unique<int[]>(snvtxs[1]+1) };
    //     // std::array<std::unique_ptr<int[]>, 2> slabel = { std::make_unique<int[]>(snvtxs[0]), std::make_unique<int[]>(snvtxs[1]) };
    //     std::array<std::unique_ptr<int[]>, 2> sadjncy = { std::make_unique<int[]>(snedges[0]), std::make_unique<int[]>(snedges[1]) };
    //     label0.resize(snvtxs[0]);
    //     label1.resize(snvtxs[1]);

    //     snvtxs[0] = snvtxs[1] = snedges[0] = snedges[1] = 0;
    //     sxadj[0][0] = sxadj[1][0] = 0;
    //     for ( int i=0; i < g.nv; ++i) {
    //         int mypart = bipartition[i];

    //         XADJ_INT istart = g.xadj[i];
    //         XADJ_INT iend = g.xadj[i+1];
    //         // if (bndptr[i] == -1) { /* This is an interior vertex */
    //         //     auxadjncy = sadjncy[mypart] + snedges[mypart] - istart;
    //         //     auxadjwgt = sadjwgt[mypart] + snedges[mypart] - istart;
    //         //     for(j=istart; j<iend; j++) {
    //         //         auxadjncy[j] = g.adjncy[j];
    //         //     }
    //         //     snedges[mypart] += iend-istart;
    //         // }
    //         // else {
    //             int *auxadjncy = sadjncy[mypart].get();
    //             int l = snedges[mypart];
    //             for (int j = istart; j < iend; j++) {
    //                 const int k = g.adjncy[j];
    //                 if ( bipartition[k] == mypart ) {
    //                     auxadjncy[l] = k;
    //                     l++;                        
    //                 }
    //             }
    //             snedges[mypart] = l;
    //         // }

    //         if ( mypart == 0 ){
    //             label0[snvtxs[mypart]]   = label[i];
    //         }else{
    //             label1[snvtxs[mypart]]   = label[i];
    //         }
    //         sxadj[mypart][++snvtxs[mypart]]  = snedges[mypart];
    //     }

    //     for (int mypart=0; mypart < 2; ++mypart) {
    //         const XADJ_INT iend = sxadj[mypart][snvtxs[mypart]];
    //         int *auxadjncy = sadjncy[mypart].get();
    //         for (XADJ_INT i=0; i<iend; i++) 
    //             auxadjncy[i] = rename[auxadjncy[i]];
    //     }

    //     Graph<XADJ_INT> lgraph(snvtxs[0], std::move(sxadj[0]), std::move(sadjncy[0]));
    //     Graph<XADJ_INT> rgraph(snvtxs[1], std::move(sxadj[1]), std::move(sadjncy[1]));
    //     return std::make_pair<Graph<XADJ_INT>, Graph<XADJ_INT>>(std::move(lgraph), std::move(rgraph));
    // }

   };
}