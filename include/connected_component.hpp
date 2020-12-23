// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#pragma once

#include <cassert>
#include <vector>
#include <queue>

//#include<timers.hpp>
#include<graph.hpp>

namespace Sppart{

    template<class XADJ_INT>
    int64_t compute_cut(const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int* const partition){
        int64_t cut = 0;
        for (int i = 0; i < nv; ++i){
            for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                const int j = adjncy[k];
                if ( partition[j] != partition[i] ){
                    cut += 1;
                }
            }
        }
        assert(cut % 2 == 0);
        cut /= 2;
        return cut;
    }

    template<class XADJ_INT>
    double compute_maxbal(const int nparts, const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int* const partition){
        int ave;
        if ( nv % nparts == 0 ){
            ave = nv / nparts;
        }else{
            ave = nv / nparts + 1;
        }
        std::vector<int> part_weights(nparts, 0);
        for (int i = 0; i < nv; ++i){
            part_weights[partition[i]] += 1;
        }
        int max_part_weight = 0;
        for (int p = 0; p < nparts; ++p){
            max_part_weight = std::max(part_weights[p], max_part_weight);
        }
        return static_cast<double>(max_part_weight) / static_cast<double>(ave);
    }


    template<class XADJ_INT>
    void duplicate_graph_without_diagonals(const int nv, const std::vector<XADJ_INT> &in_xadj, const std::vector<int> &in_adjncy, std::vector<XADJ_INT> &out_xadj, std::vector<int> &out_adjncy){
        XADJ_INT total_cnt = 0;
        out_xadj.resize(nv+1);
        out_adjncy.resize(in_xadj[nv]);
        out_xadj[0] = 0;
        for (int i = 0; i < nv; ++i){
            for (XADJ_INT k = in_xadj[i]; k < in_xadj[i+1]; ++k){
                const int j = in_adjncy[k];
                if ( i == j ){ // Diagonal entry
                    continue;
                }
                out_adjncy[total_cnt] = j;
                total_cnt++;                        
            }
            out_xadj[i+1] = total_cnt;
        }
        out_adjncy.resize(total_cnt);

        // check if there is no diagonal entries
        // for (int i = 0; i < nv; ++i){
        //     for (XADJ_INT k = out_xadj[i]; k < out_xadj[i+1]; ++k){
        //         const int j = out_adjncy[k];
        //         assert (i != j);
        //     }
        // }

        return;
    }

    template<class XADJ_INT>
    bool check_connected(const int nv, const XADJ_INT* const xadj, const int* const adjcny){
        std::queue<int> que;
        std::vector<bool> visited(nv, false);
        que.push(0); // Source is the 0-th vertex
        visited[0] = true;
        int cnt = 1;
        while( !que.empty() ){
            const int i = que.front();
            que.pop();
            for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                const int j = adjcny[k];
                if ( visited[j] ) continue;
                visited[j] = true;
                que.push(j);
                cnt++;
            }
        }
        assert(cnt <= nv);

        return cnt == nv;
    }

    template<class XADJ_INT>
    void get_largest_connected_component(const int in_nv, const std::vector<XADJ_INT> &in_xadj, const std::vector<int> &in_adjncy, int &out_nv, std::vector<XADJ_INT> &out_xadj, std::vector<int> &out_adjncy){
        constexpr int NOT_VISITED = -1;
        std::vector<int> component(in_nv, NOT_VISITED);
        int cnt = 0;
        int source = 0;
        int current_component = 0;

        while (cnt < in_nv ){
            // BFS
            std::queue<int> que;
            que.push(source);
            component[source] = current_component;
            cnt++;
            while( !que.empty() ){
                const int i = que.front();
                que.pop();
                for (XADJ_INT k = in_xadj[i]; k < in_xadj[i+1]; ++k){
                    const int j = in_adjncy[k];
                    if ( component[j] != NOT_VISITED ) continue;
                    component[j] = current_component;
                    que.push(j);
                    cnt++;
                }
            }

            // Search next source
            while ( ++source < in_nv){
                if ( component[source] == NOT_VISITED ){
                    current_component++;
                    break;
                }                
            }
        }
        assert(cnt <= in_nv);

        const int num_components = current_component + 1;
        std::vector<int> vertex_counts(num_components, 0);
        for (int i = 0; i < in_nv; ++i){
            vertex_counts[component[i]]++;
        }

        int max_count = 0;
        int largest_component = -1;
        for (int i = 0; i < num_components; ++i){
            if ( vertex_counts[i] > max_count ){
                max_count = vertex_counts[i];
                largest_component = i;
            }
        }
        assert(0 <= largest_component && largest_component <= num_components);

        std::vector<bool> in_largest_component(in_nv);
        
        for (int i = 0; i < in_nv; ++i){
            in_largest_component[i] = component[i] == largest_component;
        }

        cnt = 0;
        std::vector<int> new_index(in_nv, -1);
        for (int i = 0; i < in_nv; ++i){
            if (in_largest_component[i]) {
                new_index[i] = cnt;
                cnt++;
            }
        }
        const int component_size = cnt;
        out_xadj.resize(component_size+1);
        out_adjncy.resize(in_xadj[in_nv]);

        // Create graph of largest connected component
        int row_cnt = 0;
        int total_cnt = 0;
        out_xadj[0] = 0;
        for (int i = 0; i < in_nv; ++i){
            if (in_largest_component[i]){
                for (XADJ_INT k = in_xadj[i]; k < in_xadj[i+1]; ++k){
                    const int j = in_adjncy[k];
                    if (in_largest_component[i]){
                        assert(new_index[j] != -1);
                        out_adjncy[total_cnt] = new_index[j];
                        total_cnt++;        
                    }
                }
                row_cnt++;
                out_xadj[row_cnt] = total_cnt;
            }
        }
        out_nv = row_cnt;
        out_xadj.resize(out_nv+1);
        out_adjncy.resize(total_cnt);

        return;
    }

    template<class XADJ_INT>
    void preprocess_graph(const int in_nv, const std::vector<XADJ_INT> &in_xadj, const std::vector<int> &in_adjncy, int &out_nv, std::vector<XADJ_INT> &out_xadj, std::vector<int> &out_adjncy){
        std::vector<XADJ_INT> tmp_xadj;
        std::vector<int> tmp_adjncy;
        duplicate_graph_without_diagonals(in_nv, in_xadj, in_adjncy, tmp_xadj, tmp_adjncy);
        get_largest_connected_component(in_nv, tmp_xadj, tmp_adjncy, out_nv, out_xadj, out_adjncy);
    }


}// namespace