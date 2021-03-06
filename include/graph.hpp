// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef INCLUDE_SPPART_GRAPH_HPP
#define INCLUDE_SPPART_GRAPH_HPP
 
#include<cstdint>
#include<memory>
#include<util.hpp>

namespace Sppart {
    // Class for unweighted undirected graph
    template<class XADJ_INT>
    class Graph {
    
    public:
        const int nv;
        const XADJ_INT* const xadj; // this should be declared before up_xadj
        const int* const adjncy; // this should be declared before up_adjncy
        const bool managed;

    protected:
        std::unique_ptr<const XADJ_INT[]> up_xadj;
        std::unique_ptr<const int[]> up_adjncy;

    public:
        Graph(const int nv, const XADJ_INT* const xadj, const int* const adjncy)
        : nv(nv), xadj(xadj), adjncy(adjncy), managed(false)
        {
            assert(nv >= 0);
        }

        Graph(const int nv, std::unique_ptr<const XADJ_INT[]>&& up_xadj, std::unique_ptr<const int[]>&& up_adjncy)
        : nv(nv), xadj(up_xadj.get()), adjncy(up_adjncy.get()),  managed(true), up_xadj(std::move(up_xadj)), up_adjncy(std::move(up_adjncy))
        {
            assert(nv >= 0);
        }

        void release(){
            up_xadj.release();
            up_adjncy.release();
        }

        Graph<XADJ_INT> create_subgraph(std::function<bool(int)> is_subgraph_vertex) const {
            auto sub_index = create_up_array<int>(nv);

            int vertex_cnt = 0;
            XADJ_INT nnz_cnt = 0;            
            for (int i = 0; i < nv; ++i){
                if ( is_subgraph_vertex(i) ){
                    for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                        const int j = adjncy[k];
                        if ( is_subgraph_vertex(j) ){                            
                            nnz_cnt++;
                        }
                    }
                    sub_index[i] = vertex_cnt;
                    vertex_cnt++;
                }
            }

            const int sub_nv = vertex_cnt;
            auto sub_xadj = create_up_array<int>(sub_nv+1);
            auto sub_adjncy = create_up_array<XADJ_INT>(nnz_cnt);            

            vertex_cnt = 0;
            nnz_cnt = 0;
            sub_xadj[0] = 0;
            for (int i = 0; i < nv; ++i){
                if ( is_subgraph_vertex(i) ){
                    for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                        const int j = adjncy[k];
                        if ( is_subgraph_vertex(j) ){
                            sub_adjncy[nnz_cnt] = sub_index[j];
                            nnz_cnt++;
                        }
                    }
                    vertex_cnt++;
                    sub_xadj[vertex_cnt] = nnz_cnt;
                }
            }

            return Graph<XADJ_INT>(sub_nv, std::move(sub_xadj), std::move(sub_adjncy));
        }

        XADJ_INT degree(int v) const {
            return xadj[v+1] - xadj[v];
        }

        Graph(const Graph&) = delete;
        Graph& operator=(const Graph&) = delete;

        Graph(Graph&&) = default;
        Graph& operator=(Graph&& g) = default;

    };
}

#endif // INCLUDE_SPPART_GRAPH_HPP
