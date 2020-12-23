// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef INCLUDE_SPPART_GRAPH_HPP
#define INCLUDE_SPPART_GRAPH_HPP
 
#include<cstdint>
#include<memory>

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

        Graph(const Graph&) = delete;
        Graph& operator=(const Graph&) = delete;

        Graph(Graph&&) = default;
        Graph& operator=(Graph&& g) = default;

    };
}

#endif // INCLUDE_SPPART_GRAPH_HPP
