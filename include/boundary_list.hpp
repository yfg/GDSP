// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef INCLUDE_SPPART_BOUNDARY_LIST_HPP
#define INCLUDE_SPPART_BOUNDARY_LIST_HPP

#include <cassert>

namespace Sppart {
    class BoundaryList {
    protected:
        const int nv;
        int length;
        std::unique_ptr<int[]> index;
        std::unique_ptr<int[]> pointer;

    public:
        BoundaryList(int nv)
        : nv(nv), length(0), index(new int[nv]), pointer(new int[nv])
        {
            #pragma omp parallel for
            for (int i = 0; i < nv; ++i){
                index[i] = -1;
                pointer[i] = -1;
            }
        }

        void insert(int i_vertex){
            // assert( pointer[i_vertex] == -1 );
            index[length] = i_vertex;
            pointer[i_vertex] = length;
            length++;
        }

        void remove(int i_vertex){
            // assert( pointer[i_vertex] != -1 );
            length--;
            index[pointer[i_vertex]] = index[length];
            pointer[index[length]] = pointer[i_vertex];
            pointer[i_vertex] = -1;
        }

        bool is_boundary(int i_vertex) const {
            return pointer[i_vertex] != -1;
        }

        int get_boundary_at(int i) const {
            // assert(0 <= i && i < length);
            return index[i];
        }

        int get_length() const {
            return length;
        }

        bool empty() const {
            return length == 0;
        }

        void clear(){
            length = 0;
            #pragma omp parallel for
            for (int i = 0; i < nv; ++i){
                index[i] = -1;
                pointer[i] = -1;
            }
        }

        BoundaryList(const BoundaryList&) = delete;
        BoundaryList& operator=(const BoundaryList&) = delete;
        BoundaryList(BoundaryList&&) = delete;
        BoundaryList& operator=(BoundaryList&&) = delete;
    };

}

#endif // INCLUDE_SPPART_BOUNDARY_LIST_HPP
