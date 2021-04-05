// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef SPPART_BFS_HPP
#define SPPART_BFS_HPP

#include <cassert>
#include <vector>
#include <queue>
#include <functional>

namespace Sppart{
    template<class XADJ_INT, class FT> // assuming FT is float or double
    void bfs_std_queue(const int nv, const XADJ_INT* const xadj, const int* const adjcny, const int source, FT* const dists){
        std::queue<int> visit, visit_next; // using std::stack is also OK
        for (int i = 0; i < nv; ++i){
            dists[i] = -1.0;
        }
        visit.push(source); // Source is the 0-th vertex
        dists[source] = 0.0;
        double level = 1.0;

        while( !visit.empty() ){
            do {
                const int i = visit.front();
                // const int i = visit.top();
                visit.pop();
                for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                    const int j = adjcny[k];
                    if ( dists[j] != -1.0 ) continue;
                    dists[j] = level;
                    visit_next.push(j);
                }
            } while ( !visit.empty() );
            visit.swap(visit_next);
            level += 1.0;
        }
    }

    template<class XADJ_INT, class FT> // assuming FT is float or double
    void bfs_mt_for(const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int source, FT* const dists, const int n_threads){
        auto up_visit = create_up_array<int>(nv); // Array for stack
        auto up_visit_next = create_up_array<int>(nv); // Array for stack
        auto up_is_visited = create_up_array<int>(nv);
        int* visit = up_visit.get();
        int* visit_next = up_visit_next.get();
        int* is_visited = up_is_visited.get();
        int visit_head = 0;
        int visit_next_head = 0;

        #pragma omp parallel for
        for (int i = 0; i < nv; ++i){
            is_visited[i] = 0;
        }

        dists[source] = 0.0;
        visit[visit_head] = source;
        visit_head++;
        is_visited[source] = 1;

        #pragma omp parallel num_threads(n_threads)
        {
            FT level = 1.0;
            bool is_next_empty = false;
            do {
                #pragma omp for
                for ( int pos = 0; pos < visit_head; ++pos) {
                    const int i = visit[pos];
                    for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                        int ret;
                        const int j = adjncy[k];
                        if ( is_visited[j] == 1 ) continue;
                        #pragma omp atomic capture
                        {ret = is_visited[j]; is_visited[j] |= 1;}
                        if ( ret == 1 ) continue;
                        #pragma omp atomic capture
                        ret = visit_next_head++;
                        visit_next[ret] = j;
                        dists[j] = level;
                    }
                }
                #pragma omp barrier
                is_next_empty = visit_next_head == 0;
                #pragma omp barrier
                #pragma omp single
                {
                    visit_head = visit_next_head;
                    visit_next_head = 0;
                    std::swap(visit, visit_next);
                }
                level += 1.0;
            } while (!is_next_empty);
        }
        return;
    }

    template<class XADJ_INT, class FT>
    int bfs_mt_for_func(const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int source, std::function<void(int, FT)> func, const int n_threads){
        auto up_visit = create_up_array<int>(nv); // Array for stack
        auto up_visit_next = create_up_array<int>(nv); // Array for stack
        auto up_is_visited = create_up_array<int>(nv);
        int* visit = up_visit.get();
        int* visit_next = up_visit_next.get();
        int* is_visited = up_is_visited.get();
        int visit_head = 0;
        int visit_next_head = 0;

        #pragma omp parallel for
        for (int i = 0; i < nv; ++i){
            is_visited[i] = 0;
        }

        func(source, 0.0);
        visit[visit_head] = source;
        visit_head++;
        is_visited[source] = 1;

        int vertex_count = 1;

        #pragma omp parallel num_threads(n_threads)
        {
            FT level = 1.0;
            bool is_next_empty = false;
            do {
                #pragma omp for
                for ( int pos = 0; pos < visit_head; ++pos) {
                    const int i = visit[pos];
                    for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                        int ret;
                        const int j = adjncy[k];
                        if ( is_visited[j] == 1 ) continue;
                        #pragma omp atomic capture
                        {ret = is_visited[j]; is_visited[j] |= 1;}
                        if ( ret == 1 ) continue;
                        #pragma omp atomic capture
                        ret = visit_next_head++;
                        visit_next[ret] = j;
                        func(j, level);
                    }
                }
                #pragma omp barrier
                is_next_empty = visit_next_head == 0;
                #pragma omp barrier
                #pragma omp single
                {
                    vertex_count += visit_next_head;
                    visit_head = visit_next_head;
                    visit_next_head = 0;
                    std::swap(visit, visit_next);
                }
                level += 1.0;
            } while (!is_next_empty);
        }
        return vertex_count;
    }

    template<class XADJ_INT, class FT>
    void bfs_mt_for_by_func(const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int source, FT* const dists, const int n_threads){
        auto f = [&dists](int v, FT level){ dists[v] = level; };        
        bfs_mt_for_func<XADJ_INT, FT> (nv, xadj, adjncy, source, f, n_threads);
    }

    template<class XADJ_INT, class FT> // assuming FT is float or double
    void bfs_mt_for_bitmap(const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int source, FT* const dists){
        constexpr int NB = 8*sizeof(int); // Number of bits
        auto up_visit = create_up_array<int>(nv); // Array for stack
        auto up_visit_next = create_up_array<int>(nv); // Array for stack
        auto up_is_visited = create_up_array<int>(nv/NB+1);
        int* visit = up_visit.get();
        int* visit_next = up_visit_next.get();
        int* is_visited = up_is_visited.get();
        int visit_head = 0;
        int visit_next_head = 0;

        #pragma omp parallel for
        for (int i = 0; i < nv/NB+1; ++i){
            is_visited[i] = 0;
        }

        dists[source] = 0.0;
        visit[visit_head] = source;
        visit_head++;
        is_visited[source/NB] = 1 << (source%NB);

        #pragma omp parallel
        {

        FT level = 1.0;
        bool is_next_empty = false;
        do {
            #pragma omp for
            for ( int pos = 0; pos < visit_head; ++pos) {
                const int i = visit[pos];
                for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                    int ret;
                    const int j = adjncy[k];
                    const int p = j/NB;
                    const int flag = 1 << j%NB;
                    if ( is_visited[p] & flag ) continue;
                    #pragma omp atomic capture
                    {ret = is_visited[p]; is_visited[p] |= flag;}
                    if ( ret & flag ) continue;
                    #pragma omp atomic capture
                    ret = visit_next_head++;
                    visit_next[ret] = j;
                    dists[j] = level;
                }
             }
            #pragma omp barrier
            is_next_empty = visit_next_head == 0;
            #pragma omp barrier
            #pragma omp single
            {
                visit_head = visit_next_head;
                visit_next_head = 0;
                std::swap(visit, visit_next);
            }
            level += 1.0;
        } while (!is_next_empty);
        }
        return;
    }

    template<class XADJ_INT, class FT> // assuming FT is float or double
    void bfs_mt_for_redundant(const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int source, FT* const dists){
        auto up_visit = create_up_array<int>(nv); // Array for stack
        auto up_visit_next = create_up_array<int>(nv); // Array for stack
        auto up_is_visited = create_up_array<int>(nv);
        int* visit = up_visit.get();
        int* visit_next = up_visit_next.get();
        int* is_visited = up_is_visited.get();
        int visit_head = 0;
        int visit_next_head = 0;

        #pragma omp parallel for
        for (int i = 0; i < nv; ++i){
            is_visited[i] = 0;
        }

        dists[source] = 0.0;
        visit[visit_head] = source;
        visit_head++;
        is_visited[source] = 1;

        #pragma omp parallel
        {
            FT level = 1.0;
            bool is_next_empty = false;
            do {
                #pragma omp for
                for ( int pos = 0; pos < visit_head; ++pos) {
                    const int i = visit[pos];
                    for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                        const int j = adjncy[k];
                        if ( is_visited[j] == 1 ) continue;
                        is_visited[j] = 1;
                        int ret;
                        #pragma omp atomic capture
                        ret = visit_next_head++;
                        visit_next[ret] = j;
                        dists[j] = level;
                    }
                }
                #pragma omp barrier
                is_next_empty = visit_next_head == 0;
                #pragma omp barrier
                #pragma omp single
                {
                    visit_head = visit_next_head;
                    visit_next_head = 0;
                    std::swap(visit, visit_next);
                }
                level += 1.0;
            } while (!is_next_empty);
        }
        return;
    }

    template<class XADJ_INT, class FT> // assuming FT is float or double
    void bfs_mt_stack(const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int source, FT* const dists){
        auto visit = create_up_array<int>(nv); // Array for stack
        auto visit_next = create_up_array<int>(nv); // Array for stack
        auto is_visited = create_up_array<int>(nv);
        int visit_head = 0;
        int visit_next_head = 0;

        #pragma omp parallel for
        for (int i = 0; i < nv; ++i){
            is_visited[i] = 0;
        }

        dists[source] = 0.0;
        visit[visit_head] = source;
        visit_head++;
        is_visited[source] = 1;

        #pragma omp parallel
        {
            FT level = 1.0;
            bool is_next_empty = false;
            do {
                int i;
                int ret;
                while (1) {
                    // Pop operation for stack
                    #pragma omp atomic capture
                    ret = --visit_head;

                    if ( ret < 0 ) break; // stack is empty
                    i = visit[ret];

                    for (XADJ_INT j = xadj[i]; j < xadj[i+1]; ++j){
                        const int k = adjncy[j];
                        if ( is_visited[k] == 1 ) continue;
                        #pragma omp atomic capture
                        {ret = is_visited[k]; is_visited[k] |= 1;}
                        if ( ret == 1 ) continue;

                        // Push operation for stack
                        #pragma omp atomic capture
                        ret = visit_next_head++;

                        // assert(ret < n);
                        visit_next[ret] = k;
                        dists[k] = level;
                    }
                }
                #pragma omp barrier
                is_next_empty = visit_next_head == 0;
                #pragma omp barrier
                level += 1.0;
                #pragma omp single // wait version
                {
                    visit_head = visit_next_head;
                    visit_next_head = 0;
                    std::swap(visit, visit_next);
                }
            } while (!is_next_empty);
        }
        return;
    }

    template<class XADJ_INT, class FT> // assuming FT is float or double
    void msbfs_bitmap(const int nv, const XADJ_INT* const xadj, const int* const adjncy, const int n_source, const int* const sources, FT* const dists){
        constexpr int NB = 8*sizeof(int); // Number of bits
        auto visit = create_up_array<int>(nv); // Array for stack
        auto visit_next = create_up_array<int>(nv); // Array for stack
        auto seen_flag = create_up_array<int>(nv); // Treated as bit field
        auto visit_flag = create_up_array<int>(nv); // Treated as bit field
        auto visit_next_flag = create_up_array<int>(nv); // Treated as bit field
        int visit_head = 0;
        int visit_next_head = 0;

        #pragma omp parallel for
        for (int i = 0; i < nv; ++i){
            seen_flag[i] = 0;
            visit_flag[i] = 0;
            visit_next_flag[i] = 0;
        }

        for (int i = 0; i < n_source; ++i){
            int s = sources[i];
            seen_flag[s] = 1 << i;
            visit_flag[s] = 1 << i;
            dists[i*nv + s] = 0;
            visit[visit_head] = s;
            visit_head++;
        }

        #pragma omp parallel
        {
            int level = 1;
            bool is_next_empty = false;
            do {
                int ret;
                #pragma omp for
                for ( int pos = 0; pos < visit_head; ++pos) {
                    const int i = visit[pos];
                    if ( visit_flag[i] == 0 ) continue;

                    for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                        const int j = adjncy[k];

                        int diff = visit_flag[i] & ~seen_flag[j];
                        if ( diff == 0 ) continue;
                        #pragma omp atomic capture
                        {ret = seen_flag[j] ; seen_flag[j] |= diff;}
                        diff = visit_flag[i] & ~ret;
                        if ( diff == 0 ) continue;
                        visit_next_flag[j] |= diff;
                        // Push operation
                        #pragma omp atomic capture
                        ret = visit_next_head++;
                        visit_next[ret] = j;
                        for (int ii = 0; ii < n_source; ++ii){ 
                            if ( diff & (1 << ii) ) {
                                dists[ii*nv + j] = level;
                            }
                        }
                    }

                    visit_flag[i] = 0;
                }
                #pragma omp barrier
                is_next_empty = visit_next_head == 0;
                #pragma omp barrier
                level++;
                #pragma omp single // wait version
                {
                    visit_head = visit_next_head;
                    visit_next_head = 0;
                    std::swap(visit, visit_next);
                    std::swap(visit_flag, visit_next_flag);
                }            
            } while (!is_next_empty);
        }

        return;
    }

} // namespace

#endif // SPPART_BFS_HPP
