#include<params.hpp>
#include<graph.hpp>

#include "gapbs/bfs.h"

namespace Sppart {
    template<class XADJ_INT, class FLOAT>
    void random_source_sssp(const Graph<XADJ_INT>& g, FLOAT* const dists, const InputParams& params, std::mt19937& rand_engine, OutputInfo& info){
        const size_t nv = g.nv; // use size_t type to prevent possible overflow when allocating unique_ptrs.
        const int n_dims = params.n_dims;
        std::vector<int> sources(n_dims);
        std::uniform_int_distribution<> ud(0, nv-1);

        // get random n_dims vertices (sampling without replacement)
        for (int i = 0; i < n_dims; ++i){
            bool ok = false;
            int i_vertex;
            while (!ok){
                i_vertex = ud(rand_engine);
                ok = true;
                for (int j = 0; j < i; ++j){
                    ok &= i_vertex != sources[j];
                }
            }
            sources[i] = i_vertex;
        }

        info.time_spectral_bfs += timeit([&]{
            if ( params.bfs_alg == 0 ) {
                const int nested = omp_get_nested();
                omp_set_nested(1);
                const int outer_n_threads = std::min(n_dims, omp_get_max_threads());
                const int inner_n_threads = std::max(omp_get_max_threads()/n_dims, 1);
                #pragma omp parallel for num_threads(outer_n_threads)
                for (int i = 0; i < n_dims; ++i){
                    const int s = sources[i];
                    bfs_mt_for(g.nv, g.xadj, g.adjncy, s, &dists[i*nv], inner_n_threads);
                }
                omp_set_nested(nested);
            }
            else if ( params.bfs_alg == 1 ) {
                const int nested = omp_get_nested();
                omp_set_nested(1);
                const int outer_n_threads = std::min(n_dims, omp_get_max_threads());
                const int inner_n_threads = std::max(omp_get_max_threads()/n_dims, 1);
                #pragma omp parallel for num_threads(outer_n_threads)
                for (int i = 0; i < n_dims; ++i){
                    const int s = sources[i];
                    bfs_mt_for_by_func(g.nv, g.xadj, g.adjncy, s, &dists[i*nv], inner_n_threads);
                }
                omp_set_nested(nested);
            }
            else if ( params.bfs_alg == 2 ) {
                for (int i = 0; i < n_dims; ++i){
                    const int s = sources[i];
                    bfs_mt_for(g.nv, g.xadj, g.adjncy, s, &dists[i*nv], omp_get_max_threads());
                }
            }
            else if ( params.bfs_alg == 3 ) {
                #pragma omp parallel for
                for (int i = 0; i < n_dims; ++i){
                    const int s = sources[i];
                    bfs_std_queue(g.nv, g.xadj, g.adjncy, s, &dists[i*nv]);
                }
            }
            else if ( params.bfs_alg == 4 ) {
                for (int i = 0; i < n_dims; ++i){
                    const int s = sources[i];
                    bfs_mt_stack(g.nv, g.xadj, g.adjncy, s, &dists[i*nv]);
                }
            } else if ( params.bfs_alg == 5 ) {
                msbfs_bitmap(g.nv, g.xadj, g.adjncy, n_dims, sources.data(), dists);
            }else if ( params.bfs_alg == 6 ) {
                for (int i = 0; i < n_dims; ++i){
                    const int s = sources[i];
                    bfs_mt_for_bitmap(g.nv, g.xadj, g.adjncy, s, &dists[i*nv]);
                }
            }else if ( params.bfs_alg == 7 ) {
                for (int i = 0; i < n_dims; ++i){
                    const int s = sources[i];
                    bfs_mt_for_redundant(g.nv, g.xadj, g.adjncy, s, &dists[i*nv]);
                }
            } else{
                printf("no such bfs_alg\n");
                std::terminate();                    
            }
        });
    }

    template<class XADJ_INT, class FLOAT>
    void k_center_source_sssp(const Graph<XADJ_INT>& g, FLOAT* const dists, const InputParams& params, std::mt19937& rand_engine, OutputInfo& info){
        const size_t nv = g.nv; // use size_t type to prevent possible overflow when allocating unique_ptrs.
        const int n_dims = params.n_dims;
        std::vector<int> sources(n_dims);
        std::uniform_int_distribution<> ud(0, nv-1);
        auto dd = std::make_unique<FLOAT[]>(nv);

        #pragma omp parallel for
        for (int i = 0; i < nv; ++i){
            dd[i] = 2.0*nv; // means infinity
        }

        int source = ud(rand_engine);
        for (int i = 0; i < n_dims; ++i){
            info.time_spectral_bfs += timeit([&]{
                if ( params.bfs_alg == 0 ) {
                    bfs_mt_for(g.nv, g.xadj, g.adjncy, source, &dists[i*nv], omp_get_max_threads());
                } else if ( params.bfs_alg == 1 ) {
                    gapbs::DOBFS(g, source, &dists[i*nv]);
                } else {
                    printf("no such bfs_alg\n");
                    std::terminate();                    
                }
            });
            if ( i == n_dims - 1) break;
            #pragma omp parallel for
            for (int j = 0; j < nv; ++j){
                dd[j] = std::min(dd[j], dists[i*nv+j]);
            }
            int max_idx = 0;
            FLOAT max_val = dd[0];
            for (int j = 1; j < nv; ++j){
                if ( dd[j] > max_val ){
                    max_val = dd[j];
                    max_idx = j;
                }
            }
            source = max_idx; 
        }
    }

    // compute n_dims vectors of single source shortest path distance (SSSPD)
    // dists: g.nv x n_dims matrix (column-major) contains SSSPD vectors in its columns
    template<class XADJ_INT, class FLOAT>
    void compute_ssspd_basis(const Graph<XADJ_INT>& g, FLOAT* const dists, const InputParams& params,  std::mt19937& rand_engine, OutputInfo& info){
        if ( params.src_alg == 0 ){
            random_source_sssp(g, dists, params, rand_engine, info);
        }else if ( params.src_alg == 1 ){
            k_center_source_sssp(g, dists, params, rand_engine, info);
        } else {
            printf("no such src_alg\n");
            std::terminate();
        }
    }

} // namespace