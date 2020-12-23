// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef INCLUDE_SPPART_LINEAR_ALGEBRA_HPP
#define INCLUDE_SPPART_LINEAR_ALGEBRA_HPP

#include<cassert>
#include<vector>

#include<blas_lapack.hpp>

namespace Sppart{
    template<class FT> // assuming FT is float or double
    void make_sum_to_zero(const int nv, const int n_vecs, FT* const X){
        const double w = 1.0 / std::sqrt(static_cast<FT>(nv));

        for (size_t m = 0; m < n_vecs; ++m){ // use size_t type to prevent possible overflow of m*nv
            double sum = 0.0;
            #pragma omp parallel for reduction (+:sum)
            for (int i = 0; i < nv; ++i){
                sum += w*X[m*nv+i];
            }
            // sum *= w;
            #pragma omp parallel for
            for (int i = 0; i < nv; ++i){
                X[m*nv+i] -= sum*w;
            }
        }
    }

    template<class FT> // assuming FT is float or double
    void orthonormalize(const int m, const int n, FT* const A){
        assert(m >= n);
        std::vector<FT> tau(n);
        FT tmp;
        int lwork = -1, info;    

        // working space query
        lapack::geqrf(&m, &n, A, &m, tau.data(), &tmp, &lwork, &info);
        lwork = static_cast<int>(tmp);     
        std::vector<FT> work(lwork);
        // real run
        lapack::geqrf(&m, &n, A, &m, tau.data(), work.data(), &lwork, &info);

        lwork = -1;
        // working space query
        lapack::ungqr(&m, &n, &n, A, &m, tau.data(), &tmp, &lwork, &info);
        lwork = static_cast<int>(tmp);     
        work.resize(lwork);
        // real run
        lapack::ungqr(&m, &n, &n, A, &m, tau.data(), work.data(), &lwork, &info);

        return;
    }

    template<class FT> // assuming FT is float or double
    void calc_XtY(const int m, const int n, const FT* const X, const FT* const Y, FT* const ret){
        assert(m >= n);
        const char transa = 'T';
        const char transb = 'N';
        const FT alpha = 1.0;
        const FT beta = 0.0;
        blas::gemm(&transa, &transb, &n, &n, &m, &alpha, X, &m, Y, &m, &beta, ret, &n);        
    }

    // Compute Y = L*X;
    template<class XADJ_INT, class FT> // assuming FT is float or double
    void mult_laplacian_naive(const int nv, const int n_vecs, const XADJ_INT* const xadj, const int* const adjcny, const FT* const X, FT* const Y){
        #pragma omp parallel
        for (int64_t m = 0; m < n_vecs; ++m){
            #pragma omp for
            for (int i = 0; i < nv; ++i){
                Y[m*nv + i] = 0.0;
                for (XADJ_INT k = xadj[i]; k < xadj[i+1]; ++k){
                    const int j = adjcny[k];
                    Y[m*nv + i] += X[m*nv + i] - X[m*nv + j];
                }
            }
        }
    }

    template<class FT> // assuming FT is float or double
    void calc_eigvecs(const int n, FT* const A){
        const char jobz = 'V';
        const char uplo = 'U';
        int info;
        FT tmp, dummy;
        int lwork = -1;
        std::vector<FT> eigval(n);
        // working space query
        lapack::heev(&jobz, &uplo, &n, A, &n, eigval.data(), &tmp, &lwork, &dummy, &info);
        lwork = static_cast<int>(tmp);
        std::vector<FT> work(lwork);
        lapack::heev(&jobz, &uplo, &n, A, &n, eigval.data(), work.data(), &lwork, &dummy, &info);
        return;
    }
    
    // Y[:,0:k] = X*A[:,0:k] (numpy notation)
    template<class FT> // assuming FT is float or double
    void back_transform(const int m, const int n, const int k, const FT* const X, const FT* const A, FT* const Y){
        const char transa = 'N';
        const char transb = 'N';
        const FT alpha = 1.0;
        const FT beta = 0.0;
        blas::gemm(&transa, &transb, &m, &n, &k, &alpha, X, &m, A, &k, &beta, Y, &m);
        return;
    }

    template<class FT> // assuming FT is float or double
    void fix_sign(const int n, FT* const vec){
        const FT sign = vec[0] > 0 ? 1.0 : -1.0;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i){
            vec[i] *= sign;
        }
        return;
    }

}

#endif // INCLUDE_SPPART_LINEAR_ALGEBRA_HPP