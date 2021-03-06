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
        if ( info != 0 ){
            printf("Error: Lapack geqrf 1 error! info = %d\n", info);
            std::terminate();
        }

        lwork = -1;
        // working space query
        lapack::ungqr(&m, &n, &n, A, &m, tau.data(), &tmp, &lwork, &info);
        lwork = static_cast<int>(tmp);     
        work.resize(lwork);
        // real run
        lapack::ungqr(&m, &n, &n, A, &m, tau.data(), work.data(), &lwork, &info);
        if ( info != 0 ){
            printf("Error: Lapack geqrf 2 error! info = %d\n", info);
            std::terminate();
        }

        return;
    }

    template<class FT> // assuming FT is float or double
    void calc_XtY_gemm(const int m, const int n, const FT* const X, const FT* const Y, FT* const ret){
        // assert(m >= n);
        const char transa = 'T';
        const char transb = 'N';
        const FT alpha = 1.0;
        const FT beta = 0.0;
        blas::gemm(&transa, &transb, &n, &n, &m, &alpha, X, &m, Y, &m, &beta, ret, &n);        
    }

    template<class FT> // assuming FT is float or double
    void calc_XtY_org(const int m, const int n, const FT* const X, const FT* const Y, FT* const ret){
        // Only upper triangular part is computed
        // assert(m >= n);
        for (size_t i = 0; i < n; ++i){
            for (size_t j = i; j < n; ++j){ // upper triangular part
                const FT* const X_ptr = X + i*m;
                const FT* const Y_ptr = Y + j*m;
                FT sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (size_t k = 0; k < m; ++k){
                    sum += X_ptr[k]*Y_ptr[k];
                }
                ret[j*n + i] = sum;
            }
        }
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

    // Standard eigenvalue problem
    template<class FT> // assuming FT is float or double
    void calc_eigvecs_std(const int n, FT* const A){
        const char jobz = 'V';
        const char uplo = 'U';
        int info;
        FT tmp, dummy;
        int lwork = -1;
        std::vector<FT> eigval(n);
        // working space query
        lapack::heev(&jobz, &uplo, &n, A, &n, eigval.data(), &tmp, &lwork, &dummy, &info);
        if ( info != 0 ){
            printf("Error: Lapack heev 1 error! info = %d\n", info);
            std::terminate();
        }

        lwork = static_cast<int>(tmp);
        std::vector<FT> work(lwork);
        lapack::heev(&jobz, &uplo, &n, A, &n, eigval.data(), work.data(), &lwork, &dummy, &info);
        if ( info != 0 ){
            printf("Error: Lapack heev 2 error! info = %d\n", info);
            std::terminate();
        }
        return;
    }
    
    // Generalized eigenvalue problem
    template<class FT> // assuming FT is float or double
    void calc_eigvecs_gen(const int n, FT* const A, FT* const B){
        const int itype = 1;
        const char jobz = 'V';
        const char uplo = 'U';
        int info;
        FT tmp, dummy;
        int lwork = -1;
        std::vector<FT> eigval(n);
        // working space query
        lapack::hegv(&itype, &jobz, &uplo, &n, A, &n, B, &n, eigval.data(), &tmp, &lwork, &dummy, &info);
        if ( info != 0 ){
            printf("Error: Lapack hegv 1 error! info = %d\n", info);
            std::terminate();
        }

        lwork = static_cast<int>(tmp);
        std::vector<FT> work(lwork);
        lapack::hegv(&itype, &jobz, &uplo, &n, A, &n, B, &n, eigval.data(), work.data(), &lwork, &dummy, &info);
        if ( info != 0 ){
            printf("Error: Lapack hegv 2 error! info = %d\n", info);
            std::terminate();
        }
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