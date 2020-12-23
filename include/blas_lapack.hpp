// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef SPPART_BLAS_LAPACK_HPP
#define SPPART_BLAS_LAPACK_HPP

#include <cstdint>
#include <complex>

#include <scalar_traits.hpp>

#ifdef INT
#define SPPART_INT_BACKUP INT
#endif

#ifdef SPPART_BLAS_LAPACK_INT
#define INT SPPART_BLAS_LAPACK_INT
#else
#define INT int
#endif

extern "C"
{
    void sgemm_sppart_blas(const char* transa, const char* transb, const INT* m, const INT* n, const INT* k,
            const float* alpha, const float* A, const INT* lda, const float* B, const INT* ldb,
            const float* beta, float* C, const INT* ldc);

	void dgemm_sppart_blas(const char* transa, const char* transb, const INT* m, const INT* n, const INT* k,
				const double* alpha, const double* A, const INT* lda, const double* B, const INT* ldb,
				const double* beta, double* C, const INT* ldc);

    void cgemm_sppart_blas(const char* transa, const char* transb, const INT* m, const INT* n, const INT* k,
            const std::complex<float>* alpha, const std::complex<float>* A, const INT* lda, const std::complex<float>* B, const INT* ldb,
            const std::complex<float>* beta, std::complex<float>* C, const INT* ldc);

    void zgemm_sppart_blas(const char* transa, const char* transb, const INT* m, const INT* n, const INT* k,
            const std::complex<double>* alpha, const std::complex<double>* A, const INT* lda, const std::complex<double>* B, const INT* ldb,
            const std::complex<double>* beta, std::complex<double>* C, const INT* ldc);

	void ssymm_sppart_blas(const char* side, const char* uplo, const INT* m, const INT* n,
			const float* alpha, const float* A, const INT* lda, const float* B, const INT* ldb,
			const float* beta, float* C, const INT* ldc);

	void dsymm_sppart_blas(const char* side, const char* uplo, const INT* m, const INT* n,
			const double* alpha, const double* A, const INT* lda, const double* B, const INT* ldb,
			const double* beta, double* C, const INT* ldc);

	void chemm_sppart_blas(const char* side, const char* uplo, const INT* m, const INT* n,
			const std::complex<float>* alpha, const std::complex<float>* A, const INT* lda, const std::complex<float>* B, const INT* ldb,
			const std::complex<float>* beta, std::complex<float>* C, const INT* ldc);

	void zhemm_sppart_blas(const char* side, const char* uplo, const INT* m, const INT* n,
			const std::complex<double>* alpha, const std::complex<double>* A, const INT* lda, const std::complex<double>* B, const INT* ldb,
			const std::complex<double>* beta, std::complex<double>* C, const INT* ldc);

	void sgetrf_sppart_lapack( const INT* M, const INT* N,
			  float* A, const INT* lda, INT* ipiv,
			  INT* info );

	void dgetrf_sppart_lapack( const INT* M, const INT* N,
			  double* A, const INT* lda, INT* ipiv,
			  INT* info );

	void cgetrf_sppart_lapack( const INT* M, const INT* N,
			  std::complex<float>* A, const INT* lda, INT* ipiv,
			  INT* info );

	void zgetrf_sppart_lapack( const INT* M, const INT* N,
			  std::complex<double>* A, const INT* lda, INT* ipiv,
			  INT* info );

	void sgetrs_sppart_lapack( const char* trans, const INT* N, const INT* nrhs,
			  float* A, const INT* lda, INT* ipiv,
			  float* B, const INT* ldb, INT* info );

	void dgetrs_sppart_lapack( const char* trans, const INT* N, const INT* nrhs,
			  double* A, const INT* lda, INT* ipiv,
			  double* B, const INT* ldb, INT* info );

	void cgetrs_sppart_lapack( const char* trans, const INT* N, const INT* nrhs,
			  std::complex<float>* A, const INT* lda, INT* ipiv,
			  std::complex<float>* B, const INT* ldb, INT* info );

	void zgetrs_sppart_lapack( const char* trans, const INT* N, const INT* nrhs,
			  std::complex<double>* A, const INT* lda, INT* ipiv,
			  std::complex<double>* B, const INT* ldb, INT* info );

	void csytrf_sppart_lapack( const char* uplo, const INT* n, std::complex<float>* A, const INT* lda, INT* ipiv, std::complex<float>* work, const INT* lwork, INT* info);

	void zsytrf_sppart_lapack( const char* uplo, const INT* n, std::complex<double>* A, const INT* lda, INT* ipiv, std::complex<double>* work, const INT* lwork, INT* info);

	void csytrs_sppart_lapack( const char* uplo, const INT* n, const INT* nrhs, std::complex<float>* A, const INT* lda, INT* ipiv, std::complex<float>* B, const INT* ldb, INT* info);

	void zsytrs_sppart_lapack( const char* uplo, const INT* n, const INT* nrhs, std::complex<double>* A, const INT* lda, INT* ipiv, std::complex<double>* B, const INT* ldb, INT* info);

	void sgesvd_sppart_lapack(const char* job_u, const char* job_v, const INT* m, const INT *n,
			 float* A, const INT* lda, float* S,
			 float* U, const INT* ldu, float* V, const INT* ldv,
			 float* work, INT* lwork, INT* info);

	void dgesvd_sppart_lapack(const char* job_u, const char* job_v, const INT* m, const INT *n,
			 double* A, const INT* lda, double* S,
			 double* U, const INT* ldu, double* V, const INT* ldv,
			 double* work, INT* lwork, INT* info);

	void cgesvd_sppart_lapack(const char* job_u, const char* job_v, const INT* m, const INT *n,
			 std::complex<float>* A, const INT* lda, float* S,
			 std::complex<float>* U, const INT* ldu, std::complex<float>* V, const INT* ldv,
			 std::complex<float>* work, INT* lwork, float* rwork, INT* info);

	void zgesvd_sppart_lapack(const char* job_u, const char* job_v, const INT* m, const INT *n,
			 std::complex<double>* A, const INT* lda, double* S,
			 std::complex<double>* U, const INT* ldu, std::complex<double>* V, const INT* ldv,
			 std::complex<double>* work, INT* lwork, double* rwork, INT* info);

	void sgeev_sppart_lapack(const char* jobvl, const char* jobvr, const INT* n,
			float* A, const INT* lda, float* wr, float* wi,
			float* vl, const INT* ldvl, float* vr, const INT* ldvr,
			float* work, INT* lwork, INT* info);

	void dgeev_sppart_lapack(const char* jobvl, const char* jobvr, const INT* n,
			double* A, const INT* lda, double* wr, double* wi,
			double* vl, const INT* ldvl, double* vr, const INT* ldvr,
			double* work, INT* lwork, INT* info);

	void cgeev_sppart_lapack(const char* jobvl, const char* jobvr, const INT* n,
			std::complex<float>* A, const INT* lda, std::complex<float>* w,
			std::complex<float>* vl, const INT* ldvl, std::complex<float>* vr, const INT* ldvr,
			std::complex<float>* work, INT* lwork, float* rwork, INT* info);

	void zgeev_sppart_lapack(const char* jobvl, const char* jobvr, const INT* n,
			std::complex<double>* A, const INT* lda, std::complex<double>* w,
			std::complex<double>* vl, const INT* ldvl, std::complex<double>* vr, const INT* ldvr,
			std::complex<double>* work, INT* lwork, double* rwork, INT* info);

	void sggev_sppart_lapack(const char* jobvl, const char* jobvr, const INT* n,
			float* A, const INT* lda, float* B, const INT* ldb,
			float* alpha_r, float* alpha_i, float* beta,
			float* vl, const INT* ldvl, float* vr, const INT* ldvr,
			float* work, INT* lwork, INT* info);

	void dggev_sppart_lapack(const char* jobvl, const char* jobvr, const INT* n,
			double* A, const INT* lda, double* B, const INT* ldb,
			double* alpha_r, double* alpha_i, double* beta,
			double* vl, const INT* ldvl, double* vr, const INT* ldvr,
			double* work, INT* lwork, INT* info);

	void cggev_sppart_lapack(const char* jobvl, const char* jobvr, const INT* n,
			std::complex<float>* A, const INT* lda, std::complex<float>* B, const INT* ldb,
			std::complex<float>* alpha, std::complex<float>* beta,
			std::complex<float>* vl, const INT* ldvl, std::complex<float>* vr, const INT* ldvr,
			std::complex<float>* work, INT* lwork, float* rwork, INT* info);

	void zggev_sppart_lapack(const char* jobvl, const char* jobvr, const INT* n,
			std::complex<double>* A, const INT* lda, std::complex<double>* B, const INT* ldb,
			std::complex<double>* alpha, std::complex<double>* beta,
			std::complex<double>* vl, const INT* ldvl, std::complex<double>* vr, const INT* ldvr,
			std::complex<double>* work, INT* lwork, double* rwork, INT* info);

	void ssyev_sppart_lapack( const char* jobz, const char* uplo, const INT* n,
			 float* A, const INT* lda, float* W,
			 float* work, INT* lwork, INT* info);

	void dsyev_sppart_lapack( const char* jobz, const char* uplo, const INT* n,
			 double* A, const INT* lda, double* W,
			 double* work, INT* lwork, INT* info);

	void cheev_sppart_lapack( const char* jobz, const char* uplo, const INT* n,
			 std::complex<float>* A, const INT* lda, float* W,
			 std::complex<float>* work, INT* lwork, float* rwork, INT* info);

	void zheev_sppart_lapack( const char* jobz, const char* uplo, const INT* n,
			 std::complex<double>* A, const INT* lda, double* W,
			 std::complex<double>* work, INT* lwork, double* rwork, INT* info);

	void ssygv_sppart_lapack( const INT* itype, const char* jobz, const char* uplo, const INT* n,
			 float* A, const INT* lda, float* B, const INT* ldb, float* W,
			 float* work, INT* lwork, INT* info);

	void dsygv_sppart_lapack( const INT* itype, const char* jobz, const char* uplo, const INT* n,
			 double* A, const INT* lda, double* B, const INT* ldb, double* W,
			 double* work, INT* lwork, INT* info);

	void chegv_sppart_lapack( const INT* itype, const char* jobz, const char* uplo, const INT* n,
			 std::complex<float>* A, const INT* lda, std::complex<float>* B, const INT* ldb, float* W,
			 std::complex<float>* work, INT* lwork, float* rwork, INT* info);

	void zhegv_sppart_lapack( const INT* itype, const char* jobz, const char* uplo, const INT* n,
			 std::complex<double>* A, const INT* lda, std::complex<double>* B, const INT* ldb, double* W,
			 std::complex<double>* work, INT* lwork, double* rwork, INT* info);

	void sgeqrf_sppart_lapack( const INT* m, const INT* n,
			 float* A, const INT* lda, float* tau, float* work, INT* lwork, INT* info);


	void dgeqrf_sppart_lapack( const INT* m, const INT* n,
			 double* A, const INT* lda, double* tau, double* work, INT* lwork, INT* info);


	void cgeqrf_sppart_lapack( const INT* m, const INT* n,
			 std::complex<float>* A, const INT* lda, std::complex<float>* tau, std::complex<float>* work, INT* lwork, INT* info);


	void zgeqrf_sppart_lapack( const INT* m, const INT* n,
			 std::complex<double>* A, const INT* lda, std::complex<double>* tau, std::complex<double>* work, INT* lwork, INT* info);

	void sorgqr_sppart_lapack( const INT* m, const INT* n, const INT* k,
			 float* A, const INT* lda, float* tau, float* work, INT* lwork, INT* info);


	void dorgqr_sppart_lapack( const INT* m, const INT* n, const INT* k,
			 double* A, const INT* lda, double* tau, double* work, INT* lwork, INT* info);


	void cungqr_sppart_lapack( const INT* m, const INT* n, const INT* k,
			 std::complex<float>* A, const INT* lda, std::complex<float>* tau, std::complex<float>* work, INT* lwork, INT* info);


	void zungqr_sppart_lapack( const INT* m, const INT* n, const INT* k,
			 std::complex<double>* A, const INT* lda, std::complex<double>* tau, std::complex<double>* work, INT* lwork, INT* info);
}

namespace Sppart{
	namespace blas {
        using Int = INT;
		template<typename ST>
		void gemm(const char *transa, const char *transb, const Int *m, const Int *n, const Int *k,
				  const ST *alpha, const ST *A, const Int *lda, const ST *B, const Int *ldb,
				  const ST *beta, ST *C, const Int *ldc);
		template<>
		void gemm(const char *transa, const char *transb, const Int *m, const Int *n, const Int *k,
				  const float *alpha, const float *A, const Int *lda, const float *B, const Int *ldb,
				  const float *beta, float *C, const Int *ldc) {
			sgemm_sppart_blas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		}
		template<>
		void gemm(const char *transa, const char *transb, const Int *m, const Int *n, const Int *k,
				  const double *alpha, const double *A, const Int *lda, const double *B, const Int *ldb,
				  const double *beta, double *C, const Int *ldc) {
			dgemm_sppart_blas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		}
		template<>
		void gemm(const char *transa, const char *transb, const Int *m, const Int *n, const Int *k,
				  const std::complex<float> *alpha, const std::complex<float> *A, const Int *lda, const std::complex<float> *B, const Int *ldb,
				  const std::complex<float> *beta, std::complex<float> *C, const Int *ldc) {
			cgemm_sppart_blas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		}
		template<>
		void gemm(const char *transa, const char *transb, const Int *m, const Int *n, const Int *k,
				  const std::complex<double> *alpha, const std::complex<double> *A, const Int *lda, const std::complex<double> *B, const Int *ldb,
				  const std::complex<double> *beta, std::complex<double> *C, const Int *ldc) {
			zgemm_sppart_blas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		}

        template<typename ST>
        void hemm(const char *side, const char *uplo, const Int *m, const Int *n,
                  const ST *alpha, const ST *A, const Int *lda, const ST *B, const Int *ldb,
                  const ST *beta, ST *C, const Int *ldc);
        template<>
        void hemm(const char *side, const char *uplo, const Int *m, const Int *n,
                  const float *alpha, const float *A, const Int *lda, const float *B, const Int *ldb,
                  const float *beta, float *C, const Int *ldc) {
            ssymm_sppart_blas(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        template<>
        void hemm(const char *side, const char *uplo, const Int *m, const Int *n,
                  const double *alpha, const double *A, const Int *lda, const double *B, const Int *ldb,
                  const double *beta, double *C, const Int *ldc) {
            dsymm_sppart_blas(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        template<>
        void hemm(const char *side, const char *uplo, const Int *m, const Int *n,
                  const std::complex<float> *alpha, const std::complex<float> *A, const Int *lda, const std::complex<float> *B, const Int *ldb,
                  const std::complex<float> *beta, std::complex<float> *C, const Int *ldc) {
            chemm_sppart_blas(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        template<>
        void hemm(const char *side, const char *uplo, const Int *m, const Int *n,
                  const std::complex<double> *alpha, const std::complex<double> *A, const Int *lda, const std::complex<double> *B, const Int *ldb,
                  const std::complex<double> *beta, std::complex<double> *C, const Int *ldc) {
            zhemm_sppart_blas(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        }

    }

    namespace lapack {
        using Int = INT;

        template<typename ST>
        void getrf(const Int *M, const Int *N,
                   ST *A, const Int *lda, Int *ipiv,
                   Int *info);

        template<>
        void getrf(const Int *M, const Int *N,
                   float *A, const Int *lda, Int *ipiv,
                   Int *info) {
            sgetrf_sppart_lapack(M, N, A, lda, ipiv, info);
        }

        template<>
        void getrf(const Int *M, const Int *N,
                   double *A, const Int *lda, Int *ipiv,
                   Int *info) {
            dgetrf_sppart_lapack(M, N, A, lda, ipiv, info);
        }

        template<>
        void getrf(const Int *M, const Int *N,
                   std::complex<float> *A, const Int *lda, Int *ipiv,
                   Int *info) {
            cgetrf_sppart_lapack(M, N, A, lda, ipiv, info);
        }

        template<>
        void getrf(const Int *M, const Int *N,
                   std::complex<double> *A, const Int *lda, Int *ipiv,
                   Int *info) {
            zgetrf_sppart_lapack(M, N, A, lda, ipiv, info);
        }

        template<typename ST>
        void getrs(const char *trans, const Int *N, const Int *nrhs,
                   ST *A, const Int *lda, Int *ipiv,
                   ST *B, const Int *ldb, Int *info);

        template<>
        void getrs(const char *trans, const Int *N, const Int *nrhs,
                   float *A, const Int *lda, Int *ipiv,
                   float *B, const Int *ldb, Int *info) {
            sgetrs_sppart_lapack(trans, N, nrhs, A, lda, ipiv, B, ldb, info);
        }

        template<>
        void getrs(const char *trans, const Int *N, const Int *nrhs,
                   double *A, const Int *lda, Int *ipiv,
                   double *B, const Int *ldb, Int *info) {
            dgetrs_sppart_lapack(trans, N, nrhs, A, lda, ipiv, B, ldb, info);
        }

        template<>
        void getrs(const char *trans, const Int *N, const Int *nrhs,
                   std::complex<float> *A, const Int *lda, Int *ipiv,
                   std::complex<float> *B, const Int *ldb, Int *info) {
            cgetrs_sppart_lapack(trans, N, nrhs, A, lda, ipiv, B, ldb, info);
        }

        template<>
        void getrs(const char *trans, const Int *N, const Int *nrhs,
                   std::complex<double> *A, const Int *lda, Int *ipiv,
                   std::complex<double> *B, const Int *ldb, Int *info) {
            zgetrs_sppart_lapack(trans, N, nrhs, A, lda, ipiv, B, ldb, info);
        }


        template<typename ST>
        void
        sytrf(const char *uplo, const Int *n, ST *A, const Int *lda, Int *ipiv, ST *work, const Int *lwork, Int *info);

        template<>
        void sytrf(const char *uplo, const Int *n, std::complex<float> *A, const Int *lda, Int *ipiv,
                   std::complex<float> *work, const Int *lwork, Int *info) {
            csytrf_sppart_lapack(uplo, n, A, lda, ipiv, work, lwork, info);
        }

        template<>
        void sytrf(const char *uplo, const Int *n, std::complex<double> *A, const Int *lda, Int *ipiv,
                   std::complex<double> *work, const Int *lwork, Int *info) {
            zsytrf_sppart_lapack(uplo, n, A, lda, ipiv, work, lwork, info);
        }

        template<typename ST>
        void
        sytrs(const char *uplo, const Int *n, const Int *nrhs, ST *A, const Int *lda, Int *ipiv, ST *B, const Int *ldb,
              Int *info);

        template<>
        void sytrs(const char *uplo, const Int *n, const Int *nrhs, std::complex<float> *A, const Int *lda, Int *ipiv,
                   std::complex<float> *B, const Int *ldb, Int *info) {
            csytrs_sppart_lapack(uplo, n, nrhs, A, lda, ipiv, B, ldb, info);
        }

        void sytrs(const char *uplo, const Int *n, const Int *nrhs, std::complex<double> *A, const Int *lda, Int *ipiv,
                    std::complex<double> *B, const Int *ldb, Int *info) {
            zsytrs_sppart_lapack(uplo, n, nrhs, A, lda, ipiv, B, ldb, info);
        }

        template<typename ST>
        void gesvd(const char *job_u, const char *job_vt, const Int *m, const Int *n,
                   ST *A, const Int *lda, typename ScalarTraits<ST>::abs_type *S,
                   ST *U, const Int *ldu, ST *V, const Int *ldv,
                   ST *work, Int *lwork, typename ScalarTraits<ST>::abs_type *rwork, Int *info);
        template<>
        void gesvd(const char *job_u, const char *job_vt, const Int *m, const Int *n,
                   float *A, const Int *lda, float *S,
                   float *U, const Int *ldu, float *V, const Int *ldv,
                   float *work, Int *lwork, float *dummy, Int *info) {
            sgesvd_sppart_lapack(job_u, job_vt, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        }
        template<>
        void gesvd(const char *job_u, const char *job_vt, const Int *m, const Int *n,
                   double *A, const Int *lda, double *S,
                   double *U, const Int *ldu, double *V, const Int *ldv,
                   double *work, Int *lwork, double *dummy, Int *info) {
            dgesvd_sppart_lapack(job_u, job_vt, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        }
        template<>
        void gesvd(const char *job_u, const char *job_vt, const Int *m, const Int *n,
                   std::complex<float> *A, const Int *lda, float *S,
                   std::complex<float> *U, const Int *ldu, std::complex<float> *V, const Int *ldv,
                   std::complex<float> *work, Int *lwork, float *rwork, Int *info) {
            cgesvd_sppart_lapack(job_u, job_vt, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
        }
        template<>
        void gesvd(const char *job_u, const char *job_vt, const Int *m, const Int *n,
                     std::complex<double> *A, const Int *lda, double *S,
                     std::complex<double> *U, const Int *ldu, std::complex<double> *V, const Int *ldv,
                     std::complex<double> *work, Int *lwork, double *rwork, Int *info) {
            zgesvd_sppart_lapack(job_u, job_vt, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
        }

        template<typename ST>
        void geev(const char *jobvl, const char *jobvr, const Int *n,
                  ST *A, const Int *lda, ST *wr, ST *wi,
                  ST *vl, const Int *ldvl, ST *vr, const Int *ldvr,
                  ST *work, Int *lwork, typename ScalarTraits<ST>::abs_type *rwork, Int *info);
        template<>
        void geev(const char *jobvl, const char *jobvr, const Int *n,
                  float *A, const Int *lda, float *wr, float *wi,
                  float *vl, const Int *ldvl, float *vr, const Int *ldvr,
                  float *work, Int *lwork, float *dummy, Int *info) {
            sgeev_sppart_lapack(jobvl, jobvr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
        }
        template<>
        void geev(const char *jobvl, const char *jobvr, const Int *n,
                  double *A, const Int *lda, double *wr, double *wi,
                  double *vl, const Int *ldvl, double *vr, const Int *ldvr,
                  double *work, Int *lwork, double *dummy, Int *info) {
            dgeev_sppart_lapack(jobvl, jobvr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
        }
        template<>
        void geev(const char *jobvl, const char *jobvr, const Int *n,
                  std::complex<float> *A, const Int *lda, std::complex<float> *w, std::complex<float> *dummy,
                  std::complex<float> *vl, const Int *ldvl, std::complex<float> *vr, const Int *ldvr,
                  std::complex<float> *work, Int *lwork, float *rwork, Int *info) {
            cgeev_sppart_lapack(jobvl, jobvr, n, A, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
        }
        template<>
        void geev(const char *jobvl, const char *jobvr, const Int *n,
                    std::complex<double> *A, const Int *lda, std::complex<double> *w, std::complex<double> *dummy,
                    std::complex<double> *vl, const Int *ldvl, std::complex<double> *vr, const Int *ldvr,
                    std::complex<double> *work, Int *lwork, double *rwork, Int *info){
            zgeev_sppart_lapack(jobvl, jobvr, n, A, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
        }


        template<typename ST>
        void ggev(const char* jobvl, const char* jobvr, const Int* n,
                    ST* A, const Int* lda, ST* B, const Int* ldb,
                    ST* alpha_r, ST* alpha_i, ST* beta,
                    ST* vl, const Int* ldvl, ST* vr, const Int* ldvr,
                    ST* work, Int* lwork, typename ScalarTraits<ST>::abs_type *rwork, Int* info);
        template<>
        void ggev(const char* jobvl, const char* jobvr, const Int* n,
                    float* A, const Int* lda, float* B, const Int* ldb,
                    float* alpha_r, float* alpha_i, float* beta,
                    float* vl, const Int* ldvl, float* vr, const Int* ldvr,
                    float* work, Int* lwork, float* dummy, Int* info){
            sggev_sppart_lapack(jobvl, jobvr, n, A, lda, B, ldb, alpha_r, alpha_i, beta, vl, ldvl, vr, ldvr, work, lwork, info);
        }
        template<>
        void ggev(const char* jobvl, const char* jobvr, const Int* n,
                    double* A, const Int* lda, double* B, const Int* ldb,
                    double* alpha_r, double* alpha_i, double* beta,
                    double* vl, const Int* ldvl, double* vr, const Int* ldvr,
                    double* work, Int* lwork, double* dummy, Int* info){
            dggev_sppart_lapack(jobvl, jobvr, n, A, lda, B, ldb, alpha_r, alpha_i, beta, vl, ldvl, vr, ldvr, work, lwork, info);
        }
        template<>
        void ggev(const char* jobvl, const char* jobvr, const Int* n,
                    std::complex<float>* A, const Int* lda, std::complex<float>* B, const Int* ldb,
                    std::complex<float>* alpha, std::complex<float>* dummy, std::complex<float>* beta,
                    std::complex<float>* vl, const Int* ldvl, std::complex<float>* vr, const Int* ldvr,
                    std::complex<float>* work, Int* lwork, float* rwork, Int* info){
            cggev_sppart_lapack(jobvl, jobvr, n, A, lda, B, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
        }
        template<>
        void ggev(const char* jobvl, const char* jobvr, const Int* n,
                    std::complex<double>* A, const Int* lda, std::complex<double>* B, const Int* ldb,
                    std::complex<double>* alpha, std::complex<double>* dummy, std::complex<double>* beta,
                    std::complex<double>* vl, const Int* ldvl, std::complex<double>* vr, const Int* ldvr,
                    std::complex<double>* work, Int* lwork, double* rwork, Int* info){
            zggev_sppart_lapack(jobvl, jobvr, n, A, lda, B, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
        }

        template<typename ST>
        void heev( const char* jobz, const char* uplo, const Int* n,
                   ST* A, const Int* lda, typename ScalarTraits<ST>::abs_type* W,
                   ST* work, Int* lwork, typename ScalarTraits<ST>::abs_type* dummy, Int* info);
        template<>
        void heev( const char* jobz, const char* uplo, const Int* n,
                     float* A, const Int* lda, float* W,
                     float* work, Int* lwork, float* dummy, Int* info){
            ssyev_sppart_lapack(jobz, uplo, n, A, lda, W, work, lwork, info);
        }
        template<>
        void heev( const char* jobz, const char* uplo, const Int* n,
                     double* A, const Int* lda, double* W,
                     double* work, Int* lwork, double* rwork, Int* info){
            dsyev_sppart_lapack(jobz, uplo, n, A, lda, W, work, lwork, info);
        }
        template<>
        void heev( const char* jobz, const char* uplo, const Int* n,
                     std::complex<float>* A, const Int* lda, float* W,
                     std::complex<float>* work, Int* lwork, float* rwork, Int* info){
            cheev_sppart_lapack(jobz, uplo, n, A, lda, W, work, lwork, rwork, info);
        }
        template<>
        void heev( const char* jobz, const char* uplo, const Int* n,
                     std::complex<double>* A, const Int* lda, double* W,
                     std::complex<double>* work, Int* lwork, double* rwork, Int* info){
            zheev_sppart_lapack(jobz, uplo, n, A, lda, W, work, lwork, rwork, info);
        }

        template<typename ST>
        void hegv( const Int* itype, const char* jobz, const char* uplo, const Int* n,
                   ST* A, const Int* lda, ST* B, const Int* ldb, typename ScalarTraits<ST>::abs_type* W,
                   ST* work, Int* lwork, typename ScalarTraits<ST>::abs_type* rwork, Int* info);
        template<>
        void hegv( const Int* itype, const char* jobz, const char* uplo, const Int* n,
                   float* A, const Int* lda, float* B, const Int* ldb, float* W,
                   float* work, Int* lwork, float* dummy, Int* info){
            ssygv_sppart_lapack(itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
        }
        template<>
        void hegv( const Int* itype, const char* jobz, const char* uplo, const Int* n,
                     double* A, const Int* lda, double* B, const Int* ldb, double* W,
                     double* work, Int* lwork, double* rwork, Int* info){
            dsygv_sppart_lapack(itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
        }
        template<>
        void hegv( const Int* itype, const char* jobz, const char* uplo, const Int* n,
                     std::complex<float>* A, const Int* lda, std::complex<float>* B, const Int* ldb, float* W,
                     std::complex<float>* work, Int* lwork, float* rwork, Int* info){
            chegv_sppart_lapack(itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, rwork, info);
        }
        template<>
        void hegv( const Int* itype, const char* jobz, const char* uplo, const Int* n,
                     std::complex<double>* A, const Int* lda, std::complex<double>* B, const Int* ldb, double* W,
                     std::complex<double>* work, Int* lwork, double* rwork, Int* info){
            zhegv_sppart_lapack(itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, rwork, info);
        }

        template<typename ST>
        void geqrf( const INT* m, const INT* n,
			 ST* A, const INT* lda, ST* tau, ST* work, INT* lwork, INT* info);
        template<>
        void geqrf( const INT* m, const INT* n,
			 float* A, const INT* lda, float* tau, float* work, INT* lwork, INT* info){
            sgeqrf_sppart_lapack(m, n, A, lda, tau, work, lwork, info);
        }
        void geqrf( const INT* m, const INT* n,
			 double* A, const INT* lda, double* tau, double* work, INT* lwork, INT* info){
            dgeqrf_sppart_lapack(m, n, A, lda, tau, work, lwork, info);
        }
        void geqrf( const INT* m, const INT* n,
			 std::complex<float>* A, const INT* lda, std::complex<float>* tau, std::complex<float>* work, INT* lwork, INT* info){
            cgeqrf_sppart_lapack(m, n, A, lda, tau, work, lwork, info);
        }
        void geqrf( const INT* m, const INT* n,
			 std::complex<double>* A, const INT* lda, std::complex<double>* tau, std::complex<double>* work, INT* lwork, INT* info){
            zgeqrf_sppart_lapack(m, n, A, lda, tau, work, lwork, info);
        }

        template<typename ST>
        void ungqr( const INT* m, const INT* n, const INT* k,
			 ST* A, const INT* lda, ST* tau, ST* work, INT* lwork, INT* info);
        template<>
        void ungqr( const INT* m, const INT* n, const INT* k,
			 float* A, const INT* lda, float* tau, float* work, INT* lwork, INT* info){
            sorgqr_sppart_lapack(m, n, k, A, lda, tau, work, lwork, info);
        }
        void ungqr( const INT* m, const INT* n, const INT* k,
			 double* A, const INT* lda, double* tau, double* work, INT* lwork, INT* info){
            dorgqr_sppart_lapack(m, n, k, A, lda, tau, work, lwork, info);
        }
        void ungqr( const INT* m, const INT* n, const INT* k,
			 std::complex<float>* A, const INT* lda, std::complex<float>* tau, std::complex<float>* work, INT* lwork, INT* info){
            cungqr_sppart_lapack(m, n, k, A, lda, tau, work, lwork, info);
        }
        void ungqr( const INT* m, const INT* n, const INT* k,
			 std::complex<double>* A, const INT* lda, std::complex<double>* tau, std::complex<double>* work, INT* lwork, INT* info){
            zungqr_sppart_lapack(m, n, k, A, lda, tau, work, lwork, info);
        }

    }
}

#ifdef SPPART_INT_BACKUP
#define INT SPPART_INT_BACKUP
#else
#undef INT
#endif

#endif //SPPART_BLAS_LAPACK_HPP