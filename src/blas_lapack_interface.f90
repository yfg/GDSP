! Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

subroutine sgemm_sppart_blas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: transa, transb
    integer(c_int) :: m, n, k, lda, ldb, ldc
    real(c_float) :: alpha, A(*), B(*), beta, C(*)
    call sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
end subroutine
subroutine dgemm_sppart_blas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: transa, transb
    integer(c_int) :: m, n, k, lda, ldb, ldc
    real(c_double) :: alpha, A(*), B(*), beta, C(*)
    call dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
end subroutine
subroutine cgemm_sppart_blas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: transa, transb
    integer(c_int) :: m, n, k, lda, ldb, ldc
    complex(c_float_complex) :: alpha, A(*), B(*), beta, C(*)
    call cgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
end subroutine
subroutine zgemm_sppart_blas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: transa, transb
    integer(c_int) :: m, n, k, lda, ldb, ldc
    complex(c_double_complex) :: alpha, A(*), B(*), beta, C(*)
    call zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
end subroutine


subroutine ssymm_sppart_blas(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: side, uplo
    integer(c_int) :: m, n, lda, ldb, ldc
    real(c_float) :: alpha, A(*), B(*), beta, C(*)
    call ssymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
end subroutine
subroutine dsymm_sppart_blas(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: side, uplo
    integer(c_int) :: m, n, lda, ldb, ldc
    real(c_double) :: alpha, A(*), B(*), beta, C(*)
    call dsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
end subroutine
subroutine chemm_sppart_blas(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: side, uplo
    integer(c_int) :: m, n, lda, ldb, ldc
    complex(c_float_complex) :: alpha, A(*), B(*), beta, C(*)
    call chemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
end subroutine
subroutine zhemm_sppart_blas(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: side, uplo
    integer(c_int) :: m, n, lda, ldb, ldc
    complex(c_double_complex) :: alpha, A(*), B(*), beta, C(*)
    call zhemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
end subroutine


subroutine sgetrf_sppart_lapack(m, n, A, lda, ipiv, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, lda, ipiv(*), info
    real(c_float) :: A(*)
    call sgetrf(m, n, A, lda, ipiv, info)
end subroutine
subroutine dgetrf_sppart_lapack(m, n, A, lda, ipiv, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, lda, ipiv(*), info
    real(c_double) :: A(*)
    call dgetrf(m, n, A, lda, ipiv, info)
end subroutine
subroutine cgetrf_sppart_lapack(m, n, A, lda, ipiv, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, lda, ipiv(*), info
    complex(c_float_complex) :: A(*)
    call cgetrf(m, n, A, lda, ipiv, info)
end subroutine
subroutine zgetrf_sppart_lapack(m, n, A, lda, ipiv, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, lda, ipiv(*), info
    complex(c_double_complex) :: A(*)
    call zgetrf(m, n, A, lda, ipiv, info)
end subroutine


subroutine sgetrs_sppart_lapack(trans, n, nrhs, A, lda, ipiv, B, ldb, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: trans
    integer(c_int) :: n, nrhs, lda, ipiv(*), ldb, info
    real(c_float) :: A(*), B(*)
    call sgetrs(trans, n, nrhs, A, lda, ipiv, B, ldb, info)
end subroutine
subroutine dgetrs_sppart_lapack(trans, n, nrhs, A, lda, ipiv, B, ldb, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: trans
    integer(c_int) :: n, nrhs, lda, ipiv(*), ldb, info
    real(c_double) :: A(*), B(*)
    call dgetrs(trans, n, nrhs, A, lda, ipiv, B, ldb, info)
end subroutine
subroutine cgetrs_sppart_lapack(trans, n, nrhs, A, lda, ipiv, B, ldb, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: trans
    integer(c_int) :: n, nrhs, lda, ipiv(*), ldb, info
    complex(c_float_complex) :: A(*), B(*)
    call cgetrs(trans, n, nrhs, A, lda, ipiv, B, ldb, info)
end subroutine
subroutine zgetrs_sppart_lapack(trans, n, nrhs, A, lda, ipiv, B, ldb, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: trans
    integer(c_int) :: n, nrhs, lda, ipiv(*), ldb, info
    complex(c_double_complex) :: A(*), B(*)
    call zgetrs(trans, n, nrhs, A, lda, ipiv, B, ldb, info)
end subroutine


subroutine csytrf_sppart_lapack(uplo, n, A, lda, ipiv, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: uplo
    integer(c_int) :: n, lda, ipiv(*), lwork, info
    complex(c_float_complex) :: A(*), work(*)
    call csytrf(uplo, n, A, lda, ipiv, work, lwork, info)
end subroutine
subroutine zsytrf_sppart_lapack(uplo, n, A, lda, ipiv, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: uplo
    integer(c_int) :: n, lda, ipiv(*), lwork, info
    complex(c_double_complex) :: A(*), work(*)
    call zsytrf(uplo, n, A, lda, ipiv, work, lwork, info)
end subroutine


subroutine csytrs_sppart_lapack(uplo, n, nrhs, A, lda, ipiv, B, ldb, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: uplo
    integer(c_int) :: n, nrhs, lda, ipiv(*), ldb, info
    complex(c_float_complex) :: A(*), B(*)
    call csytrs(uplo, n, nrhs, A, lda, ipiv, B, ldb, info)
end subroutine
subroutine zsytrs_sppart_lapack(uplo, n, nrhs, A, lda, ipiv, B, ldb, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: uplo
    integer(c_int) :: n, nrhs, lda, ipiv(*), ldb, info
    complex(c_double_complex) :: A(*), B(*)
    call zsytrs(uplo, n, nrhs, A, lda, ipiv, B, ldb, info)
end subroutine


subroutine sgesvd_sppart_lapack(job_u, job_v, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_u, job_v
    integer(c_int) :: m, n, lda, ldb, ldu, ldv, lwork, info
    real(c_float) :: A(*), S(*), U(*), V(*), work(*)
    call sgesvd(job_u, job_v, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info)
end subroutine
subroutine dgesvd_sppart_lapack(job_u, job_v, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_u, job_v
    integer(c_int) :: m, n, lda, ldb, ldu, ldv, lwork, info
    real(c_double) :: A(*), S(*), U(*), V(*), work(*)
    call dgesvd(job_u, job_v, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info)
end subroutine
subroutine cgesvd_sppart_lapack(job_u, job_v, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_u, job_v
    integer(c_int) :: m, n, lda, ldb, ldu, ldv, lwork, info
    complex(c_float_complex) :: A(*), S(*), U(*), V(*), work(*)
    real(c_float) :: rwork(*)
    call cgesvd(job_u, job_v, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info)
end subroutine
subroutine zgesvd_sppart_lapack(job_u, job_v, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_u, job_v
    integer(c_int) :: m, n, lda, ldb, ldu, ldv, lwork, info
    complex(c_double_complex) :: A(*), S(*), U(*), V(*), work(*)
    real(c_double) :: rwork(*)
    call zgesvd(job_u, job_v, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info)
end subroutine


subroutine sgeev_sppart_lapack(job_vl, job_vr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_vl, job_vr
    integer(c_int) :: n, lda, ldvl, ldvr, lwork, info
    real(c_float) :: A(*), wr(*), wi(*), vl(*), vr(*), work(*)
    call sgeev(job_vl, job_vr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info)
end subroutine
subroutine dgeev_sppart_lapack(job_vl, job_vr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_vl, job_vr
    integer(c_int) :: n, lda, ldvl, ldvr, lwork, info
    real(c_double) :: A(*), wr(*), wi(*), vl(*), vr(*), work(*)
    call dgeev(job_vl, job_vr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info)
end subroutine
subroutine cgeev_sppart_lapack(job_vl, job_vr, n, A, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_vl, job_vr
    integer(c_int) :: n, lda, ldvl, ldvr, lwork, info
    complex(c_float_complex) :: A(*), w(*), vl(*), vr(*), work(*)
    real(c_float) :: rwork(*)
    call cgeev(job_vl, job_vr, n, A, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info)
end subroutine
subroutine zgeev_sppart_lapack(job_vl, job_vr, n, A, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_vl, job_vr
    integer(c_int) :: n, lda, ldvl, ldvr, lwork, info
    complex(c_double_complex) :: A(*), w(*), vl(*), vr(*), work(*)
    real(c_double) :: rwork(*)
    call zgeev(job_vl, job_vr, n, A, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info)
end subroutine


subroutine sggev_sppart_lapack(job_vl, job_vr, n, A, lda, B, ldb, &
        alpha_r, alpha_i, beta, vl, ldvl, vr, ldvr, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_vl, job_vr
    integer(c_int) :: n, lda, ldb, ldvl, ldvr, lwork, info
    real(c_float) :: A(*), B(*), alpha_r(*), alpha_i(*), beta(*), vl(*), vr(*), work(*)
    call sggev(job_vl, job_vr, n, A, lda, B, ldb, alpha_r, alpha_i, beta, vl, ldvl, vr, ldvr, work, lwork, info)
end subroutine
subroutine dggev_sppart_lapack(job_vl, job_vr, n, A, lda, B, ldb, &
        alpha_r, alpha_i, beta, vl, ldvl, vr, ldvr, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_vl, job_vr
    integer(c_int) :: n, lda, ldb, ldvl, ldvr, lwork, info
    real(c_double) :: A(*), B(*), alpha_r(*), alpha_i(*), beta(*), vl(*), vr(*), work(*)
    call dggev(job_vl, job_vr, n, A, lda, B, ldb, alpha_r, alpha_i, beta, vl, ldvl, vr, ldvr, work, lwork, info)
end subroutine
subroutine cggev_sppart_lapack(job_vl, job_vr, n, A, lda, B, ldb, &
        alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_vl, job_vr
    integer(c_int) :: n, lda, ldb, ldvl, ldvr, lwork, info
    complex(c_float_complex) :: A(*), B(*), alpha(*), beta(*), vl(*), vr(*), work(*)
    real(c_float) :: rwork(*)
    call cggev(job_vl, job_vr, n, A, lda, B, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info)
end subroutine
subroutine zggev_sppart_lapack(job_vl, job_vr, n, A, lda, B, ldb, &
        alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: job_vl, job_vr
    integer(c_int) :: n, lda, ldb, ldvl, ldvr, lwork, info
    complex(c_double_complex) :: A(*), B(*), alpha(*), beta(*), vl(*), vr(*), work(*)
    real(c_double) :: rwork(*)
    call zggev(job_vl, job_vr, n, A, lda, B, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info)
end subroutine


subroutine ssyev_sppart_lapack(jobz, uplo, n, A, lda, w, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: jobz, uplo
    integer(c_int) :: n, lda, lwork, info
    real(c_float) :: A(*), w(*), work(*)
    call ssyev(jobz, uplo, n, A, lda, w, work, lwork, info)
end subroutine
subroutine dsyev_sppart_lapack(jobz, uplo, n, A, lda, w, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: jobz, uplo
    integer(c_int) :: n, lda, lwork, info
    real(c_double) :: A(*), w(*), work(*)
    call dsyev(jobz, uplo, n, A, lda, w, work, lwork, info)
end subroutine
subroutine cheev_sppart_lapack(jobz, uplo, n, A, lda, w, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: jobz, uplo
    integer(c_int) :: n, lda, lwork, info
    complex(c_float_complex) :: A(*), w(*), work(*)
    real(c_float) :: rwork(*)
    call cheev(jobz, uplo, n, A, lda, w, work, lwork, rwork, info)
end subroutine
subroutine zheev_sppart_lapack(jobz, uplo, n, A, lda, w, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: jobz, uplo
    integer(c_int) :: n, lda, lwork, info
    complex(c_double_complex) :: A(*), w(*), work(*)
    real(c_double) :: rwork(*)
    call zheev(jobz, uplo, n, A, lda, w, work, lwork, rwork, info)
end subroutine


subroutine ssygv_sppart_lapack(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: jobz, uplo
    integer(c_int) :: itype, n, lda, ldb, lwork, info
    real(c_float) :: A(*), B(*), w(*), work(*)
    call ssygv(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, info)
end subroutine
subroutine dsygv_sppart_lapack(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: jobz, uplo
    integer(c_int) :: itype, n, lda, ldb, lwork, info
    real(c_double) :: A(*), B(*), w(*), work(*)
    call dsygv(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, info)
end subroutine
subroutine chegv_sppart_lapack(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: jobz, uplo
    integer(c_int) :: itype, n, lda, ldb, lwork, info
    complex(c_double_complex) :: A(*), B(*), w(*), work(*)
    real(c_double) :: rwork(*)
    call chegv(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, info)
end subroutine
subroutine zhegv_sppart_lapack(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, info) bind(c)
    use iso_c_binding
    implicit none
    character(c_char) :: jobz, uplo
    integer(c_int) :: itype, n, lda, ldb, lwork, info
    complex(c_double_complex) :: A(*), B(*), w(*), work(*)
    real(c_double) :: rwork(*)
    call zhegv(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, info)
end subroutine

subroutine sgeqrf_sppart_lapack(m, n, A, lda, tau, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, lda, lwork, info
    complex(c_float) :: A(*), tau(*), work(*)
    call sgeqrf(m, n, A, lda, tau, work, lwork, info)
end subroutine
subroutine dgeqrf_sppart_lapack(m, n, A, lda, tau, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, lda, lwork, info
    complex(c_double) :: A(*), tau(*), work(*)
    call dgeqrf(m, n, A, lda, tau, work, lwork, info)
end subroutine
subroutine cgeqrf_sppart_lapack(m, n, A, lda, tau, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, lda, lwork, info
    complex(c_float_complex) :: A(*), tau(*), work(*)
    call cgeqrf(m, n, A, lda, tau, work, lwork, info)
end subroutine
subroutine zgeqrf_sppart_lapack(m, n, A, lda, tau, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, lda, lwork, info
    complex(c_double_complex) :: A(*), tau(*), work(*)
    call zgeqrf(m, n, A, lda, tau, work, lwork, info)
end subroutine

subroutine sorgqr_sppart_lapack(m, n, k, A, lda, tau, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, k, lda, lwork, info
    complex(c_float) :: A(*), tau(*), work(*)
    call sorgqr(m, n, k, A, lda, tau, work, lwork, info)
end subroutine
subroutine dorgqr_sppart_lapack(m, n, k, A, lda, tau, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, k, lda, lwork, info
    complex(c_double) :: A(*), tau(*), work(*)
    call dorgqr(m, n, k, A, lda, tau, work, lwork, info)
end subroutine
subroutine cungqr_sppart_lapack(m, n, k, A, lda, tau, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, k, lda, lwork, info
    complex(c_float_complex) :: A(*), tau(*), work(*)
    call cungqr(m, n, k, A, lda, tau, work, lwork, info)
end subroutine
subroutine zungqr_sppart_lapack(m, n, k, A, lda, tau, work, lwork, info) bind(c)
    use iso_c_binding
    implicit none
    integer(c_int) :: m, n, k, lda, lwork, info
    complex(c_double_complex) :: A(*), tau(*), work(*)
    call zungqr(m, n, k, A, lda, tau, work, lwork, info)
end subroutine

