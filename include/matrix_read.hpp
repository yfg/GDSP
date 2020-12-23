// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef INCLUDE_SPPART_MATRIX_READ_HPP
#define INCLUDE_SPPART_MATRIX_READ_HPP

#include<cstdint>
#include<string>
#include<limits>
#include<julia.h>

namespace Sppart{

    template <class XADJ_INT>
    void matrix_read(const std::string &file_path, int &nv_out, std::vector<XADJ_INT> &xadj_out, std::vector<int> &adjncy_out){
        jl_init();

        jl_eval_string("using MAT");
        jl_eval_string("using SparseArrays");
        jl_value_t *mod_MAT = (jl_value_t*)jl_eval_string("MAT");
        jl_value_t *mod_SparseArrays = (jl_value_t*)jl_eval_string("SparseArrays");
        jl_function_t *matread_func = jl_get_function((jl_module_t*)mod_MAT,"matread");
        jl_value_t *dict0 = jl_call1(matread_func, jl_cstr_to_string(file_path.c_str())); // dict0 = matread("")
        jl_function_t *getindex_func = jl_get_function(jl_base_module, "getindex");
        jl_value_t *dict1 = jl_call2(getindex_func, dict0, jl_cstr_to_string("Problem")); // dict1 = dict0["Problem"]
        jl_value_t *matrix = jl_call2(getindex_func, dict1, jl_cstr_to_string("A")); // matrix = dict1["A"]
        jl_value_t *jl_n = (jl_value_t*)jl_get_field(matrix, "n");
        const int64_t n = jl_unbox_int64(jl_n);
        jl_value_t *jl_m = (jl_value_t*)jl_get_field(matrix, "m");
        const int64_t m = jl_unbox_int64(jl_m);
        assert(n == m);
        assert(n <= std::numeric_limits<int>::max());
        nv_out = n;

        jl_array_t *jl_colptr = (jl_array_t*)jl_get_field(matrix, "colptr");
        int64_t *colptr = (int64_t*)jl_array_data(jl_colptr); // colptr = matrix.colptr
        jl_array_t *jl_rowval = (jl_array_t*)jl_get_field(matrix, "rowval");
        int64_t *rowval = (int64_t*)jl_array_data(jl_rowval); // colptr = matrix.rowval

        const int64_t nnz = colptr[nv_out] - 1;
        
        xadj_out.resize(nv_out+1);
        adjncy_out.resize(nnz);

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < nv_out+1; ++i){
                xadj_out[i] = colptr[i] - 1; // convert 1-origin to 0-origin
            }
            #pragma omp for
            for (int64_t i = 0; i < nnz; ++i){
                adjncy_out[i] = rowval[i] - 1; // convert 1-origin to 0-origin
            }
        }

        jl_atexit_hook(0);
        return;
    }
}

#endif
