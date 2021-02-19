// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <vector>
#include <julia.h>
#include <CLI11/CLI11.hpp>
#include <mtmetis.h>

#include <matrix_read.hpp>
#include <connected_component.hpp>
#include <timer.hpp>


JULIA_DEFINE_FAST_TLS() // only define this once, in an executable (not in a shared library) if you want fast code.

int main(int argc, char* argv[]){
    CLI::App app("test");
    
    std::string matrix_file_path = "";
    int nparts = 2;
    app.add_option("--mat", matrix_file_path, "File path of MATLAB mat file for input graph (matrix) from SuiteSparse Matrix Collection")
        ->required(true)
        ->check(CLI::ExistingFile);
    app.add_option("--npart", nparts, "Number of part for partitioning")
        ->default_val(2);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);;
    }

    //using XADJ_INT = int64_t;
    using XADJ_INT = int;

    int nv0;
    std::vector<XADJ_INT> xadj0;
    std::vector<int> adjncy0;
    Sppart::matrix_read(matrix_file_path, nv0, xadj0, adjncy0);

    int nv;
    std::vector<XADJ_INT> xadj;
    std::vector<int> adjncy;
    Sppart::preprocess_graph(nv0, xadj0, adjncy0, nv, xadj, adjncy);

    std::vector<mtmetis_adj_type> xadj_metis(nv+1);
    for (int i = 0; i < nv+1; ++i){
        xadj_metis[i] = xadj[i];
    }

    mtmetis_vtx_type nvtxs = nv;
    mtmetis_vtx_type ncon = 1;
    mtmetis_pid_type nparts_mt = nparts;
    int objval = -1;
    int ret;
    std::vector<mtmetis_pid_type> part(nv);
    mtmetis_vtx_type* adjncy_ptr = reinterpret_cast<mtmetis_vtx_type*>(adjncy.data());

    // int metis_options[METIS_NOPTIONS];
    // double mtmetis_options[MTMETIS_NOPTIONS];
    double *mtmetis_options = mtmetis_init_options();
    if ( mtmetis_options == NULL ) {
        printf("options is null!\n");
        std::terminate();
    }

    mtmetis_real_type ubvec = 1.001; // default is 1.001?

    // ret = METIS_SetDefaultOptions(mtmetis_options);
    // metis_options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO;
    double mtmetis_time = Sppart::timeit([&]{
        ret = MTMETIS_PartGraphRecursive(&nvtxs, &ncon, xadj_metis.data(), adjncy_ptr, NULL, NULL, NULL, &nparts_mt, NULL, &ubvec, mtmetis_options, &objval, part.data());
    });
    double maxbal = Sppart::compute_maxbal(nparts, nv, xadj.data(), adjncy.data(), reinterpret_cast<int*>(part.data()));
    int cut2 = Sppart::compute_cut(nv, xadj.data(), adjncy.data(), reinterpret_cast<int*>(part.data()));

    // if (mtmetis_options) {
    //     dl_free(mtmetis_options);
    // }

    printf("mtmetis cut %d %d\n", ret, objval);
    printf("mtmetis cut2 %d\n", cut2);
    // printf("metis maxbal %lf\n", maxbal);
    printf("mtmetis maxbal %lf\n", maxbal);
    printf("mtmetis time %lf\n", mtmetis_time);

    return 0;
}

