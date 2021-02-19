// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <vector>
#include <julia.h>
#include <CLI11/CLI11.hpp>
#include <metis.h>

#include <matrix_read.hpp>
#include <connected_component.hpp>
#include <timer.hpp>

JULIA_DEFINE_FAST_TLS() // only define this once, in an executable (not in a shared library) if you want fast code.

int main(int argc, char* argv[]){
    CLI::App app("test");
    
    std::string matrix_file_path = "";
    int nparts = 2;
    real_t ubfactor = 1.001;
    int rand_seed = 0;
    app.add_option("--mat", matrix_file_path, "File path of MATLAB mat file for input graph (matrix) from SuiteSparse Matrix Collection")
        ->required(true)
        ->check(CLI::ExistingFile);
    app.add_option("--npart", nparts, "Number of part for partitioning")
        ->default_val(2);
    app.add_option("--ub", ubfactor, "Unbalance tolerance")
        ->default_val(1.001);
    app.add_option("--seed", rand_seed, "Seed for random number generator")
        ->default_val(0);

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

    std::vector<int> xadj_metis(nv+1);
    for (int i = 0; i < nv+1; ++i){
        xadj_metis[i] = xadj[i];
    }

    int nvtxs = nv;
    int ncon = 1;
    int objval = -1;
    int ret;
    std::vector<int> part(nv);

    int metis_options[METIS_NOPTIONS];
    ret = METIS_SetDefaultOptions(metis_options);
    // metis_options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO;
    metis_options[METIS_OPTION_SEED] = rand_seed;
    double metis_time = Sppart::timeit([&]{
        ret = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj_metis.data(), adjncy.data(), NULL, NULL, NULL, &nparts, NULL, &ubfactor, metis_options, &objval, part.data());
    });

    int cut2 = Sppart::compute_cut(nv, xadj.data(), adjncy.data(), part.data());
    double maxbal = Sppart::compute_maxbal(nparts, nv, xadj.data(), adjncy.data(), part.data());


    printf("metis cut %d %d\n", ret, objval);
    printf("cut2 %d %d\n", cut2);
    printf("metis maxbal %lf\n", maxbal);
    printf("metis time %lf\n", metis_time);

    return 0;
}

