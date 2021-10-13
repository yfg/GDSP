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
#include <util.hpp>
#include <json.hpp>
#include <fstream>
#include <version.hpp>

#include <mkl.h>

JULIA_DEFINE_FAST_TLS() // only define this once, in an executable (not in a shared library) if you want fast code.

int main(int argc, char* argv[]){
    CLI::App app("metis_test");

    std::string matrix_file_path = "";
    int rand_seed = 0;
    app.add_option("--mat", matrix_file_path, "File path of MATLAB mat file for input graph (matrix) from SuiteSparse Matrix Collection")
        ->required(true)
        ->check(CLI::ExistingFile);
    app.add_option("--seed", rand_seed, "Seed for random number generator")
        ->default_val(0);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);;
    }

    using XADJ_INT = int;

    int nv0;
    std::vector<XADJ_INT> xadj0;
    std::vector<int> adjncy0;
    Sppart::matrix_read(matrix_file_path, nv0, xadj0, adjncy0);

    int nv;
    std::vector<XADJ_INT> xadj;
    std::vector<int> adjncy;
    Sppart::preprocess_graph(nv0, xadj0, adjncy0, nv, xadj, adjncy);

    int metis_options[METIS_NOPTIONS];
    int ret = METIS_SetDefaultOptions(metis_options);
    // metis_options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO;
    metis_options[METIS_OPTION_SEED] = rand_seed;
    auto perm = Sppart::create_up_array<MKL_INT>(nv);
    auto inv_perm = Sppart::create_up_array<int>(nv);

    ret = METIS_NodeND(&nv, xadj.data(), adjncy.data(), NULL, metis_options, perm.get(), inv_perm.get());
    if ( ret != 1 ) {
        printf("METIS error !\n");
        std::terminate();
    }

    std::vector<XADJ_INT> xadj_upt;
    std::vector<int> adjncy_upt;
    Sppart::create_upper_triangular(nv, xadj, adjncy, xadj_upt, adjncy_upt);

    _MKL_DSS_HANDLE_t pt[64];
    const MKL_INT mtype = -2; // real and symmetric indefinite
    MKL_INT iparm[64];

    pardisoinit(pt, &mtype, iparm);

    const MKL_INT maxfct = 1, mnum = 1, nrhs=1;
    const MKL_INT msglvl = 0; // 0: Do not print information, 1: print information
    const MKL_INT n = nv;
    const MKL_INT phase = 11; // Only analysis phase
    MKL_INT error;
    auto val = Sppart::create_up_array<double>(adjncy_upt.size());
    for (size_t i = 0; i < adjncy_upt.size(); ++i){
        val[i] = 1.0;
    }
    iparm[0] = 1; // use non default values
    iparm[4] = 1; // use the user supplied fill-in reducing permutation from the perm array.
    iparm[17] = -1; // Enable reporting the number of non-zero elements in the factors
    iparm[34] = 1; // Zero-based indexing
    
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, val.get(), xadj_upt.data(), adjncy_upt.data(), perm.get(), &nrhs, iparm, &msglvl, NULL, NULL, &error);
    if ( error != 0 ){
        printf("Pardiso error! error code = %d\n", error);
        std::terminate();
    }
    printf("Number of non-zero elements in the factors: %d\n", iparm[17]);

    return 0;
}
