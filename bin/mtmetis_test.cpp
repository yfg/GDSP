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
#include <util.hpp>
#include <json.hpp>
#include <fstream>
#include <omp.h>
#include <version.hpp>

JULIA_DEFINE_FAST_TLS() // only define this once, in an executable (not in a shared library) if you want fast code.

int main(int argc, char* argv[]){
    CLI::App app("mtmetis_test");
    
    std::string matrix_file_path = "";
    int nparts = 2;
    std::string json_file_path = "";
    int n_trial = 1;
    int rand_seed = 0;
    double ubfactor = 1.001;
    app.add_option("--mat", matrix_file_path, "File path of MATLAB mat file for input graph (matrix) from SuiteSparse Matrix Collection")
        ->required(true)
        ->check(CLI::ExistingFile);
    app.add_option("--npart", nparts, "Number of part for partitioning")
        ->default_val(2);
    app.add_option("--ub", ubfactor, "Unbalance tolerance")
        ->default_val(1.001);
    app.add_option("--seed", rand_seed, "Seed for random number generator")
        ->default_val(0);
    app.add_option("--json", json_file_path, "File path for output JSON file");
    app.add_option("--ntry", n_trial, "Number of trials")
        ->default_val(1);

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

    std::vector<int64_t> cut_vec(n_trial);
    std::vector<double> maxbal_vec(n_trial);
    std::vector<double> mtmetis_time_vec(n_trial);

    printf("Git hash %s\n", GIT_COMMIT_HASH);
    const int nthreads = omp_get_max_threads();
    printf("Num threads = %d\n", nthreads);

    for (int i = 0; i < n_trial; ++i){
        // int metis_options[METIS_NOPTIONS];
        // double mtmetis_options[MTMETIS_NOPTIONS];
        double *mtmetis_options = mtmetis_init_options();
        if ( mtmetis_options == NULL ) {
            printf("options is null!\n");
            std::terminate();
        }
        mtmetis_options[MTMETIS_OPTION_SEED] = rand_seed + i;
        mtmetis_real_type ubvec = 1.001; // dummy, in fact, this is not used in mt-Metis
        mtmetis_options[MTMETIS_OPTION_UBFACTOR] = ubfactor;
        // ret = METIS_SetDefaultOptions(mtmetis_options);
        // metis_options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO;
        mtmetis_time_vec[i] = Sppart::timeit([&]{
            ret = MTMETIS_PartGraphRecursive(&nvtxs, &ncon, xadj_metis.data(), adjncy_ptr, NULL, NULL, NULL, &nparts_mt, NULL, &ubvec, mtmetis_options, &objval, part.data());
        });
        maxbal_vec[i] = Sppart::compute_maxbal(nparts, nv, xadj.data(), adjncy.data(), reinterpret_cast<int*>(part.data()));
        int cut2 = Sppart::compute_cut(nv, xadj.data(), adjncy.data(), reinterpret_cast<int*>(part.data()));
        free(mtmetis_options);
        // if (mtmetis_options) {
        //     dl_free(mtmetis_options);
        // }
        if ( ret != 1 || objval != cut2 ) {
            printf("Something wrong with MTMETIS !!!\n");
            std::terminate();
        }

        cut_vec[i] = objval;        
        printf("mtmetis cut %d %d\n", ret, objval);
        printf("mtmetis cut2 %d\n", cut2);
        printf("mtmetis maxbal %lf\n", maxbal_vec[i]);
        printf("mtmetis time %lf\n", mtmetis_time_vec[i]);
    }

    int64_t cut_mean = 0, cut_stdev = 0;
    double maxbal_mean = 0.0, maxbal_stdev = 0.0;
    double mtmetis_time_mean = 0.0, mtmetis_time_stdev = 0.0;
    for (int i = 0; i < n_trial; ++i){
        cut_mean += cut_vec[i];
        maxbal_mean += maxbal_vec[i];
        mtmetis_time_mean += mtmetis_time_vec[i];
    }
    cut_mean = std::round(((double)cut_mean) / n_trial);
    maxbal_mean /= n_trial;
    mtmetis_time_mean /= n_trial;
    for (int i = 0; i < n_trial; ++i){
        cut_stdev += (cut_vec[i] - cut_mean)*(cut_vec[i] - cut_mean);
        maxbal_stdev += (maxbal_vec[i] - maxbal_mean)*(maxbal_vec[i] - maxbal_mean);
        mtmetis_time_stdev += (mtmetis_time_vec[i] - mtmetis_time_mean)*(mtmetis_time_vec[i] - mtmetis_time_mean);
    }
    cut_stdev = std::round(std::sqrt(((double)cut_stdev) / n_trial));
    maxbal_stdev = std::sqrt(maxbal_stdev / n_trial);
    mtmetis_time_stdev = std::sqrt(mtmetis_time_stdev / n_trial);

    printf("mtmetis cut mean %d %d\n", cut_mean, cut_stdev);
    printf("mtmetis maxbal mean %lf %lf\n", maxbal_mean, maxbal_stdev);
    printf("mtmetis time %lf %lf\n", mtmetis_time_mean, mtmetis_time_stdev);

    if ( !json_file_path.empty() ){
        std::string mat_name = Sppart::get_filename_wo_ext(matrix_file_path);
        nlohmann::json json;
        json["git hash"] = GIT_COMMIT_HASH;;
        json["method"] = "mtmetis";
        json["mat"] = mat_name;
        json["npart"] = nparts;
        json["nthreads"] = nthreads;
        json["ntry"] = n_trial;
        json["param"]["seed"] = rand_seed;
        json["param"]["ub"] = ubfactor;
        json["result"]["cut"]["mean"] = cut_mean;
        json["result"]["cut"]["std"] = cut_stdev;
        json["result"]["maxbal"]["mean"] = maxbal_mean;
        json["result"]["maxbal"]["std"] = maxbal_stdev;
        json["result"]["time"]["total"]["mean"] = mtmetis_time_mean;
        json["result"]["time"]["total"]["std"] = mtmetis_time_stdev;
        std::ofstream fs(json_file_path);
        fs << json.dump(4) << std::endl;
    }

    return 0;
}

