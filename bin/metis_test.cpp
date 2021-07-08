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
#include <util.hpp>
#include <json.hpp>
#include <fstream>
#include <version.hpp>

JULIA_DEFINE_FAST_TLS() // only define this once, in an executable (not in a shared library) if you want fast code.

int main(int argc, char* argv[]){
    CLI::App app("metis_test");
    
    std::string matrix_file_path = "";
    int nparts = 2;
    real_t ubfactor = 1.001;
    int rand_seed = 0;
    std::string json_file_path = "";
    int n_trial = 1;
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

    std::vector<int> xadj_metis(nv+1);
    for (int i = 0; i < nv+1; ++i){
        xadj_metis[i] = xadj[i];
    }

    int nvtxs = nv;
    int ncon = 1;
    int objval = -1;
    int ret;

    std::vector<int64_t> cut_vec(n_trial);
    std::vector<double> maxbal_vec(n_trial);
    std::vector<double> metis_time_vec(n_trial);

    std::vector<int> part(nv);
    printf("Git hash %s\n", GIT_COMMIT_HASH);
    for (int i = 0; i < n_trial; ++i){
        int metis_options[METIS_NOPTIONS];
        ret = METIS_SetDefaultOptions(metis_options);
        // metis_options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO;
        metis_options[METIS_OPTION_SEED] = rand_seed + i;
        metis_time_vec[i] = Sppart::timeit([&]{
            ret = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj_metis.data(), adjncy.data(), NULL, NULL, NULL, &nparts, NULL, &ubfactor, metis_options, &objval, part.data());
        });

        int cut2 = Sppart::compute_cut(nv, xadj.data(), adjncy.data(), part.data());
        maxbal_vec[i] = Sppart::compute_maxbal(nparts, nv, xadj.data(), adjncy.data(), part.data());

        if ( ret != 1 || objval != cut2 ) {
            printf("Something wrong with METIS !!!\n");
            std::terminate();
        }

        cut_vec[i] = objval;
        printf("metis cut %d %d\n", ret, objval);
        printf("cut2 %d\n", cut2);
        printf("metis maxbal %lf\n", maxbal_vec[i]);
        printf("metis time %lf\n", metis_time_vec[i]);
    }

    int64_t cut_mean = 0, cut_stdev = 0;
    double maxbal_mean = 0.0, maxbal_stdev = 0.0;
    double metis_time_mean = 0.0, metis_time_stdev = 0.0;
    for (int i = 0; i < n_trial; ++i){
        cut_mean += cut_vec[i];
        maxbal_mean += maxbal_vec[i];
        metis_time_mean += metis_time_vec[i];
    }
    cut_mean = std::round(((double)cut_mean) / n_trial);
    maxbal_mean /= n_trial;
    metis_time_mean /= n_trial;
    for (int i = 0; i < n_trial; ++i){
        cut_stdev += (cut_vec[i] - cut_mean)*(cut_vec[i] - cut_mean);
        maxbal_stdev += (maxbal_vec[i] - maxbal_mean)*(maxbal_vec[i] - maxbal_mean);
        metis_time_stdev += (metis_time_vec[i] - metis_time_mean)*(metis_time_vec[i] - metis_time_mean);
    }
    cut_stdev = std::round(std::sqrt(((double)cut_stdev) / n_trial));
    maxbal_stdev = std::sqrt(maxbal_stdev / n_trial);
    metis_time_stdev = std::sqrt(metis_time_stdev / n_trial);

    printf("metis cut mean %d %d\n", cut_mean, cut_stdev);
    printf("metis maxbal mean %lf %lf\n", maxbal_mean, maxbal_stdev);
    printf("metis time %lf %lf\n", metis_time_mean, metis_time_stdev);

    if ( !json_file_path.empty() ){
        std::string mat_name = Sppart::get_filename_wo_ext(matrix_file_path);
        nlohmann::json json;
        json["git hash"] = GIT_COMMIT_HASH;;
        json["method"] = "metis";
        json["mat"] = mat_name;
        json["npart"] = nparts;
        json["ntry"] = n_trial;
        json["param"]["seed"] = rand_seed;
        json["param"]["ub"] = ubfactor;
        json["result"]["cut"]["mean"] = cut_mean;
        json["result"]["cut"]["std"] = cut_stdev;
        json["result"]["cut"]["all"] = cut_vec;
        json["result"]["maxbal"]["mean"] = maxbal_mean;
        json["result"]["maxbal"]["std"] = maxbal_stdev;
        json["result"]["maxbal"]["all"] = maxbal_vec;
        json["result"]["time"]["total"]["mean"] = metis_time_mean;
        json["result"]["time"]["total"]["std"] = metis_time_stdev;
        json["result"]["time"]["total"]["all"] = metis_time_vec;
        std::ofstream fs(json_file_path);
        fs << json.dump(4) << std::endl;
    }

    return 0;
}

