// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#include<iostream>
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <vector>
#include <julia.h>
#include <CLI11/CLI11.hpp>
#include <metis.h>
#include <omp.h>

#include <matrix_read.hpp>
#include <json.hpp>
#include <fstream>
#include <sppart.hpp>
#include <version.hpp>

JULIA_DEFINE_FAST_TLS() // only define this once, in an executable (not in a shared library) if you want fast code.

int main(int argc, char* argv[]){
    CLI::App app("sppart test");
    // using XADJ_INT = int64_t;
    using XADJ_INT = int;
    
    std::string matrix_file_path = "";
    bool use_single = false;
    Sppart::InputParams params;
    int nparts = 2;
    int random_seed = 0;
    std::string json_file_path = "";
    int n_trial = 1;
    bool fix_seed = false;
    app.add_option("--mat", matrix_file_path, "File path of MATLAB mat file for input graph (matrix) from SuiteSparse Matrix Collection")
        ->required(true)
        ->check(CLI::ExistingFile);
    app.add_option("--npart", nparts, "Number of part for partitioning")
        ->default_val(2);
    app.add_option("--ub", params.ubfactor, "Unbalance tolerance")
        ->default_val(1.001);
    app.add_option("--dims", params.n_dims, "Dimension of high dimensional embedding")
        ->default_val(8)
        ->check(CLI::Range(1,1000));
    app.add_option("--srcalg", params.src_alg, "Algorithm for choosing SSSP sources")
        ->default_val(0);
    app.add_option("--bfsalg", params.bfs_alg, "Algorithm of BFS")
        ->default_val(0);
    app.add_option("--maxpass", params.fm_max_pass, "Maximum number of passes of FM refinement")
        ->default_val(10);
        // ->check(CLI::Range(0,10000));
    app.add_option("--limit", params.fm_limit, "limit for FM refinement")
        ->default_val(10000);
    app.add_flag("--nolimit", params.fm_no_limit, "Use no limit mode for FM refinement")
        ->default_val(false);
    app.add_option("--seed", random_seed, "Seed for random number generator")
        ->default_val(0);
    app.add_flag("--use_single_precision", use_single, "Use signle precision instead of double precision")
        ->default_val(false);
    app.add_option("--roundalg", params.round_alg, "Rounding method for initialize partitioning with Fiedler vector")
        ->default_val(0);
    app.add_option("--json", json_file_path, "File path for output JSON file");
    app.add_option("--ntry", n_trial, "Number of trials")
        ->default_val(1);
    app.add_option("--orthalg", params.orth_alg, "Algorithm for orthonormalization")
        ->default_val(0);
    app.add_option("--xtyalg", params.xty_alg, "Algorithm for transpose(X)*Y")
        ->default_val(0);
    app.add_flag("--dobfstd", params.dobfs_td, "Force direction optimaized BFS to always use top-down scheme")
        ->default_val(false);
    app.add_flag("--fixseed", fix_seed, "Fix random seed for multiple trials")
        ->default_val(false);
    app.add_option("--fmmaxmove", params.fm_max_move, "Maximum move counts for FM refinement. If it is negative, treated as the number of vertices.")
        ->default_val(-1);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);;
    }

    int nv0;
    std::vector<XADJ_INT> xadj0;
    std::vector<int> adjncy0;
    double time_matread = Sppart::timeit([&]{
        Sppart::matrix_read(matrix_file_path, nv0, xadj0, adjncy0); });

    int nv;
    std::vector<XADJ_INT> xadj;
    std::vector<int> adjncy;
    double time_preprocess = Sppart::timeit([&]{
        Sppart::preprocess_graph(nv0, xadj0, adjncy0, nv, xadj, adjncy); });

    Sppart::Graph<XADJ_INT> g(nv, xadj.data(), adjncy.data());

    const int nthreads = omp_get_max_threads();
    printf("Git hash %s\n", GIT_COMMIT_HASH);
    printf("------ Execution info ------------\n");
    printf("Num threads = %d\n", nthreads);
    printf("Num parts = %d\n", nparts);

    printf("------ Graph info ----------------\n");
    printf("Original nv = %d\n", nv0);
    printf("Original ne = %d\n", xadj0[nv0]/2);
    printf("nv          = %d\n", nv);
    printf("ne          = %d\n", xadj[nv]/2);

    printf("------ Timing result preprocess --\n");
    printf("%-12s %10.4lf\n", "matread", time_matread);
    printf("%-12s %10.4lf\n", "preprocess", time_preprocess);

    std::vector<int64_t> cut_vec(n_trial);
    std::vector<double> maxbal_vec(n_trial);
    std::vector<double> total_time_vec(n_trial);

    Sppart::OutputInfo info;
    double time_total_tmp;
    params.rand_seed = random_seed;
    for (int i = 0; i < n_trial; ++i){
        std::unique_ptr<int[]> output_partition;

        time_total_tmp = Sppart::timeit([&]{
            if ( use_single ){
                Sppart::RecursivePartitioner<XADJ_INT,float> partitioner(g, params);
                info = partitioner.run_partitioning(nparts);
                output_partition = partitioner.get_partition();
            }else{
                Sppart::RecursivePartitioner<XADJ_INT,double> partitioner(g, params);
                info = partitioner.run_partitioning(nparts);
                output_partition = partitioner.get_partition();
            }
        });    
        double final_maxbal = Sppart::compute_maxbal(nparts, g.nv, g.xadj, g.adjncy, output_partition.get());
        int cut2 = Sppart::compute_cut(g.nv, g.xadj, g.adjncy, output_partition.get());

        printf("------ Sppart params -------------\n");
        if ( nparts == 2 ){
            printf("pass count      : %d\n", info.fm_pass_count);
            printf("Spectral cut    : %lld\n", info.spectral_cut);
            printf("Spectral maxbal : %lf\n", info.spectral_maxbal);
        }
        printf("Balance cut       : %lld\n", info.balance_cut);
        printf("Balance maxbal    : %lf\n", info.balance_maxbal);
        printf("Final cut       : %lld\n", info.cut);
        printf("Final maxbal    : %lf\n", final_maxbal);

        if ( info.cut != cut2 ) {
            printf("Something wrong with Sppart !!!\n");
            std::terminate();
        }

        cut_vec[i] = info.cut;
        maxbal_vec[i] = final_maxbal;
        total_time_vec[i] = time_total_tmp;

        if ( ! fix_seed ) params.rand_seed++;
    }
    // now variable "info" is that of the final trial

    int64_t cut_mean = 0, cut_stdev = 0;
    double maxbal_mean = 0.0, maxbal_stdev = 0.0;
    double total_time_mean = 0.0, total_time_stdev = 0.0;
    for (int i = 0; i < n_trial; ++i){
        cut_mean += cut_vec[i];
        maxbal_mean += maxbal_vec[i];
        total_time_mean += total_time_vec[i];
    }
    cut_mean = std::round(((double)cut_mean) / n_trial);
    maxbal_mean /= n_trial;
    total_time_mean /= n_trial;
    for (int i = 0; i < n_trial; ++i){
        cut_stdev += (cut_vec[i] - cut_mean)*(cut_vec[i] - cut_mean);
        maxbal_stdev += (maxbal_vec[i] - maxbal_mean)*(maxbal_vec[i] - maxbal_mean);
        total_time_stdev += (total_time_vec[i] - total_time_mean)*(total_time_vec[i] - total_time_mean);
    }
    cut_stdev = std::round(std::sqrt(((double)cut_stdev) / n_trial));
    maxbal_stdev = std::sqrt(maxbal_stdev / n_trial);
    total_time_stdev = std::sqrt(total_time_stdev / n_trial);

    printf("------ Timing result sppart ------\n");
    printf("%-12s %10.3lf\n", "Total", time_total_tmp);
    printf("    %-12s %10.3lf\n", "connect", info.time_connect);
    printf("    %-12s %10.3lf\n", "spectral", info.time_spectral);
    printf("        %-12s %10.3lf\n", "bfs", info.time_spectral_bfs);
    printf("        %-12s %10.3lf\n", "spmm", info.time_spectral_spmm);
    printf("        %-12s %10.3lf\n", "sumzero", info.time_spectral_sumzero);
    printf("        %-12s %10.3lf\n", "orth", info.time_spectral_orth);
    printf("        %-12s %10.3lf\n", "XtY", info.time_spectral_XtY);
    printf("        %-12s %10.3lf\n", "eig", info.time_spectral_eig);
    printf("        %-12s %10.3lf\n", "back", info.time_spectral_back);
    printf("        %-12s %10.3lf\n", "round", info.time_spectral_round);
    printf("    %-12s %10.3lf\n", "balance", info.time_balance);
    printf("    %-12s %10.3lf\n", "fm", info.time_fm);
    printf("    %-12s %10.3lf\n", "split", info.time_split);

    printf("sppart cut mean %d %d\n", cut_mean, cut_stdev);
    printf("sppart maxbal mean %lf %lf\n", maxbal_mean, maxbal_stdev);
    printf("sppart time %lf %lf\n", total_time_mean, total_time_stdev);

    if ( !json_file_path.empty() ){
        std::string mat_name = Sppart::get_filename_wo_ext(matrix_file_path);
        nlohmann::json json;
        json["git hash"] = GIT_COMMIT_HASH;;
        json["method"] = "sppart";
        json["mat"] = mat_name;
        json["npart"] = nparts;
        json["nthreads"] = nthreads;
        json["ntry"] = n_trial;
        json["fixseed"] = fix_seed;
        json["param"]["ub"] = params.ubfactor;
        json["param"]["seed"] = random_seed;
        json["param"]["dims"] = params.n_dims;
        json["param"]["srcalg"] = params.src_alg;
        json["param"]["bfsalg"] = params.bfs_alg;
        json["param"]["maxpass"] = params.fm_max_pass;
        json["param"]["limit"] = params.fm_limit;
        json["param"]["nolimit"] = params.fm_no_limit;
        json["param"]["use_single_precision"] = use_single;
        json["param"]["roundalg"] = params.round_alg;
        json["param"]["orthalg"] = params.orth_alg;
        json["param"]["xtyalg"] = params.xty_alg;
        json["param"]["dobfstd"] = params.dobfs_td;
        json["result"]["cut"]["mean"] = cut_mean;
        json["result"]["cut"]["std"] = cut_stdev;
        json["result"]["cut"]["all"] = cut_vec;
        json["result"]["maxbal"]["mean"] = maxbal_mean;
        json["result"]["maxbal"]["std"] = maxbal_stdev;
        json["result"]["maxbal"]["all"] = maxbal_vec;
        json["result"]["time"]["total"]["mean"] = total_time_mean;
        json["result"]["time"]["total"]["std"] = total_time_stdev;
        json["result"]["time"]["total"]["last"] = time_total_tmp;
        json["result"]["time"]["total"]["all"] = total_time_vec;
        json["result"]["time"]["connect"] = info.time_connect;
        json["result"]["time"]["spectral"]["total"] = info.time_spectral;
        json["result"]["time"]["spectral"]["bfs"] = info.time_spectral_bfs;
        json["result"]["time"]["spectral"]["spmm"] = info.time_spectral_spmm;
        json["result"]["time"]["spectral"]["sumzero"] = info.time_spectral_sumzero;
        json["result"]["time"]["spectral"]["orth"] = info.time_spectral_orth;
        json["result"]["time"]["spectral"]["XtY"] = info.time_spectral_XtY;
        json["result"]["time"]["spectral"]["eig"] = info.time_spectral_eig;
        json["result"]["time"]["spectral"]["back"] = info.time_spectral_back;
        json["result"]["time"]["spectral"]["round"] = info.time_spectral_round;
        json["result"]["time"]["balance"] = info.time_balance;
        json["result"]["time"]["fm"] = info.time_fm;
        json["result"]["time"]["split"] = info.time_split;
        std::ofstream fs(json_file_path);
        fs << json.dump(4) << std::endl;
    }
    return 0;
}

