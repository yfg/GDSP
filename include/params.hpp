// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#pragma once
namespace Sppart{

    struct InputParams {
        int n_dims = 8;
        int fm_max_pass = 10;
        int fm_limit = 10000;
        bool fm_no_limit = false;
        int rand_seed = 0;
        int src_alg = 0;
        int bfs_alg = 0;
        int round_alg = 0;
        double ubfactor = 1.001;
    };

    struct OutputInfo {
        int64_t cut;
        double maxbal;
        int64_t spectral_cut;
        double spectral_maxbal;
        int64_t balance_cut;
        double balance_maxbal;
        int fm_pass_count;
        double time_spectral = 0.0;
        double time_spectral_bfs = 0.0;
        double time_spectral_spmm = 0.0;
        double time_spectral_sumzero = 0.0;
        double time_spectral_orth = 0.0;
        double time_spectral_XtY = 0.0;
        double time_spectral_eig = 0.0;
        double time_spectral_back = 0.0;
        double time_spectral_round = 0.0;
        double time_balance = 0.0;
        double time_fm = 0.0;
        double time_split = 0.0; 
        double time_connect = 0.0; 
    };

}
