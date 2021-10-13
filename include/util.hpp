// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#pragma once
#include<chrono>
#include<functional>
#include <sstream>

namespace Sppart {

    template<class TYPE>
    std::unique_ptr<TYPE[]> create_up_array(size_t n){
        return std::unique_ptr<TYPE[]>(new TYPE[n]);
    }
    
    class Timer{
    private:
        std::chrono::system_clock::time_point start_point;
        std::chrono::system_clock::time_point end_point;
        bool running;

    public:
        Timer() : running(false){}
        void start(){
            if ( running ){
                printf("Warning: Timer.start() is called although the timer is running\n");
            }
            start_point = std::chrono::system_clock::now();
            running = true;
        }
        void stop(){
            if ( !running ){
                printf("Warning: Timer.stop() is called although the timer is not running\n");
            }
            end_point = std::chrono::system_clock::now();
            running = false;
        }
        double get_time(){
            if ( running ){
                printf("Warning: Timer.get_time() is called although the timer is not running\n");
            }
            return std::chrono::duration_cast<std::chrono::milliseconds>(end_point-start_point).count()/1000.0;            
        }
        bool is_running(){
            return running;
        }
        void clear(){
            running = false;
        }
    };

    double timeit(std::function<void()> f){
        auto start = std::chrono::system_clock::now();
        f();
        auto end = std::chrono::system_clock::now();;
        return std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()/1000.0;
    }

    // get filename without extension from file path
    std::string get_filename_wo_ext(std::string& file_path){
        std::stringstream ss(file_path);
        std::string str;
        std::vector<std::string> vec;
        vec[vec.size()-1];
        while ( getline(ss, str, '/') ){
            vec.push_back(str);
        }
        str = vec[vec.size()-1];
        vec.clear();
        std::stringstream ss2(str);
        while ( getline(ss2, str, '.') ){
            vec.push_back(str);
        }
        return vec[0];
    }

    // Create a graph of the upper triangular part of the input graph
    // Input graph must be undirected and has no self loop
    template<class XADJ_INT>
    void create_upper_triangular(const int nv, const std::vector<XADJ_INT> &in_xadj, const std::vector<int> &in_adjncy, std::vector<XADJ_INT> &out_xadj, std::vector<int> &out_adjncy){
        out_xadj.resize(in_xadj.size());
        out_adjncy.resize(in_adjncy.size() / 2);

        XADJ_INT cnt = 0;
        for (int i = 0; i < nv; ++i){
            out_xadj[i] = cnt;
            for (XADJ_INT k = in_xadj[i]; k < in_xadj[i+1]; ++k){
                const int j = in_adjncy[k];
                if ( j > i ){
                    out_adjncy[cnt] = j;
                    cnt++;
                }
            }
        }
        out_xadj[nv] = cnt;
    }

}