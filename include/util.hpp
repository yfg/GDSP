// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#pragma once
#include<chrono>
#include<functional>

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
}