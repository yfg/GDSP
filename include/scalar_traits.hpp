// Copyright (c) 2020 Yasunori Futamura All Rights Reserved.

#ifndef SPPART_SCALAR_TRAITS_HPP
#define SPPART_SCALAR_TRAITS_HPP

#include<cmath>
#include<complex>
#include<limits>

namespace Sppart{

    template<typename T>
    struct ScalarTraits{
        //Necessary for all kinds of scalar types (Real/Complex)
        using type = T;
        using abs_type = T;

        static const bool is_complex = false;
        static inline T zero();
        static inline T one();
        static inline T conj(T val);
        static inline abs_type abs(T val);
        static inline abs_type real(T val);

        //Necessary for only real scalar types
        static inline abs_type eps();
        static inline T sqrt(T val);
        static inline T pi();
        static inline T sin(T val);
        static inline T cos(T val);
    };

    template<>
    struct ScalarTraits<float>{
        using type = float;
        using abs_type = float;

        static const bool is_complex = false;
        static inline float zero(){
            return 0.0f;
        }
        static inline float one(){
            return 1.0f;
        }
        static inline float conj(float val){
            return val;
        }
        static inline abs_type abs(float val){
            return std::abs(val);
        }
        static inline abs_type real(float val){
            return val;
        }
        static inline abs_type eps(){
            return std::numeric_limits<float>::epsilon();
        }
        static inline float sqrt(float val){
            return std::sqrt(val);
        }
        static inline float pi(){
            return 4.0f*std::atan(1.0f);
        }
        static inline float sin(float val){
            return std::sin(val);
        }
        static inline float cos(float val){
            return std::cos(val);
        }
    };

    template<>
    struct ScalarTraits<double>{
        using type = double;
        using abs_type = double;

        static const bool is_complex = false;
        static inline double zero(){
            return 0.0;
        }
        static inline double one(){
            return 1.0;
        }
        static inline double conj(double val){
            return val;
        }
        static inline abs_type abs(double val){
            return std::abs(val);
        }
        static inline abs_type real(double val){
            return val;
        }
        static inline abs_type eps(){
            return std::numeric_limits<double>::epsilon();
        }
        static inline double sqrt(double val){
            return std::sqrt(val);
        }
        static inline double pi(){
            return 4.0*std::atan(1.0);
        }
        static inline double sin(double val){
            return std::sin(val);
        }
        static inline double cos(double val){
            return std::cos(val);
        }
    };

    template<>
    struct ScalarTraits<std::complex<float>>{
        using type = std::complex<float>;
        using abs_type = float;

        static const bool is_complex = true;
        static inline std::complex<float> zero(){
            return std::complex<float>(0.0f);
        }
        static inline std::complex<float> one(){
            return std::complex<float>(1.0f);
        }
        static inline std::complex<float> conj(std::complex<float> val){
            return std::conj(val);
        }
        static inline abs_type abs(std::complex<float> val){
            return std::abs(val);
        }
        static inline abs_type real(std::complex<float> val){
            return val.real();
        }
        static inline std::complex<float> sqrt(std::complex<float> val){
            return std::sqrt(val);
        }
    };

    template<>
    struct ScalarTraits<std::complex<double>>{
        using type = std::complex<double>;
        using abs_type = double;

        static const bool is_complex = true;
        static inline std::complex<double> zero(){
            return std::complex<double>(0.0);
        }
        static inline std::complex<double> one(){
            return std::complex<double>(1.0);
        }
        static inline std::complex<double> conj(std::complex<double> val){
            return std::conj(val);
        }
        static inline abs_type abs(std::complex<double> val){
            return std::abs(val);
        }
        static inline abs_type real(std::complex<double> val){
            return val.real();
        }
        static inline std::complex<double> sqrt(std::complex<double> val){
            return std::sqrt(val);
        }
    };
}

#endif //SPPART_SCALAR_TRAITS_HPP