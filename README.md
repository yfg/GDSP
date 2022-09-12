# Geodesic Distance-based Spectral Graph Partitioner (GDSP)
This repo contains the code used for our [paper](https://ieeexplore.ieee.org/document/9622831)
```
Y. Futamura, R. Wakaki and T. Sakurai, "Spectral Graph Partitioning Using Geodesic Distance-based Projection," 2021 IEEE High Performance Extreme Computing Conference (HPEC), 2021, pp. 1-7, doi: 10.1109/HPEC49654.2021.9622831.
```
If you use our code in your published work, please cite the paper.

BibTeX entry:
```bibtex
@INPROCEEDINGS{9622831,
  author={Futamura, Yasunori and Wakaki, Ryota and Sakurai, Tetsuya},
  booktitle={2021 IEEE High Performance Extreme Computing Conference (HPEC)}, 
  title={Spectral Graph Partitioning Using Geodesic Distance-based Projection}, 
  year={2021},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/HPEC49654.2021.9622831}}
```

# Requirement
* C and C++11 compiler (e.g. `gcc/g++`)
* Fortran compiler supporting `ISO_C_BINDING` (e.g. `gfortran`)
* CMake 3.1+
* [Julia](https://julialang.org/) 1.3.1+ with [`MAT.jl`](https://github.com/JuliaIO/MAT.jl) package (used to read MATLAB .mat files)
* BLAS/LAPACK library (e.g. those in Intel MKL)

# Installation
```bash
$ # assume that the path to a Julia executable is in $PATH
$ mkdir build && cd build
$ cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS="-O3 -DNDEBUG" -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG"
$ make
```

# Usage
```bash
$ cd build
# Run GDSP
$ OMP_NUM_THREADS=4 ./sppart_test --mat=/path/to/graph.mat --npart=2 --ub=1.03
# See help message
$ ./bin/sppart_test --help
```

# Note
* `--npart > 2` is experimental. **Do not use for performance evaluation. It may violate the balance constraint.**
* `sppart_test` options used for parallel performance evaluation in the paper: `--npart=2 --ub=1.03 --dims=8 (or --dims=4) --limit=1000 --bfsalg=0 --dobfstd --orthalg=1 --roundalg=1 --seed=0 --srcalg=1 --xtyalg=1`

# Authors
* Yasunori Futamura, Ryota Wakaki, Tetsuya Sakurai
  * University of Tsukuba
  * futamura (at) cs.tsukuba.ac.jp

# License
Most parts of this repo are under the [MIT license](https://en.wikipedia.org/wiki/MIT_License), except for the following
* `external` directory contains following third party programs
  * [CLI11](https://github.com/CLIUtils/CLI11) by H. Schreiner et al., BSD 3-Clause license
  * [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) v5.1.0 by Univ. Minnesota, Apache 2.0 license (`CMakeLists.txt` is modifined so that it can be built as a submodule)
  * [MT-METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) v0.7.2 by Univ. Minnesota, Apache 2.0 license (`CMakeLists.txt` is modifined so that it can be built as a submodule)
  * [nlohmann/json](https://github.com/nlohmann/json) by N. Lohmann, MIT license
* `include/gapbs` contains [GAP Benchmark Suite](https://github.com/sbeamer/gapbs) by Univ. California, BSD 3-Clause <u>(modified by us)</u>
* `include/maxflow` contains [Maxflow](https://github.com/Zagrosss/maxflow) by J. Groschaft, MIT license <u>(modified by us)</u>
* `./FindJulia.cmake` is from [xtensor-julia-cookiecutter](https://github.com/xtensor-stack/xtensor-julia-cookiecutter) by J. Mabille et al., BSD 3-Clause
* `./FindMKL.cmake` is from [PyTorch](https://github.com/pytorch/pytorch) by Facebook Inc., BSD-style license
