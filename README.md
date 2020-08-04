# Code for the paper Slow-fast dynamics of strongly coupled adaptive frequency oscillators

This repository contains code to simulate adaptive frequency oscillators and pool of these oscillators

## Material from the paper
This repository contains the code to generate all the experiments and figures in the python/ directory

## C++ AFOs
A C++ library (src/) has fast numerical implementations of all the examples shown in the paper
It also contains python wrappers so the classes can be called from python

## DEPENDENCIES
* cmake and c++11
* pybind11 (for python bindings)
* numpy/scipy/matplotlib (python)
* numba (python compilation of code)

## INSTALLATION
in the root directory
```
mkdir build
cd build
cmake ..
make
make install
```
after this, there should be a pyafos library contained in the python folder
the libafos.a (in the build folder) is the static C++ library that can be used as well for C++ code

