cmake_minimum_required(VERSION 3.0)

project(afos)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall -O3 -DEIGEN_NO_DEBUG")
add_compile_options(-std=c++11)
SET(CMAKE_VERBOSE_MAKEFILE on )

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${EXECUTABLE_OUTPUT_PATH})

include_directories(/usr/local/include/eigen3/)

add_library(afos STATIC src/PhaseAFO.cpp src/InputPerturbation.cpp, src/DualPhasePhaseAFO.cpp)

ADD_EXECUTABLE(test_afo
                src/test_phase_afo.cpp)

TARGET_LINK_LIBRARIES(test_afo afos)


set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 REQUIRED)

pybind11_add_module(pyafos src/python_wrapper.cpp)

target_link_libraries(pyafos PRIVATE afos)

install(TARGETS  pyafos LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/python)
