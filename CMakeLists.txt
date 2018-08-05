cmake_minimum_required(VERSION 3.11)

PROJECT(afos)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall -O3 -DEIGEN_NO_DEBUG")
add_compile_options(-std=c++11)
SET(CMAKE_VERBOSE_MAKEFILE on )

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${EXECUTABLE_OUTPUT_PATH})

set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 REQUIRED)

include_directories(/usr/local/include/eigen3/)

pybind11_add_module(afos SHARED src/python_wrapper.cpp src/PhaseAFO.cpp)
set_target_properties(afos
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY python/afos
)

ADD_EXECUTABLE(test_afo
                src/test_phase_afo.cpp
                src/PhaseAFO.cpp)


# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)

endif(DOXYGEN_FOUND)
