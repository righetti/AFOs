cmake_minimum_required(VERSION 2.6)

PROJECT(afos)


SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall -O3")
SET(CMAKE_VERBOSE_MAKEFILE on )

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${EXECUTABLE_OUTPUT_PATH})

ADD_EXECUTABLE(test_afo
                src/phase_afo.cpp)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)
