cmake_minimum_required(VERSION 2.6)

PROJECT(afos)


SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall -O3 -DEIGEN_NO_DEBUG")
SET(CMAKE_VERBOSE_MAKEFILE on )

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/afos)

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${EXECUTABLE_OUTPUT_PATH})


find_package(Boost COMPONENTS filesystem system date_time python REQUIRED)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

execute_process(COMMAND python-config --ldflags OUTPUT_VARIABLE PYTHON_LD OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python-config --includes OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import numpy; print '-I' + numpy.get_include();" OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Include dirs for numpy: " ${NUMPY_INCLUDE_DIRS} )


include_directories(
    ${Boost_INCLUDE_DIRS}
)

SET(Boost_USE_STATIC_LIBS ON)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)

ADD_LIBRARY(afos SHARED src/python_wrapper.cpp src/PhaseAFO.cpp)
set_target_properties(afos
    PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    COMPILE_FLAGS "${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS}"
    LINK_FLAGS ${PYTHON_LD}
)
TARGET_LINK_LIBRARIES(afos ${Boost_LIBRARIES})

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
