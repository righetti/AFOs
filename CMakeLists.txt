cmake_minimum_required(VERSION 2.6)

PROJECT(afos)


SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall -O3 -DEIGEN_NO_DEBUG")
SET(CMAKE_VERBOSE_MAKEFILE on )

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${EXECUTABLE_OUTPUT_PATH})

find_package(Boost COMPONENTS filesystem system date_time python REQUIRED)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

find_package(PythonLibs REQUIRED)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Libs of Python: " ${PYTHON_LIBRARIES} )

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
)

SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)

ADD_LIBRARY(afos SHARED src/python_wrapper.cpp src/PhaseAFO.cpp)
TARGET_LINK_LIBRARIES(afos ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

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
