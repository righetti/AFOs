cmake_minimum_required(VERSION 2.6)

PROJECT(AFOs)

FIND_PACKAGE(BOOST)

#INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})
#link_directories ( ${BOOST_LIBRARY_DIRS} )

INCLUDE_DIRECTORIES(include src)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../lib)

ADD_EXECUTABLE(afo_systematic_test_main
	src/afo_systematic_test_main.cpp
	src/adaptive_phase_oscillator.cpp
	include/state_recorder.hpp
)

#target_link_libraries (
#    afo_systematic_test_main
#    ${BOOST_LIBRARIES}
#)


set(CMAKE_CXX_FLAGS "-g -Wall")
