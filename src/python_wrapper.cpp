/*
 * python_wrapper.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "PhaseAFO.h"

#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(afos)
{
    class_<PhaseAFO>("PhaseAFO", init<double,double,double>())
        .def("getK", &PhaseAFO::getK)
    ;
}
