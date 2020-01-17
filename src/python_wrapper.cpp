/*
 * python_wrapper.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

//#define PY_ARRAY_UNIQUE_SYMBOL afos_ARRAY_API
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "PhaseAFO.h"


namespace py = pybind11;

using namespace afos;

PYBIND11_MODULE(pyafos, m)
{
  py::class_<PhaseAFO>(m,"PhaseAFO")
	.def(py::init<>())
	.def("integrate", &PhaseAFO::integrate)
    .def("initialize_sine", &PhaseAFO::init_sine)
    .def("initialize_vec_of_sines", &PhaseAFO::init_vec_of_sines)
    .def("initialize_frequency_changing_sine", &PhaseAFO::init_frequency_changing_sine)
	.def("t", &PhaseAFO::t, py::return_value_policy::reference_internal)
	.def("y", &PhaseAFO::y, py::return_value_policy::reference_internal);
}
