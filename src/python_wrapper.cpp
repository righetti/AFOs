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
#include "InputPerturbation.h"


namespace py = pybind11;

using namespace afos;

PYBIND11_MODULE(pyafos, m)
{
  py::class_<PhaseAFO>(m,"PhaseAFO")
	  .def(py::init<>())
    .def("initialize", &PhaseAFO::initialize)
	  .def("integrate", &PhaseAFO::integrate)
    .def("t", &PhaseAFO::t, py::return_value_policy::reference_internal)
	  .def("y", &PhaseAFO::y, py::return_value_policy::reference_internal)
    .def("input", &PhaseAFO::input, py::return_value_policy::reference_internal);

  py::class_<InputPerturbation>(m, "InputPerturbation")
    .def(py::init<>())
    .def("sine", &InputPerturbation::sine)
    .def("vec_of_sines", &InputPerturbation::vec_of_sines)
    .def("frequency_changing_sine", &InputPerturbation::frequency_changing_sine)
    .def("get", &InputPerturbation::get);
}
