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
#include "DualPhasePhaseAFO.h"
#include "PoolAFO.h"
#include "Integrator.h"


namespace py = pybind11;

using namespace afos;

PYBIND11_MODULE(pyafos, m)
{
  m.def("integrate", &euler_integration<PhaseAFO>);
  m.def("integrate", &euler_integration<DualPhasePhaseAFO>);
  m.def("integrate", &euler_integration<PoolAFO>);

  m.def("integrateRK", &rungekutta_integration<PhaseAFO>);
  m.def("integrateRK", &rungekutta_integration<DualPhasePhaseAFO>);

  py::class_<PhaseAFO>(m,"PhaseAFO")
	  .def(py::init<>())
    .def("initialize", &PhaseAFO::initialize)
    .def("input", &PhaseAFO::input, py::return_value_policy::reference_internal)
    .def("t", &PhaseAFO::t, py::return_value_policy::reference_internal)
	  .def("y", &PhaseAFO::y, py::return_value_policy::reference_internal);

  py::class_<InputPerturbation>(m, "InputPerturbation")
    .def(py::init<>())
    .def("sine", &InputPerturbation::sine)
    .def("vec_of_sines", &InputPerturbation::vec_of_sines)
    .def("frequency_changing_sine", &InputPerturbation::frequency_changing_sine)
    .def("chirps_and_exponentials", &InputPerturbation::chirps_and_exponentials)
    .def("get", &InputPerturbation::get);    

  py::class_<DualPhasePhaseAFO>(m,"DualPhasePhaseAFO")
	  .def(py::init<>())
    .def("initialize", &DualPhasePhaseAFO::initialize)
    .def("input", &DualPhasePhaseAFO::input, py::return_value_policy::reference_internal)
    .def("t", &DualPhasePhaseAFO::t, py::return_value_policy::reference_internal)
	  .def("y", &DualPhasePhaseAFO::y, py::return_value_policy::reference_internal);

  py::class_<PoolAFO>(m,"PoolAFO")
	  .def(py::init<>())
    .def("initialize", &PoolAFO::initialize)
    .def("input", &PoolAFO::input, py::return_value_policy::reference_internal)
    .def("t", &PoolAFO::t, py::return_value_policy::reference_internal)
	  .def("y", &PoolAFO::y, py::return_value_policy::reference_internal);
}
