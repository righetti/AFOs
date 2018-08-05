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

/*
py::object integrate_afo(double t_init, double t_end,
                                    double K, double lambda, np::ndarray& init,
                                    np::ndarray& freq, np::ndarray& amp, np::ndarray& phase,
                                    double dt=0.001, double save_dt=0.001)
{
  PhaseAFO afo;

  Eigen::Vector2d eigen_init;
  eigen_init(0) = extract<double>(init[0]);
  eigen_init(1) = extract<double>(init[1]);

  Eigen::VectorXd freq_eigen, amp_eigen, phase_eigen;
  int num = extract<int>(freq.attr("size"));
  freq_eigen.resize(num);
  amp_eigen.resize(num);
  phase_eigen.resize(num);

  for(int i=0; i<num; ++i)
  {
    freq_eigen(i) = extract<double>(freq[i]);
    amp_eigen(i) = extract<double>(amp[i]);
    phase_eigen(i) = extract<double>(phase[i]);
  }

  afo.init(K, freq_eigen, amp_eigen, phase_eigen, lambda);

  Eigen::MatrixXd y;
  Eigen::VectorXd t;
  afo.integrate(t_init, t_end, eigen_init, t, y, dt, save_dt);

  long N[2];
  N[0] = 3;
  N[1] = long(t.size());
  tuple shape = make_tuple(N[0],N[1]);
  np::dtype dtype = np::dtype::get_builtin<double>();
  np::ndarray res = np::zeros(shape, dtype);
//  np::ndarray res = (static_cast<np::ndarray>(handle<>(PyArray_SimpleNew(nd,N,NPY_DOUBLE))));

  for(int i=0; i<t.size(); ++i)
  {
    res[make_tuple(0,i)] = t(i);
    res[make_tuple(1,i)] = y(0,i);
    res[make_tuple(2,i)] = y(1,i);
  }

  return res.copy(); //copy the object so numpy owns the copy
}
*/

PYBIND11_MODULE(afos, m)
{
  py::class_<PhaseAFO>(m,"PhaseAFO")
	.def(py::init<>())
	.def("integrate", &PhaseAFO::integrate)
	.def("initialize", &PhaseAFO::init)
	.def("initialize_vec", &PhaseAFO::init1)
	.def("t", &PhaseAFO::t, py::return_value_policy::reference_internal)
	.def("y", &PhaseAFO::y, py::return_value_policy::reference_internal);
}
