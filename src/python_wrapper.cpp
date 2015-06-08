/*
 * python_wrapper.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#define PY_ARRAY_UNIQUE_SYMBOL afos_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <cstdio>

#include "PhaseAFO.h"


using namespace boost::python;


boost::python::object integrate_afo(double t_init, double t_end,
                                    double K, double lambda, numeric::array& init,
                                    numeric::array& freq, numeric::array& amp, numeric::array& phase,
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
  int nd = 2;
  numeric::array res = (static_cast<numeric::array>(handle<>(PyArray_SimpleNew(nd,N,NPY_DOUBLE))));

  for(int i=0; i<t.size(); ++i)
  {
    res[make_tuple(0,i)] = t(i);
    res[make_tuple(1,i)] = y(0,i);
    res[make_tuple(2,i)] = y(1,i);
  }

  return res.copy(); //copy the object so numpy owns the copy
}

BOOST_PYTHON_FUNCTION_OVERLOADS(integrate_afo_overloads, integrate_afo, 8, 10)

BOOST_PYTHON_MODULE(afos)
{
  //important to have correct array passing between python and c++
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();


  class_<PhaseAFO>("PhaseAFO", no_init)
  	                    .enable_pickling();
  def("integrate_afo",integrate_afo,integrate_afo_overloads());
}
