/*
 * python_wrapper.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "PhaseAFO.h"

#include <boost/python.hpp>

using namespace boost::python;


void integrate_afo(PhaseAFO& afo, double t_init, double t_end,
                             numeric::array& init, double dt=0.001,
                             double save_dt=0.001)
{
//  Eigen::Vector2d eigen_init;
//  eigen_init(0) = extract<double>(init[0]);
//  eigen_init(1) = extract<double>(init[1]);
//
//  Eigen::MatrixXd y;
//  Eigen::VectorXd t;
//
//  afo.integrate(t_init, t_end, eigen_init, t, y, dt, save_dt);
//
//  numeric::array res(3,int(t.size()));
//  for(int i=0; i<t.size(); ++i)
//  {
//    res[make_tuple(0,i)] = t(i);
//    res[make_tuple(1,i)] = y(0,i);
//    res[make_tuple(2,i)] = y(1,i);
//  }
//  return res;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(integrate_afo_overloads, integrate_afo, 4, 6)


BOOST_PYTHON_MODULE(libafos)
{
	class_<PhaseAFO>("PhaseAFO", init<double,double,double>());

//	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	def("integrate_afo",integrate_afo,integrate_afo_overloads());

    ;
}
