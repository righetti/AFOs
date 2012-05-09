/*
 * test_afo_sys_init.cpp
 *
 *  Created on: Nov 9, 2011
 *      Author: righetti
 */

#include <iostream>
#include <fstream>
#include <vector>

#include <boost/numeric/odeint.hpp>

#include <state_recorder.hpp>
#include <adaptive_phase_oscillator.h>

#include <euler.hpp>

using namespace boost::numeric::odeint;

int main(int argc, char** argv)
{
  std::vector<double> x_init;
  x_init.resize(2);


  double d_phi = 2*M_PI;
  double T = 100.0;
  double dt = 0.0001;
  double dt_save = 0.0001;
  double omegaF = 30.0;
  double K = 1000.0;
  double omega0 = 10.0;

  double range = 2*M_PI;

#pragma omp parallel for
  for(int j=0; j<int(range/d_phi);j++)
  {
    x_init[0] = double(j)*d_phi;
    x_init[1] = omega0;

    std::vector<std::vector<double> > x_result;
    std::vector<double> time;

    afos::AdaptivePhaseOscillator afo(omegaF, K);
    afos::StateRecorder<std::vector<double> > recorder(x_result, time);
    //        euler<std::vector<double> > euler_stepper;
    //        runge_kutta4<std::vector<double> >  my_stepper;


    std::cout << "iteration " << j << std::endl;
    //        size_t steps = integrate_const(my_stepper, afo,
    //                                       x_init, 0.0, T, dt, recorder);
    size_t steps = afos::euler_integrate(afo, x_init, 0.0, T, dt, recorder, dt_save);


    std::stringstream filename;
    filename << "test_result" << j << ".dat";

    std::fstream my_file(filename.str().c_str(), std::fstream::out);

    for(size_t i=0; i<=steps; i++)
    {
      my_file << time[i] << '\t'
          << x_result[i][0] << '\t' << x_result[i][1] << '\n';
    }
    my_file.close();

  }

  return 0;
}
