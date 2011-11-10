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


int main(int argc, char** argv)
{
  std::vector<double> x_init;
  x_init.resize(2);
  x_init[0] = 0.0;
  x_init[1] = 10.0;

  std::vector<std::vector<double> > x_result;
  std::vector<double> time;

  afos::AdaptivePhaseOscillator afo(30.0, 1000.0);
  afos::StateRecorder<std::vector<double> > recorder(x_result, time);

  size_t steps = boost::numeric::odeint::integrate(afo, x_init, 0.0, 10.0, 0.1, recorder);


  std::fstream my_file("test_result.dat", std::fstream::out);

  for(size_t i=0; i<=steps; i++)
  {
    my_file << time[i] << '\t'
        << x_result[i][0] << '\t' << x_result[i][1] << '\n';
  }
  my_file.close();

  return 0;
}
