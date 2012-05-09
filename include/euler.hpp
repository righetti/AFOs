/*
 * euler.h
 *
 *  Created on: May 8, 2012
 *      Author: righetti
 */

#ifndef EULER_H_
#define EULER_H_

#include <vector>

namespace afos
{

template<class ODE, class Recorder>
size_t euler_integrate(ODE& eqs, const std::vector<double>& x_init, double time_init,
                        double time_end, double dt, Recorder& rec, double dt_save = 0.1)
{
  size_t num_steps = 0;

  std::vector<double> x = x_init;
  double t = time_init;

  rec(x_init, time_init);

  double deltaT = 0.0;

  for(size_t i=0; i<size_t((time_end-time_init)/dt); ++i)
  {
    euler_integrate_step(eqs, x, t, dt);
    deltaT += dt;
    //save at a slower pace
    if(deltaT >= dt_save)
    {
      rec(x, t);
      deltaT = 0.0;
      num_steps++;
    }
  }

  return num_steps;
}

template<class ODE>
void euler_integrate_step(ODE& eqs, std::vector<double>& x, double& t, double dt)
{
  std::vector<double> dxdt(x.size());
  eqs(x, dxdt, t);
  for(int i=0; i<(int)x.size(); ++i)
  {
    x[i] += dxdt[i] * dt;
  }
  t+= dt;
}


} /* namespace inverse_dynamics */
#endif /* EULER_H_ */
