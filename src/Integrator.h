/*
 * Integrator.h
 *
 *  Created on: Jan 29, 2020
 *      Author: righetti
 */


#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <eigen3/Eigen/Eigen>

namespace afos
{

template <class DNS>
void euler_integration(DNS &dyn, double t_init, double t_end,
                         const Eigen::VectorXd& init,
                         double dt=0.001, double save_dt=0.001)
{
  int inner_loop = int(save_dt/dt);
  int length = int((t_end - t_init)/save_dt);

  dyn.y_.resize(dyn.num_states(),length);
  dyn.t_.resize(length);
  dyn.y_.col(0) = init;
  dyn.t_(0) = t_init;

  for(int i=1; i<length; ++i)
  {
    dyn.y_.col(i) = dyn.y_.col(i-1);
    for(int j=0; j<inner_loop; ++j)
      dyn.y_.col(i) += dyn.dydt(dyn.y_.col(i),dyn.t_(i-1)+double(j)*dt) * dt;
    dyn.t_(i) = dyn.t_(i-1) + save_dt;
  }
}

template <class DNS>
void rungekutta_integration(DNS &dyn, double t_init, double t_end,
                         const Eigen::VectorXd& init,
                         double dt=0.001, double save_dt=0.001)
{
  int inner_loop = int(save_dt/dt);
  int length = int((t_end - t_init)/save_dt);

  dyn.y_.resize(dyn.num_states(),length);
  dyn.t_.resize(length);
  dyn.y_.col(0) = init;
  dyn.t_(0) = t_init;

  for(int i=1; i<length; ++i)
  {
    dyn.y_.col(i) = dyn.y_.col(i-1);
    for(int j=0; j<inner_loop; ++j)
      {
        auto t = dyn.t_(i-1)+double(j)*dt;
        auto k1 = dyn.dydt(dyn.y_.col(i),t);
        auto k2 = dyn.dydt(dyn.y_.col(i) + dt*0.5*k1, t + 0.5*dt);
        auto k3 = dyn.dydt(dyn.y_.col(i) + dt*0.5*k2, t + 0.5*dt);
        auto k4 = dyn.dydt(dyn.y_.col(i) + dt*k3, t + dt);
        dyn.y_.col(i) += 1/6. * dt * (k1 + 2*k2 + 2*k3 + k4);
      }
    dyn.t_(i) = dyn.t_(i-1) + save_dt;
  }
}

}
#endif /* INTEGRATOR_H */
