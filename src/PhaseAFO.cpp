/*
 * PhaseAFO.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "PhaseAFO.h"

#include <cmath>

PhaseAFO::PhaseAFO(double K, double omegaF, double lambda)
{
  K_ = K;
  omegaF_ = omegaF;
  lambda_ = lambda;
}

PhaseAFO::~PhaseAFO()
{
}

Eigen::Vector2d PhaseAFO::dydt(const Eigen::Vector2d& y, double t)
{
  Eigen::Vector2d dydt;
  dydt(0) = lambda_ * y(1) - K_ * sin(y(0)) * cos(omegaF_*t);
  dydt(1) = - K_ * sin(y(0)) * cos(omegaF_*t);

  return dydt;
}

void PhaseAFO::integrate(double t_init, double t_end,
                         const Eigen::Vector2d& init,
                         Eigen::VectorXd& t, Eigen::MatrixXd& y,
                         double dt, double save_dt)
{
  int inner_loop = int(save_dt/dt);
  int length = int((t_end - t_init)/save_dt);
  y.resize(2,length);
  t.resize(length);
  y.col(0) = init;

  for(int i=1; i<length; ++i)
  {
    Eigen::Vector2d y_tmp;
    y_tmp = y.col(i-1);
    for(int j=0; j<inner_loop; ++j)
      y_tmp += dydt(y_tmp,t(i-1)+double(j)*dt) * dt;
    y.col(i) = y_tmp;
    t(i) = t(i-1) + save_dt;
  }

}
