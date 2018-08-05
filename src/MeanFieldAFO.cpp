/*
 * MeanFieldAFO.cpp
 *
 *  Created on: Aug 6, 2018
 *      Author: righetti
 */

#include "MeanFieldAFO.h"

#include <cmath>
#include <cstdio>
#include <functional>

MeanFieldAFO::MeanFieldAFO() : initialized_(false){}

void MeanFieldAFO::init(int num_oscill, double K, double omegaF, double lambda)
{
  K_ = K;
  freq_.resize(1);
  amp_.resize(1);
  phase_.resize(1);

  freq_(0) = omegaF;
  amp_(0) = 1.0;
  phase_(0) = 0.0;
  lambda_ = lambda;
  initialized_ = true;
}

inline Eigen::VectorXd MeanFieldAFO::dydt(const Eigen::Vector2d& y, double t)
{
  Eigen::Vector2d dydt;
  double perturbation = ((freq_*t + phase_).array().cos()).matrix().dot(amp_);
  double tt = - K_ * sin(y(0)) * perturbation;

  dydt(0) = lambda_ * y(1) + tt;
  dydt(1) = tt;

  return dydt;
}

void MeanFieldAFO::integrate(double t_init, double t_end,
                         const Eigen::Vector2d& init,
                         double dt, double save_dt)
{
  if(!initialized_)
  {
    printf("did not initialize the object\n");
    return;
  }
  int inner_loop = int(save_dt/dt);
  int length = int((t_end - t_init)/save_dt);

  y_.resize(2,length);
  t_.resize(length);
  y_.col(0) = init;
  t_(0) = t_init;

  Eigen::Vector2d y_tmp;
  for(int i=1; i<length; ++i)
  {
    y_.col(i) = y_.col(i-1);
    for(int j=0; j<inner_loop; ++j)
      y_.col(i) += dydt(y_.col(i),t_(i-1)+double(j)*dt) * dt;
    t_(i) = t_(i-1) + save_dt;
  }

}
