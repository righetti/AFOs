/*
 * PhaseAFO.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "PhaseAFO.h"

#include <cmath>
#include <cstdio>

PhaseAFO::PhaseAFO()
{
  initialized_ = false;
}

void PhaseAFO::init(double K, double omegaF, double lambda)
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

void PhaseAFO::init(double K, const Eigen::VectorXd& freq,
               const Eigen::VectorXd& amp, const Eigen::VectorXd& phase,
               double lambda)
{
  K_ = K;

  freq_ = freq;
  amp_ = amp;
  phase_ = phase;

  lambda_ = lambda;
  initialized_ = true;
}

PhaseAFO::~PhaseAFO()
{
}

Eigen::Vector2d PhaseAFO::dydt(const Eigen::Vector2d& y, double t)
{
  if(!initialized_)
  {
    printf("did not initialize the object\n");
    return Eigen::Vector2d::Zero();
  }

  Eigen::Vector2d dydt;

  double perturbation = 0.0;
  for(int i=0; i<freq_.size(); ++i)
  {
    perturbation += amp_(i)*cos(freq_(i)*t + phase_(i));
  }

  dydt(0) = lambda_ * y(1) - K_ * sin(y(0)) * perturbation;
  dydt(1) = - K_ * sin(y(0)) * perturbation;

  return dydt;
}

void PhaseAFO::integrate(double t_init, double t_end,
                         const Eigen::Vector2d& init,
                         Eigen::VectorXd& t, Eigen::MatrixXd& y,
                         double dt, double save_dt)
{
  if(!initialized_)
  {
    printf("did not initialize the object\n");
    return;
  }
  int inner_loop = int(save_dt/dt);
  int length = int((t_end - t_init)/save_dt);

  y.resize(2,length);
  t.resize(length);
  y.col(0) = init;
  t(0) = t_init;

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
