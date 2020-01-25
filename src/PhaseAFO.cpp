/*
 * PhaseAFO.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "PhaseAFO.h"

#include <cmath>
#include <cstdio>
#include <functional>

namespace afos
{

PhaseAFO::PhaseAFO()
{
  initialized_ = false;
}

void PhaseAFO::init_sine(double K, double omegaF, double lambda)
{
  K_ = K;
  freq_.resize(1);
  amp_.resize(1);
  phase_.resize(1);

  freq_(0) = omegaF;
  amp_(0) = 1.0;
  phase_(0) = 0.0;
  lambda_ = lambda;

  input_fun_ = [&](double t){
      return double(((freq_*t + phase_).array().cos()).matrix().dot(amp_));};

  initialized_ = true;
}

void PhaseAFO::initialize_vec_of_sines(double K, const Eigen::VectorXd& freq,
               const Eigen::VectorXd& amp, const Eigen::VectorXd& phase,
               double lambda)
{
  K_ = K;

  freq_ = freq;
  amp_ = amp;
  phase_ = phase;

  lambda_ = lambda;

  input_fun_ = [&](double t){
      return double(((freq_*t + phase_).array().cos()).matrix().dot(amp_));};

  initialized_ = true;
}

void PhaseAFO::init_frequency_changing_sine(double K, double omega_F, double omega_C, double lambda)
{
    K_ = K;
    lambda_ = lambda;
    double omega_C_inv = 1.0/omega_C;
    freq_.resize(1);
    freq_(0) = omega_F;
    input_fun_ = [=](double t){
        return sin(omega_C_inv * sin(omega_C*t) + omega_F*t);
    };

    initialized_ = true;
}


inline Eigen::Vector2d PhaseAFO::dydt(const Eigen::Vector2d& y, double t)
{
  Eigen::Vector2d dydt;
  double tt = - K_ * sin(y(0)) * input_fun_(t);

  dydt(0) = lambda_ * y(1) + tt;
  dydt(1) = tt;

  return dydt;
}

void PhaseAFO::integrate(double t_init, double t_end,
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

}
