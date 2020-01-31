/*
 * PhaseAFO.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "PhaseAFO.h"


namespace afos
{

PhaseAFO::PhaseAFO()
{
  K_ = 0.0;
  lambda_ = 1.0;
}

void PhaseAFO::initialize(double K, double lambda)
{
  K_ = K;
  lambda_ = lambda;
}

const Eigen::Vector2d PhaseAFO::dydt(const Eigen::Vector2d& y, double t)
{
  Eigen::Vector2d dydt;
  double tt = - K_ * sin(y(0)) * input_.get(t);

  dydt(0) = lambda_ * y(1) + tt;
  dydt(1) = tt;

  return dydt;
}

}
