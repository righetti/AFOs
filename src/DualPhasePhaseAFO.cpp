/*
 * PhaseAFO.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "DualPhasePhaseAFO.h"

namespace afos
{

DualPhasePhaseAFO::DualPhasePhaseAFO()
{
  K_ = 0.0;
  lambda_ = 1.0;
  omegaF_ = 1.0;
}

void DualPhasePhaseAFO::initialize(double K, double lambda, double omegaF)
{
  K_ = K;
  lambda_ = lambda;
  omegaF_ = omegaF;
}

const Eigen::Vector3d DualPhasePhaseAFO::dydt(const Eigen::Vector3d& y, double t)
{
  Eigen::Vector3d dydt;
  double tt = - K_ * sin(y(1)) * input_.get(t);
  double tt2 = - K_ * sin(y(0)) * input_.get(t);

  dydt(0) = lambda_ * omegaF_ + tt2;
  dydt(1) = lambda_ * y(2) + tt;
  dydt(2) = tt;

  return dydt;
}

}
