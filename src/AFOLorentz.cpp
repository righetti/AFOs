/*
 * PhaseAFO.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "AFOLorentz.h"


namespace afos
{

AfoLorentz::AfoLorentz()
{
  sigma_ = 10.0;
  rho_ = 28.;
  beta_ = 8./3.;

  K_ = 0.0;
  lambda_ = 1.0;
}

void AfoLorentz::initialize(double K, double lambda)
{
  K_ = K;
  lambda_ = lambda;
}

const Eigen::VectorXd AfoLorentz::dydt(const Eigen::VectorXd& y, double t)
{
  Eigen::VectorXd dydt;
  dydt.resize(5);

  dydt(0) = lambda_ * y(1) - K_ * sin(y(0)) * (y(4) - 23.);
  dydt(1) = - K_ * sin(y(0)) * (y(4) - 23.);

  dydt(2) = sigma_ * (y(3) - y(2));
  dydt(3) = y(2) * (rho_ - y(4)) - y(3);
  dydt(4) = y(2) * y(3) - beta_ * y(4);

  return dydt;
}

}
