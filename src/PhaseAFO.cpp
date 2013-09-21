/*
 * PhaseAFO.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "PhaseAFO.h"

#include <cmath>

PhaseAFO::PhaseAFO(double K, double omegaF, double lambda): K_(K), omegaF_(omegaF), lambda_(lambda)
{
}

PhaseAFO::~PhaseAFO()
{
}

Eigen::Vector2d PhaseAFO::dydt(const Eigen::Vector2d& y, double t)
{
  Eigen::Vector2d dydt;
  dydt(0) = lambda_ * y(1) - K_ * sin(y(0)) * cos(omegaF_*t);
  dydt(1) = - K_ * sin(y(0)) * cos(omegaF_*t-M_PI_2);

  return dydt;
}
