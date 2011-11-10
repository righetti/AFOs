/*
 * adaptivephaseoscillator.cpp
 *
 *  Created on: Nov 9, 2011
 *      Author: righetti
 */

#include "adaptive_phase_oscillator.h"

namespace afos
{

AdaptivePhaseOscillator::AdaptivePhaseOscillator(double omega_F, double K)
{
  K_ = K;
  omega_F_ = omega_F;
}

AdaptivePhaseOscillator::~AdaptivePhaseOscillator()
{
}

void AdaptivePhaseOscillator::operator ()(const std::vector<double>& x, std::vector<double>& dxdt, const double t)
{
  dxdt[0] = x[1] - sin(x[0]) * K_ * cos(omega_F_ * t);
  dxdt[1] = - sin(x[0]) * K_ * cos(omega_F_ * t);
}
} /* namespace afos */
