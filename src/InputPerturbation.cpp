/*
 * PhaseAFO.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#include "InputPerturbation.h"


namespace afos
{

InputPerturbation::InputPerturbation()
{
  //by default the perturbation is zero
  input_ = [&](double t){return 0.0;};
}

void InputPerturbation::sine(double omegaF)
{
  freq_.resize(1);
  amp_.resize(1);
  phase_.resize(1);

  freq_(0) = omegaF;
  amp_(0) = 1.0;
  phase_(0) = 0.0;

  input_ = [&](double t) { return double(((freq_*t + phase_).array().cos()).matrix().dot(amp_)); };
}

void InputPerturbation::vec_of_sines(const Eigen::VectorXd& freq,
               const Eigen::VectorXd& amp, const Eigen::VectorXd& phase)
{
  freq_ = freq;
  amp_ = amp;
  phase_ = phase;

  input_ = [&](double t) { return double(((freq_*t + phase_).array().sin()).matrix().dot(amp_)); };
}

void InputPerturbation::frequency_changing_sine(double omega_F, double omega_C)
{
    double omega_C_inv = 1.0/omega_C;
    freq_.resize(3);
    freq_(0) = omega_F;
    freq_(1) = omega_C;
    freq_(2) = omega_C_inv;
    input_ = [&](double t){ return sin(freq_(2) * sin(freq_(1)*t) + freq_(0)*t); };
}

void InputPerturbation::chirps_and_exponentials()
{
  input_ = [&](double t){return sin(200*t + 2*t*t) + sin(400*t - t*t*t/15) +
                         sin(300*t)*exp(-(t-5)*(t-5)/2.5) +
                         sin(400*t)*exp(-(t-30)*(t-30)/5.); };
}
}
