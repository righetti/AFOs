/*
 * adaptivephaseoscillator.h
 *
 *  Created on: Nov 9, 2011
 *      Author: righetti
 */

#ifndef ADAPTIVEPHASEOSCILLATOR_H_
#define ADAPTIVEPHASEOSCILLATOR_H_

#include <vector>
#include <cmath>

namespace afos
{

class AdaptivePhaseOscillator
{
public:
  AdaptivePhaseOscillator(double omega_F = 0.0, double K = 0.0);
  virtual ~AdaptivePhaseOscillator();

  void operator() (const std::vector<double>& x , std::vector<double>& dxdt , const double t );


private:
  double K_;
  double omega_F_;
};

} /* namespace afos */
#endif /* ADAPTIVEPHASEOSCILLATOR_H_ */
