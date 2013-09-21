/*
 * PhaseAFO.h
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#ifndef PHASEAFO_H_
#define PHASEAFO_H_

#include <eigen3/Eigen/Eigen>

class PhaseAFO {
public:
  PhaseAFO(double K, double omegaF, double lambda=1.0);
  virtual ~PhaseAFO();

  Eigen::Vector2d dydt(const Eigen::Vector2d& y, double t);

  double getK(){return K_;};
private:
  double K_;
  double omegaF_;
  double lambda_;
};

#endif /* PHASEAFO_H_ */
