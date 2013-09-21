/*
 * EulerIntegration.h
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#ifndef EULERINTEGRATION_H_
#define EULERINTEGRATION_H_

#include <eigen3/Eigen/Eigen>

class EulerIntegration
{
public:
  EulerIntegration();
  virtual ~EulerIntegration();

  void integrate(Eigen::VectorXd time, Eigen::MatrixXd);

private:
};

#endif /* EULERINTEGRATION_H_ */
