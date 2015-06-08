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
  PhaseAFO();
  ~PhaseAFO();

  Eigen::Vector2d dydt(const Eigen::Vector2d& y, double t);

  void init(double K, double omegaF, double lambda);
  void init(double K, const Eigen::VectorXd& freq,
           const Eigen::VectorXd& amp, const Eigen::VectorXd& phase,
           double lambda);


  void integrate(double t_init, double t_end,
                            const Eigen::Vector2d& init,
                            Eigen::VectorXd& t, Eigen::MatrixXd& y,
                            double dt=0.001,
                            double save_dt=0.001);

private:
  double K_;
  double lambda_;

  bool initialized_;

  Eigen::VectorXd freq_, amp_, phase_;

};

#endif /* PHASEAFO_H_ */
