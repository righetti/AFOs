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

  void init(double K, double omegaF, double lambda);
  void init1(double K, const Eigen::VectorXd& freq,
           const Eigen::VectorXd& amp, const Eigen::VectorXd& phase,
           double lambda);


  void integrate(double t_init, double t_end,
                            const Eigen::Vector2d& init,
                            double dt=0.001,
                            double save_dt=0.001);
  const Eigen::VectorXd& t(){return t_;};
  const Eigen::MatrixXd& y(){return y_;};
private:
  Eigen::Vector2d dydt(const Eigen::Vector2d& y, double t);

  double K_;
  double lambda_;

  bool initialized_;

  Eigen::VectorXd freq_, amp_, phase_;
  Eigen::MatrixXd y_;
  Eigen::VectorXd t_;

};

#endif /* PHASEAFO_H_ */
