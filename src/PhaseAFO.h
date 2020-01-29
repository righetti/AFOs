/*
 * PhaseAFO.h
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#ifndef PHASEAFO_H_
#define PHASEAFO_H_

#include <eigen3/Eigen/Eigen>

#include "InputPerturbation.h"

namespace afos
{

class PhaseAFO {
public:
    PhaseAFO();
    ~PhaseAFO(){};

    void initialize(double K, double lambda);


    void integrate(double t_init, double t_end,
                   const Eigen::Vector2d& init,
                   double dt=0.001,
                   double save_dt=0.001);
    const Eigen::VectorXd& t(){return t_;}
    const Eigen::MatrixXd& y(){return y_;}

    InputPerturbation& input(){return input_;}
    

private:
    Eigen::VectorXd dydt(const Eigen::Vector2d& y, double t);

    double K_;
    double lambda_;

    Eigen::MatrixXd y_;
    Eigen::VectorXd t_;

    InputPerturbation input_;

    bool initialized_;
};
}
#endif /* PHASEAFO_H_ */
