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

class PhaseAFO{
public:
    PhaseAFO();
    ~PhaseAFO(){};

    void initialize(double K, double lambda);

    InputPerturbation& input(){return input_;}
    
    const Eigen::Vector2d dydt(const Eigen::Vector2d& y, double t);
    
    const int num_states(){return num_states_;}
    
    const Eigen::VectorXd& t(){return t_;}
    const Eigen::MatrixXd& y(){return y_;}

    Eigen::VectorXd t_;
    Eigen::MatrixXd y_;

private:
    double K_;
    double lambda_;

    const int num_states_ = 2;

    InputPerturbation input_;
};
}
#endif /* PHASEAFO_H_ */
