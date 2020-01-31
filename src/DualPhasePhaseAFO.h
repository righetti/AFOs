/*
 * DualPhasePhaseAFO.h
 *
 *  Created on: Jan 29, 2020
 *      Author: righetti
 */

#ifndef DUALPHASEPHASEAFO_H
#define DUALPHASEPHASEAFO_H

#include <eigen3/Eigen/Eigen>

#include "InputPerturbation.h"
#include "Integrator.h"

namespace afos
{

class DualPhasePhaseAFO {
public:
    DualPhasePhaseAFO();
    ~DualPhasePhaseAFO(){};

    void initialize(double K, double lambda, double omegaF);

    InputPerturbation& input(){return input_;}
    
    const Eigen::Vector3d dydt(const Eigen::Vector3d& y, double t);
    
    const int num_states(){return num_states_;}
    
    const Eigen::VectorXd& t(){return t_;}
    const Eigen::MatrixXd& y(){return y_;}

    Eigen::VectorXd t_;
    Eigen::MatrixXd y_;

private:
    double K_;
    double lambda_;
    double omegaF_;

    const int num_states_ = 3;

    InputPerturbation input_;
};
}
#endif /* DUALPHASEPHASEAFO_H */
