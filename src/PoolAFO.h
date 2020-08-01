/*
 * MeanFieldAFO.h
 *
 *  Created on: Aug 6, 2018
 *      Author: righetti
 *
 *  This class implements the meanfield experiments in the siam paper
 *  the input is a time varying sine
 */

#ifndef POOLAFO_H_
#define POOLAFO_H_

#include <eigen3/Eigen/Eigen>

#include "InputPerturbation.h"
#include "Integrator.h"


namespace afos
{

class PoolAFO{
public:
    PoolAFO();
    ~PoolAFO(){};

    void initialize(int num_oscill, double K, double lambda, double eta);

    InputPerturbation& input(){return input_;}

    const Eigen::VectorXd dydt(const Eigen::VectorXd& y, double t);

    const int num_states(){return num_states_;}

    const Eigen::VectorXd& t(){return t_;}
    const Eigen::MatrixXd& y(){return y_;}

    Eigen::VectorXd t_;
    Eigen::MatrixXd y_;

private:

    double K_;
    double lambda_;
    double eta_;
    int num_oscill_;
    int num_states_;

    InputPerturbation input_;
};

}

#endif // POOLAFO_H_
