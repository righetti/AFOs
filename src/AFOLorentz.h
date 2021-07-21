/*
 * Lorentz.h
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#ifndef AFOLORENTZ_H_
#define AFOLORENTZ_H_

#include <eigen3/Eigen/Eigen>


namespace afos
{

class AfoLorentz{
public:
    AfoLorentz();
    ~AfoLorentz(){};

    void initialize(double K, double lambda);

    const Eigen::VectorXd dydt(const Eigen::VectorXd& y, double t);
    
    const int num_states(){return num_states_;}
    
    const Eigen::VectorXd& t(){return t_;}
    const Eigen::MatrixXd& y(){return y_;}

    Eigen::VectorXd t_;
    Eigen::MatrixXd y_;

private:
    double sigma_, rho_, beta_;
    double K_, lambda_;

    const int num_states_ = 5;
};
}
#endif /* PHASEAFO_H_ */
