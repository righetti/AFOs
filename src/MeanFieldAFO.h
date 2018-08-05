/*
 * MeanFieldAFO.h
 *
 *  Created on: Aug 6, 2018
 *      Author: righetti
 *
 *  This class implements the meanfield experiments in the siam paper
 *  the input is a time varying sine
 */

#pragma once

#include <eigen3/Eigen/Eigen>

class MeanFieldAFO{
public:
    MeanFieldAFO();
    ~MeanFieldAFO(){};

    void init(double K, double omegaF, double lambda);
    void init_vec(double K, const Eigen::VectorXd& freq,
                  const Eigen::VectorXd& amp, const Eigen::VectorXd& phase,
                  double lambda);


    void integrate(double t_init, double t_end,
                   const Eigen::Vector2d& init,
                   double dt=0.001,
                   double save_dt=0.001);
    const Eigen::VectorXd& t(){return t_;}
    const Eigen::MatrixXd& y(){return y_;}
private:
    Eigen::VectorXd dydt(const Eigen::Vector2d& y, double t);

    double K_;
    double lambda_;
    int num_oscill_;

    bool initialized_;

    Eigen::VectorXd freq_, amp_, phase_;
    Eigen::MatrixXd y_;
    Eigen::VectorXd t_;

};

#endif // MEANFIELDAFO_H
