/*
 * PhaseAFO.h
 *
 *  Created on: Sep 20, 2013
 *      Author: righetti
 */

#ifndef PHASEAFO_H_
#define PHASEAFO_H_

#include <functional>
#include <eigen3/Eigen/Eigen>

namespace afos
{

class PhaseAFO {
public:
    PhaseAFO();
    ~PhaseAFO(){};

    //!
    //! \brief init_sine
    //! sets a simple sine input with unitary amp and no delay
    void init_sine(double K, double omegaF, double lambda);

    //!
    //! \brief init_vec_of_sines
    //! sets a sum of sine inputs with amplitudes and delays
    void init_vec_of_sines(double K, const Eigen::VectorXd& freq,
                           const Eigen::VectorXd& amp, const Eigen::VectorXd& phase,
                           double lambda);
    //!
    //! \brief init_frequency_changing_sine
    //! set an input that is of the form sin(omegaF * t + 1\omegaC * sin(omegaC*t))
    //! so the frequency of the sine frequency is changing at a rate of omegaC
    void init_frequency_changing_sine(double K, double omega_F, double omega_C, double lambda);


    void integrate(double t_init, double t_end,
                   const Eigen::Vector2d& init,
                   double dt=0.001,
                   double save_dt=0.001);
    const Eigen::VectorXd& t(){return t_;}
    const Eigen::MatrixXd& y(){return y_;}
private:
    Eigen::Vector2d dydt(const Eigen::Vector2d& y, double t);

    double K_;
    double lambda_;

    bool initialized_;

    Eigen::VectorXd freq_, amp_, phase_;
    Eigen::MatrixXd y_;
    Eigen::VectorXd t_;

    std::function<double(double)> input_fun_;
};
}
#endif /* PHASEAFO_H_ */
