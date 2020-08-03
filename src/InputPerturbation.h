/*
 * DualPhasePhaseAFO.h
 *
 *  Created on: Jan 29, 2020
 *      Author: righetti
 */

#ifndef INPUTPERTURBATION_H_
#define INPUTPERTURBATION_H_

#include <functional>
#include <eigen3/Eigen/Eigen>

namespace afos
{

class InputPerturbation {
public:
    InputPerturbation();
    ~InputPerturbation(){};

    //!
    //! \brief sine
    //! sets a simple sine input with unitary amp and no phase shift
    void sine(double omegaF);

    //!
    //! \brief vec_of_sines
    //! sets a sum of sine inputs with amplitudes and phase shifts
    void vec_of_sines(const Eigen::VectorXd& freq,
                           const Eigen::VectorXd& amp, const Eigen::VectorXd& phase);
    //!
    //! \brief frequency_changing_sine
    //! set an input that is of the form sin(omegaF * t + 1\omegaC * sin(omegaC*t))
    //! so the frequency of the sine frequency is changing at a rate of omegaC
    void frequency_changing_sine(double omega_F, double omega_C);

    void chirps_and_exponentials();

    double get(double t){return input_(t);};

private:
    Eigen::VectorXd freq_, amp_, phase_;
    Eigen::MatrixXd y_;
    Eigen::VectorXd t_;

    std::function<double(double)> input_;
};
}
#endif /* INPUTPERTURBATION_H_ */
