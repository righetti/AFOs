/*
 * euler.cpp
 *
 *  Created on: May 8, 2012
 *      Author: righetti
 */

#include "euler.h"

namespace afos
{

Euler::Euler()
{
  // TODO Auto-generated constructor stub

}

Euler::~Euler()
{
  // TODO Auto-generated destructor stub
}

template<class ODE, class Recorder>
size_t Euler::integrate(ODE& eqs, const std::vector<double>& x_init, double time_init,
                        double time_end, double dt, Recorder& rec)
{
  return 0;
}

template<class ODE, class Recorder>
void integrate_step(ODE& eqs, std::vector<double>& x0, double& t,
                      double dt, Recorder& rec)
{

}

} /* namespace inverse_dynamics */
