/*
 * MeanFieldAFO.cpp
 *
 *  Created on: Aug 6, 2018
 *      Author: righetti
 */

#include "PoolAFO.h"


namespace afos
{

PoolAFO::PoolAFO()
{
  K_ = 0.;
  lambda_ = 1.;
  eta_ = 0.;
}

void PoolAFO::initialize(int num_oscill, double K, double lambda, double eta)
{
  K_ = K;
  lambda_ = lambda;
  eta_ = eta;

  num_oscill_ = num_oscill;
  num_states_ = 3 * num_oscill_;
}

const Eigen::VectorXd PoolAFO::dydt(const Eigen::VectorXd& y, double t)
{
  Eigen::VectorXd dydt;
  dydt.resize(num_states_);
  
  //the output of the oscills
  double out = input_.get(t) - y.tail(num_oscill_).dot(Eigen::cos(y.head(num_oscill_).array()).matrix());

  //the input
  auto tt = - K_ * Eigen::sin(y.head(num_oscill_).array()).matrix() * out;

  //the differential equation
  dydt.head(num_oscill_) = lambda_ * y.segment(num_oscill_, num_oscill_) + tt;
  dydt.segment(num_oscill_, num_oscill_) = tt;
  dydt.tail(num_oscill_) = eta_ * out * Eigen::cos(y.head(num_oscill_).array()).matrix();

  return dydt;
}


}