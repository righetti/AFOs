/*
 * state_recorder.h
 *
 *  Created on: Nov 9, 2011
 *      Author: righetti
 */

#ifndef STATE_RECORDER_H_
#define STATE_RECORDER_H_

#include <vector>

namespace afos
{

template <class state_type> class StateRecorder
{
public:
  StateRecorder(std::vector<state_type>& states, std::vector<double>& time) :
    states_(states), times_(time) {};

  virtual ~StateRecorder(){};

  void operator()(const state_type& x , double t)
  {
    states_.push_back(x);
    times_.push_back(t);
  };

private:
  std::vector<state_type>& states_;
  std::vector<double>& times_;
};

} /* namespace afos */
#endif /* STATE_RECORDER_H_ */
