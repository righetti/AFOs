#include <cstdio>
#include <vector>
#include <cmath>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "PhaseAFO.h"
#include "Integrator.h"

int main(int argc, char **argv)
{
  boost::posix_time::ptime run_start(boost::posix_time::microsec_clock::local_time());

  double dt = 0.0001;
  double save_dt = 0.001;

  double t_init = 0.0;
  double t_end = 10.0;

  Eigen::Vector2d y_init;
  y_init << 0.0, 10.0;

  afos::PhaseAFO my_afo;
  my_afo.initialize(1000.0, 1.);
  my_afo.input().sine(100.);
  euler_integration(my_afo, t_init, t_end,y_init,dt, save_dt);
  Eigen::VectorXd t = my_afo.t();
  Eigen::MatrixXd y = my_afo.y();

  //save file
  FILE *save_file = fopen("result.txt","w");
  for(int i=0; i<t.rows(); ++i)
    fprintf(save_file, "%f %f %f\n",t(i),y(0,i),y(1,i));
  fclose(save_file);

  boost::posix_time::ptime run_end(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration run_duration = run_end - run_start;
  double cycle_duration = run_duration.total_microseconds()/1000000.0;
  printf("time taken: %f\n",cycle_duration);
  return 0;
}
