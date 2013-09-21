#include <cstdio>
#include <vector>
#include <cmath>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "PhaseAFO.h"

int main(int argc, char **argv)
{
  boost::posix_time::ptime run_start(boost::posix_time::microsec_clock::local_time());

  double dt = 0.0000001;
  double save_dt = 0.001;
  int inner_loop = int(save_dt/dt);

  double t_init = 0.0;
  double t_end = 10.0;
  int length = int((t_end - t_init)/save_dt);

  Eigen::Vector2d y_init;
  y_init << 0.0, 10.0;

  Eigen::Matrix<double, 2, Eigen::Dynamic> y(2,length);

  y.col(0) = y_init;

  PhaseAFO my_afo(10000000.0, 100);

  Eigen::Vector2d dydt;
  dydt.resize(2,0.0);

  Eigen::Matrix<double, 1, Eigen::Dynamic> t(1,length);

  for(int i=1; i<length; ++i)
  {
    Eigen::Vector2d y_tmp;
    y_tmp = y.col(i-1);
    for(int j=0; j<inner_loop; ++j)
      y_tmp += my_afo.dydt(y_init,t(0,i-1)+double(j)*dt) * dt;
    y.col(i) = y_tmp;
    t(0,i) = t(0,i-1) + save_dt;
  }

  //save file
  FILE *save_file = fopen("result.txt","w");
  for(int i=0; i<length; ++i)
    fprintf(save_file, "%f %f %f\n",t(0,i),y(0,i),y(0,i));
  fclose(save_file);

  boost::posix_time::ptime run_end(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration run_duration = run_end - run_start;
  double cycle_duration = run_duration.total_microseconds()/1000000.0;
  printf("time taken: %f\n",cycle_duration);
  return 0;
}
