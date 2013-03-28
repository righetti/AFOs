#include <cstdio>
#include <vector>
#include <cmath>

class Afo{

public:
  Afo(double K, double omegaF);
  ~Afo(){};

  void dydt(const std::vector<double>& y, const double& t, std::vector<double>& dydt);

private:
  double K_;
  double omegaF_;
};

Afo::Afo(double K, double omegaF)
{
  K_=K;
  omegaF_=omegaF;
}

void Afo::dydt(const std::vector<double>& y, const double& t, std::vector<double>& dydt)
{
  dydt[0] = y[1] - K_ * sin(y[0]) * cos(omegaF_*t-M_PI_2);
  dydt[1] = - K_ * sin(y[0]) * cos(omegaF_*t-M_PI_2);
}


int main(int argc, char **argv)
{
  double dt = 0.0000001;
  double save_dt = 0.001;
  int inner_loop = int(save_dt/dt);

  double t_init = 0.0;
  double t_end = 10.0;
  int length = int((t_end - t_init)/save_dt);

  std::vector<double> y_init;
  y_init.resize(2);
  y_init[0] = 0.0;
  y_init[1] = 10;

  std::vector<std::vector<double> > y;
  y.resize(length,y_init);

  Afo my_afo(10000000.0, 100);

  std::vector<double> dydt;
  dydt.resize(2,0.0);

  std::vector<double> t;
  t.resize(length,0.0);

  for(int i=1; i<length; ++i)
  {
    y_init = y[i-1];
    for(int j=0; j<inner_loop; ++j)
    {
      my_afo.dydt(y_init,t[i-1]+double(j)*dt,dydt);
      y_init[0]+= dydt[0]*dt;
      y_init[1]+= dydt[1]*dt;
    }
    y[i] = y_init;
    t[i] = t[i-1] + save_dt;
  }

  //save file
  FILE *save_file = fopen("result.txt","w");
  for(int i=0; i<length; ++i)
  {
    fprintf(save_file, "%f %f %f\n",t[i],y[i][0],y[i][1]);
  }
  fclose(save_file);

  return 0;
}
