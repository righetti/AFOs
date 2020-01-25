import numpy as np

def entrainment_basin_test(omegaF,amplitude,phase,omega,K,ind,savename):
  
  dt = 0.001
  T = 100
  t = np.arange(0., T+dt, dt)
 
  avg_freq = np.zeros(np.size(K),np.size(omega))
  conv_freq = np.zeros_like(avg_freq)
  max_error = np.zeros_like(avg_freq)
  mean_error = np.zeros_like(avg_freq)
  amp_error = np.zeros_like(avg_freq)
  
  if(ind~=1)
    load(savename2);
  end
  
  for i = ind:length(K)
    if(K(i) < 10)
      T = 200;
    elseif(K(i) < 50)
      T = 120;
    else
      T = 50;
    end
    
    for j = 1:length(omega)
      [K(i) omega(j)]
      init = [0 0 omega(j)];
      
      %[t,y] = ode45(@(t,y)(phase_ode(t,y,F,omega(j),K(i))),t,init,options);
      command = sprintf('unset LD_LIBRARY_PATH;./entrain_c %f %f %f %f ',T,dt,omega(j),K(i));
      system(command);
      res = load('/tmp/sim_result.dat');
      t = res(:,1);
      y = res(:,2:4);
      
      avg_freq(i,j) = mean(diff(y(round(length(t)/2):end,1))/dt);
      conv_freq(i,j) = mean(y(round(length(t)/2):end,3));
      
      amp_error(i,j) = max(y(round(length(t)/2):end,3)-conv_freq(i,j));
      
      f = find_closer_freq(conv_freq(i,j),omegaF);
      
      if(abs(f-conv_freq(i,j))>0.05*f)
        mean_error(i,j) = NaN;
        max_error(i,j) = NaN;
        'NaN'
      else
        y_avg = f + (omega(j)-f)*exp(-t);
        
        %[avg_freq(i,j) conv_freq(i,j)]
        mean_error(i,j) = mean(y(:,3)-y_avg);
        max_error(i,j) = max(abs(y(:,3)-y_avg));
      end
    end
    save(savename2,'avg_freq','conv_freq','K','omega','omegaF','phase','amplitude','mean_error','max_error','i','amp_error');
    command = sprintf('cp %s %s_%d.mat',savename2,savename,i);
    system(command);
  end
  
 
def res = find_closer_freq(avg_freq,omega):
  
  dist = abs(omega-avg_freq);
  res = omega(find(dist==min(dist),1));
  
def dydt = phase_ode(t,y,F,omega,K):
  
  %%%normal phase oscill
  dydt(1) = omega - K*F(t)*sin(y(1));
  
  %%%adapt phase oscill
  dydt(2) = y(3) - K*F(t)*sin(y(2));
  dydt(3) = - K*F(t)*sin(y(2));

  dydt = dydt';


omegaF = np.array([30,60,90])
amplitude = np.array([1.3,1,1.4])
phase = np.array([0.4,0,1.3])
 
# ss = [omegaF.T amplitude.T phase.T];

# save 'pert_params.dat' ss -ascii;

K = np.logspace(1,3,200)
omega = [20:1:50 50.5:0.5:69.5 70:1:100]
# %omega = 10:0.5:90;
  

entrain_basin_test(omegaF,amplitude,phase,omega,K,1,'result2sin');

