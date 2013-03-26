function adaptivePhaseOscill

dt = 0.01;
t = 0:dt:4;

options = odeset('reltol',10e-6);

phi_init = 0;
omega_init = 10;

omegaF = 30;
K = 10^7;


[t,y_phase] = ode45(@(t,y)(phase_ode(t,y,K,omegaF)),t,[phi_init omega_init],options);


subplot(2,1,1);
plot(t, y_phase(:,1) - omegaF*t);


subplot(2,1,2);
plot(t, y_phase(:,2));



function dydt = phase_ode(t,y,K,omegaF)

dydt(1) = y(2) - K * sin(y(1)) * cos(omegaF*t);
dydt(2) =  - K * sin(y(1)) * cos(omegaF*t);

dydt = dydt';