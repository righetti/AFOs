from pylab import *
from scipy import *

def plot_result(filename):
    
    with open('../test_result.dat') as f:
        t = [double(line.split()[0]) for line in f]
    with open('../test_result.dat') as f:
        phi = [double(line.split()[1]) for line in f]
    with open('../test_result.dat') as f:
        omega = [double(line.split()[2]) for line in f]
    
    print len(t)
    print len(phi)
    print len(omega)
    
    
    figure()
    subplot(211)
    plot(t, mod(phi, 2*pi))
    subplot(212)
    plot(t, omega)
    
    figure()
    plot(mod(phi, pi), omega)
    show()