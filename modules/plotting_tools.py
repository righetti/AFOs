from pylab import *
from scipy import *
import numpy

def plot_result(filename):
    
    with open(filename) as f:
        t = numpy.array([double(line.split()[0]) for line in f])
    with open(filename) as f:
        phi = numpy.array([double(line.split()[1]) for line in f])
    with open(filename) as f:
        omega = numpy.array([double(line.split()[2]) for line in f])
    
    if omega[-1] > 100 or omega[-1] < 0:
        print filename
        figure(1)
        subplot(211)
        plot(t, phi - 30 * t)
        subplot(212)
        plot(t, omega)
    
        figure(2)
        plot(mod(phi, 2*pi), omega)
    
    
def plot_all():
    for i in range(0,627):
        print i
        plot_result('../test_result' + str(i) + '.dat')
              
               
    show()

        
    