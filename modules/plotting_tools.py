from pylab import *
from scipy import *
import numpy


def plot_result(filename):
    
    #data = genfromtxt(filename, dtype = float)
#    t, phi, omega = loadtxt(filename, unpack = True)
    with open(filename) as f:
        t = numpy.array([float(line.split()[0]) for line in f])
#    with open(filename) as f:
        f.seek(0)
        phi = numpy.array([float(line.split()[1]) for line in f])
#    with open(filename) as f:
        f.seek(0)
        omega = numpy.array([double(line.split()[2]) for line in f])
    
#    t = data[:,0]
#    phi = data[:,1]
#    omega = data[:,2]
#    
    figure(1)
    subplot(211)
    plot(t, phi - 30 * t)
    subplot(212)
    plot(t, omega)
    
#    figure(2)
#    plot(mod(phi, 2*pi), omega)
    
def plot_all():
    
    for i in range(0,1, 1):
        print i
        plot_result('../test_result' + str(i) + '.dat')
    show()

        
    