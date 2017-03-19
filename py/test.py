#!/usr/bin/env python
import numpy as np
from scipy.integrate import quad
import  matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.optimize import *

def I(m):
    return quad(lambda r:r*r*np.sqrt(r*r+m*m)/(np.exp(r)+1),0,800)[0]


C1=1.803085354 #3/2*zeta(3)
C2=23.33087449#45/2*zeta(5)
C3=714.6675501#2835/4*zeta(7)
mar=np.logspace(-2,3,1000)
tr=np.array([I(m) for m in mar])
lowa=[7/120.*np.pi**4 for m in mar]
lowa2=[7/120.*np.pi**4+1/24.*np.pi**2*m**2 for m in mar]

higha=[C1*m for m in mar]
higha2=[C1*m+C2/(2*m)-C3/(8*m**3) for m in mar]

##intermediate

ix=np.arange(-1.1,2.8,0.01)
mix=np.exp(ix)
iy=np.array([np.log(I(x)) for x in np.exp(ix)])
dix=ix*1.0
def aprx(ip):
    return (ip[0]+ip[1]*dix+ip[2]*dix**2.+ip[3]*dix**3/6.+ip[4]*dix**4/24.
                       +ip[5]*dix**5/120.+ip[6]*dix**6/720.+ip[7]*dix**7/5040. )

def testf(ip):
    return ((aprx(ip)-iy)**2).sum()

res=fmin_powell(testf,[  1.79932818 , 0.1085803 ,  0.0915494 ,  0.29250564 , 0.15887711, -0.77330889,
                         -0.11432634,  1.11719266])
print res


if True:
    plt.plot(mar,lowa/tr,label="radiation limit")
    plt.plot(mar,lowa2/tr,label="m=0 taylor")
    plt.plot(mar,higha/tr,label="CDM limit")
    plt.plot(mar,higha2/tr,label="m=infinit taylor")
    dix=np.log(mar)
    mid=np.exp(aprx(res))
    plt.plot(mar,mid/tr,label="mid fit")
    plt.plot(mar,np.ones(len(mar))+1e-3,'k--')
    plt.plot(mar,np.ones(len(mar))-1e-3,'k--')
    plt.ylim(0.99,1.01)
    plt.semilogx()
    plt.xlabel("$m_T$",fontsize=16)
    plt.ylabel("${\\rm approx}/{\\rm truth}$",fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./fitfig.pdf")
    plt.show()


