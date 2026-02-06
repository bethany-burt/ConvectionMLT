# Python script to compute temperature-pressure profiles
# by Kevin Heng (October 2014)

from numpy import arange,zeros
from scipy import special
from matplotlib import pyplot as plt

# function to compute T-P profile
def tpprofile(m,m0,tint,tirr,kappa_S,kappa0,kappa_cia,beta_S0,beta_L0,el1,el3):
    albedo = (1.0-beta_S0)/(1.0+beta_S0)
    kappa_L = kappa0 + kappa_cia*m/m0
    beta_S = kappa_S*m/beta_S0
    coeff1 = 0.25*(tint**4)
    coeff2 = 0.125*(tirr**4)*(1.0-albedo)
    term1 = 1.0/el1 + m*( kappa0 + 0.5*kappa_cia*m/m0 )/el3/(beta_L0**2)
    term2 = 0.5/el1 + special.expn(2,beta_S)*( kappa_S/kappa_L/beta_S0 - kappa_cia*m*beta_S0/el3/kappa_S/m0/(beta_L0**2) )
    term3 = kappa0*beta_S0*( 1./3. - special.expn(4,beta_S) )/el3/kappa_S/(beta_L0**2)
    term4 = kappa_cia*(beta_S0**2)*( 0.5 - special.expn(3,beta_S) )/el3/m0/(kappa_S**2)/(beta_L0**2)
    result = ( coeff1*term1 + coeff2*(term2 + term3 + term4) )**0.25
    return result


# input parameters (default values)
g = 1e3         # surface gravity
tint = 150.0    # internal temperature (K)
tirr = 1200.0   # irradiation temperature (K)
kappa_S0 = 0.01 # shortwave opacity
kappa0 = 0.02   # infrared opacity (constant component)
kappa_cia = 0.0 # CIA opacity normalization
beta_S0 = 1.0   # shortwave scattering parameter
beta_L0 = 1.0   # longwave scattering parameter
el1 = 3.0/8.0   # first longwave Eddington coefficient
el3 = 1.0/3.0   # second longwave Eddington coefficient

# define pressure and column mass arrays
logp = arange(-3,2.01,0.01)
pressure = 10.0**logp  # pressure in bars
bar2cgs = 1e6          # convert bar to cgs units 
p0 = max(pressure)     # BOA pressure
m = pressure*bar2cgs/g # column mass
m0 = p0*bar2cgs/g      # BOA column mass

# Experiment 1: greenhouse effect and CIA
np = len(m)
tp0 = zeros(np)
tp1 = zeros(np)
tp2 = zeros(np)
tp3 = zeros(np)

# (set all Tint=0 except for fiducial)
for i in range(0,np):
    tp0[i] = tpprofile(m[i],m0,0.0,tirr,kappa_S0,kappa0,kappa_cia,beta_S0,beta_L0,el1,el3)
    tp1[i] = tpprofile(m[i],m0,0.0,tirr,kappa_S0,0.03,kappa_cia,beta_S0,beta_L0,el1,el3)
    tp2[i] = tpprofile(m[i],m0,0.0,tirr,kappa_S0,kappa0,1.0,beta_S0,beta_L0,el1,el3)
    tp3[i] = tpprofile(m[i],m0,tint,tirr,kappa_S0,kappa0,kappa_cia,beta_S0,beta_L0,el1,el3)

line1, =plt.plot(tp0, pressure, linewidth=4, color='k', linestyle='-')
line2, =plt.plot(tp1, pressure, linewidth=4, color='k', linestyle='--')
line3, =plt.plot(tp2, pressure, linewidth=4, color='k', linestyle=':')
plt.plot(tp3, pressure, linewidth=2, color='k', linestyle='-')
plt.yscale('log')
plt.xlim([800,1100])
plt.ylim([1e2,1e-3])
plt.xlabel('$T$ (K)', fontsize=18)   #x-label
plt.ylabel('$P$ (bar)', fontsize=18) #y=label
plt.legend([line1,line2,line3],['fiducial',r'$\kappa_0=0.03$ cm$^2$ g$^{-1}$'
                                ,r'$\kappa_{\rm CIA}=1$ cm$^2$ g$^{-1}$'],frameon=False,prop={'size':18})
plt.savefig('tp1.eps', format='eps') #save in EPS format
plt.close()

# Experiment #2: anti-greenhouse effect
tp4 = zeros(np)
tp5 = zeros(np)

for i in range(0,np):
    tp4[i] = tpprofile(m[i],m0,tint,tirr,0.02,kappa0,kappa_cia,beta_S0,beta_L0,el1,el3)
    tp5[i] = tpprofile(m[i],m0,tint,tirr,0.04,kappa0,kappa_cia,beta_S0,beta_L0,el1,el3)

line1, =plt.plot(tp3, pressure, linewidth=4, color='k', linestyle='-')
line2, =plt.plot(tp4, pressure, linewidth=4, color='k', linestyle='--')
line3, =plt.plot(tp5, pressure, linewidth=4, color='k', linestyle=':')
plt.yscale('log')
plt.xlim([800,1100])
plt.ylim([1e2,1e-3])
plt.xlabel('$T$ (K)', fontsize=18)   #x-label
plt.ylabel('$P$ (bar)', fontsize=18) #y=label
plt.legend([line1,line2,line3],['fiducial',r'$\kappa_{\rm S_0}=0.02$ cm$^2$ g$^{-1}$'
                                ,r'$\kappa_{\rm S_0}=0.04$ cm$^2$ g$^{-1}$'],frameon=False,prop={'size':18})
plt.savefig('tp2.eps', format='eps') #save in EPS format
plt.close()

# Experiment #3: scattering greenhouse and anti-greenhouse
tp6 = zeros(np)
tp7 = zeros(np)
tp8 = zeros(np)
tp9 = zeros(np)

for i in range(0,np):
    tp6[i] = tpprofile(m[i],m0,tint,tirr,kappa_S0,kappa0,kappa_cia,0.75,beta_L0,el1,el3)
    tp7[i] = tpprofile(m[i],m0,tint,tirr,kappa_S0,kappa0,kappa_cia,beta_S0,0.75,el1,el3)
    tp8[i] = tpprofile(m[i],m0,tint,tirr,kappa_S0,kappa0,kappa_cia,0.5,beta_L0,el1,el3)
    tp9[i] = tpprofile(m[i],m0,tint,tirr,kappa_S0,kappa0,kappa_cia,beta_S0,0.5,el1,el3)
    
line1, =plt.plot(tp3, pressure, linewidth=4, color='k', linestyle='-')
line2, =plt.plot(tp7, pressure, linewidth=4, color='r', linestyle=':')
line3, =plt.plot(tp9, pressure, linewidth=4, color='r', linestyle='--')
line4, =plt.plot(tp6, pressure, linewidth=4, color='b', linestyle=':')
line5, =plt.plot(tp8, pressure, linewidth=4, color='b', linestyle='--')
plt.yscale('log')
plt.xlim([700,1450])
plt.ylim([1e2,1e-3])
plt.xlabel('$T$ (K)', fontsize=18)   #x-label
plt.ylabel('$P$ (bar)', fontsize=18) #y=label
plt.legend([line1,line2,line3,line4,line5],['fiducial',r'$\beta_{\rm L_0}=0.75$',r'$\beta_{\rm L_0}=0.5$'
                                            ,r'$\beta_{\rm S_0}=0.75$',r'$\beta_{\rm S_0}=0.5$'],frameon=False,prop={'size':18})
plt.savefig('tp3.eps', format='eps') #save in EPS format
plt.close()

