# %%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# constant loading 
gpmfD4 = np.genfromtxt(open("gpmfD_4.csv", "r"), delimiter=",", dtype =float)
gpUmD4 = np.genfromtxt(open("gpUmD_4.csv", "r"), delimiter=",", dtype =float)
gpshcpD4 = np.genfromtxt(open("gpshcpD_4.csv", "r"), delimiter=",", dtype =float)
gpmarD4 = np.genfromtxt(open("gpmarD_4.csv", "r"), delimiter=",", dtype =float)

gpmfD45 = np.genfromtxt(open("gpmfD_45.csv", "r"), delimiter=",", dtype =float)
gpUmD45 = np.genfromtxt(open("gpUmD_45.csv", "r"), delimiter=",", dtype =float)
gpshcpD45 = np.genfromtxt(open("gpshcpD_45.csv", "r"), delimiter=",", dtype =float)
gpmarD45 = np.genfromtxt(open("gpmarD_45.csv", "r"), delimiter=",", dtype =float)

gpmfD5 = np.genfromtxt(open("gpmfD_5.csv", "r"), delimiter=",", dtype =float)
gpUmD5 = np.genfromtxt(open("gpUmD_5.csv", "r"), delimiter=",", dtype =float)
gpshcpD5 = np.genfromtxt(open("gpshcpD_5.csv", "r"), delimiter=",", dtype =float)
gpmarD5 = np.genfromtxt(open("gpmarD_5.csv", "r"), delimiter=",", dtype =float)


# %%

numpthsD = 12*(12-1)*2
numpthsB = 13*(13-1)*2
numpthsT = 11*(11-1)*2
years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)
Rs = 32

def rateconv(modfor):
    if modfor == 2:
        rate = 50
    elif modfor == 4:
        rate = 100
    elif modfor == 8:
        rate = 150    
    elif modfor == 16:
        rate = 200
    elif modfor == 32:
        rate = 250
    elif modfor == 64:
        rate = 300
    elif modfor == 128:
        rate = 350
    return rate

def thrptcalc(gpmf, gpUm, gpshcp, gpmar,numpths):
    ratesgp = np.empty([numpths,numyears])
    FECOH = 0.2
    for i in range(numpths):
        for j in range(numyears):
            ratesgp[i][j] = rateconv(gpmf[i][j])
    totthrptgp = np.sum(ratesgp, axis=0)/1e3
    totgpshcp = np.sum(gpshcp,axis=0).reshape(numyears,1)*Rs*(1-FECOH)*2/1e3
    medUmgp = np.median(gpUm,axis=0).reshape(numyears,1)
    medmargp = np.median(gpmar, axis=0).reshape(numyears,1)
    gpDm = gpmar - gpUm
    medDmgp = np.median(gpDm, axis=0).reshape(numyears,1)
    return totthrptgp, medUmgp, totgpshcp, medmargp, medDmgp


totthrptgpD4, medUmgpD4, totgpshcpD4, medmargpD4, medDmgpD4 = thrptcalc(gpmfD4, gpUmD4, gpshcpD4, gpmarD4,numpthsD)
totthrptgpD45, medUmgpD45, totgpshcpD45, medmargpD45, medDmgpD45 = thrptcalc(gpmfD45, gpUmD45, gpshcpD45, gpmarD45,numpthsD)
totthrptgpD5, medUmgpD5, totgpshcpD5, medmargpD5, medDmgpD5 = thrptcalc(gpmfD5, gpUmD5, gpshcpD5, gpmarD5,numpthsD)

# %% 

D4area = sum([ (totthrptgpD4[i] + totthrptgpD4[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9
D45area = sum([ (totthrptgpD45[i] + totthrptgpD45[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9
D5area = sum([ (totthrptgpD5[i] + totthrptgpD5[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9




# %% total throughput DTAG, constant loading 

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

fig, ax1 = plt.subplots()
ln1 = ax1.plot(years, totthrptgpD4,'--', color = 'b',label = 'DTAG 4$\sigma$')
ln2 = ax1.plot(years, totthrptgpD45,'--', color = 'r',label = 'DTAG 4.5$\sigma$')
ln3 = ax1.plot(years, totthrptgpD5,'--', color = 'g',label = 'DTAG 5$\sigma$')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
plt.savefig('totthrptnlD.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% median applied D margin, DTAG, constant loading 

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)


fig, ax1 = plt.subplots()
ln1 = ax1.plot(years, medDmgpD4,'--', color = 'b',label = 'DTAG 4$\sigma$')
ln2 = ax1.plot(years, medDmgpD45,'--', color = 'r',label = 'DTAG 4.5$\sigma$')
ln3 = ax1.plot(years, medDmgpD5,'--', color = 'g',label = 'DTAG 5$\sigma$')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("median applied D margin (dB)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
plt.savefig('medDmnlD.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% median total margin DTAG, constant loading 

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)


fig, ax1 = plt.subplots()
ln1 = ax1.plot(years, medmargpD4,'--', color = 'b',label = 'DTAG 4$\sigma$')
ln2 = ax1.plot(years, medmargpD45,'--', color = 'r',label = 'DTAG 4.5$\sigma$')
ln3 = ax1.plot(years, medmargpD5,'--', color = 'g',label = 'DTAG 5$\sigma$')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("median total margin (dB)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
plt.savefig('medtotalmarnlD.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% median U margin DTAG, constant loading 

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)


fig, ax1 = plt.subplots()
ln1 = ax1.plot(years, medUmgpD4,'--', color = 'b',label = 'DTAG 4$\sigma$')
ln2 = ax1.plot(years, medUmgpD45,'--', color = 'r',label = 'DTAG 4.5$\sigma$')
ln3 = ax1.plot(years, medUmgpD5,'--', color = 'g',label = 'DTAG 5$\sigma$')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("median U margin (dB)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
plt.savefig('medUmnlD.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %%
