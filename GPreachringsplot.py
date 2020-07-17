import numpy as np
import matplotlib
import matplotlib.pyplot as plt


totgpiUmD1 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD2 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD3 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD4 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD5 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD = (totgpiUmD1 + totgpiUmD2 + totgpiUmD3 + totgpiUmD4 + totgpiUmD5)/5

totthrptfmD1 = np.genfromtxt(open("totthrptfmrdD1.csv", "r"), delimiter=",", dtype =float)
totthrptfmD2 = np.genfromtxt(open("totthrptfmrdD2.csv", "r"), delimiter=",", dtype =float)
totthrptfmD3 = np.genfromtxt(open("totthrptfmrdD3.csv", "r"), delimiter=",", dtype =float)
totthrptfmD4 = np.genfromtxt(open("totthrptfmrdD4.csv", "r"), delimiter=",", dtype =float)
totthrptfmD5 = np.genfromtxt(open("totthrptfmrdD5.csv", "r"), delimiter=",", dtype =float)
totthrptfmD = (totthrptfmD1 + totthrptfmD2 + totthrptfmD3 + totthrptfmD4 + totthrptfmD5)/5

totfmUmD1 = np.genfromtxt(open("totfmUmrdD1.csv", "r"), delimiter=",", dtype =float)
totfmUmD2 = np.genfromtxt(open("totfmUmrdD2.csv", "r"), delimiter=",", dtype =float)
totfmUmD3 = np.genfromtxt(open("totfmUmrdD3.csv", "r"), delimiter=",", dtype =float)
totfmUmD4 = np.genfromtxt(open("totfmUmrdD4.csv", "r"), delimiter=",", dtype =float)
totfmUmD5 = np.genfromtxt(open("totfmUmrdD5.csv", "r"), delimiter=",", dtype =float)
totfmUmD = (totfmUmD1 + totfmUmD2 + totfmUmD3 + totfmUmD4 + totfmUmD5)/5

totthrptgpD1 = np.genfromtxt(open("totthrptgprdD1.csv", "r"), delimiter=",", dtype =float)
totthrptgpD2 = np.genfromtxt(open("totthrptgprdD2.csv", "r"), delimiter=",", dtype =float)
totthrptgpD3 = np.genfromtxt(open("totthrptgprdD3.csv", "r"), delimiter=",", dtype =float)
totthrptgpD4 = np.genfromtxt(open("totthrptgprdD4.csv", "r"), delimiter=",", dtype =float)
totthrptgpD5 = np.genfromtxt(open("totthrptgprdD5.csv", "r"), delimiter=",", dtype =float)
totthrptgpD = (totthrptgpD1 + totthrptgpD2 + totthrptgpD3 + totthrptgpD4 + totthrptgpD5)/5

totUmgpD1 = np.genfromtxt(open("totUmgprdD1.csv", "r"), delimiter=",", dtype =float)
totUmgpD2 = np.genfromtxt(open("totUmgprdD2.csv", "r"), delimiter=",", dtype =float)
totUmgpD3 = np.genfromtxt(open("totUmgprdD3.csv", "r"), delimiter=",", dtype =float)
totUmgpD4 = np.genfromtxt(open("totUmgprdD4.csv", "r"), delimiter=",", dtype =float)
totUmgpD5 = np.genfromtxt(open("totUmgprdD5.csv", "r"), delimiter=",", dtype =float)
totUmgpD = (totUmgpD1 + totUmgpD2 + totUmgpD3 + totUmgpD4 + totUmgpD5)/5

totgpshcpD1 = np.genfromtxt(open("totgpshcprdD1.csv", "r"), delimiter=",", dtype =float)
totgpshcpD2 = np.genfromtxt(open("totgpshcprdD2.csv", "r"), delimiter=",", dtype =float)
totgpshcpD3 = np.genfromtxt(open("totgpshcprdD3.csv", "r"), delimiter=",", dtype =float)
totgpshcpD4 = np.genfromtxt(open("totgpshcprdD4.csv", "r"), delimiter=",", dtype =float)
totgpshcpD5 = np.genfromtxt(open("totgpshcprdD5.csv", "r"), delimiter=",", dtype =float)
totgpshcpD = (totgpshcpD1 + totgpshcpD2 + totgpshcpD3 + totgpshcpD4 + totgpshcpD5)/5

totthrptrtmD1 = np.genfromtxt(open("totthrptrtmrdD1.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD2 = np.genfromtxt(open("totthrptrtmrdD2.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD3 = np.genfromtxt(open("totthrptrtmrdD3.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD4 = np.genfromtxt(open("totthrptrtmrdD4.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD5 = np.genfromtxt(open("totthrptrtmrdD5.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD = (totthrptrtmD1 + totthrptrtmD2 + totthrptrtmD3 + totthrptrtmD4 + totthrptrtmD5)/5

totUmrtmD1 = np.genfromtxt(open("totUmrtmrdD1.csv", "r"), delimiter=",", dtype =float)
totUmrtmD2 = np.genfromtxt(open("totUmrtmrdD2.csv", "r"), delimiter=",", dtype =float)
totUmrtmD3 = np.genfromtxt(open("totUmrtmrdD3.csv", "r"), delimiter=",", dtype =float)
totUmrtmD4 = np.genfromtxt(open("totUmrtmrdD4.csv", "r"), delimiter=",", dtype =float)
totUmrtmD5 = np.genfromtxt(open("totUmrtmrdD5.csv", "r"), delimiter=",", dtype =float)
totUmrtmD = (totUmrtmD1 + totUmrtmD2 + totUmrtmD3 + totUmrtmD4 + totUmrtmD5)/5

totrtmshcpD1 = np.genfromtxt(open("totrtmshcprdD1.csv", "r"), delimiter=",", dtype =float)
totrtmshcpD2 = np.genfromtxt(open("totrtmshcprdD2.csv", "r"), delimiter=",", dtype =float)
totrtmshcpD3 = np.genfromtxt(open("totrtmshcprdD3.csv", "r"), delimiter=",", dtype =float)
totrtmshcpD4 = np.genfromtxt(open("totrtmshcprdD4.csv", "r"), delimiter=",", dtype =float)
totrtmshcpD5 = np.genfromtxt(open("totrtmshcprdD5.csv", "r"), delimiter=",", dtype =float)
totrtmshcpD = (totrtmshcpD1 + totrtmshcpD2 + totrtmshcpD3 + totrtmshcpD4 + totrtmshcpD5)/5

gpUmD1 = np.genfromtxt(open("gpUmrdD1.csv", "r"), delimiter=",", dtype =float)
gpUmD2 = np.genfromtxt(open("gpUmrdD2.csv", "r"), delimiter=",", dtype =float)
gpUmD3 = np.genfromtxt(open("gpUmrdD3.csv", "r"), delimiter=",", dtype =float)
gpUmD4 = np.genfromtxt(open("gpUmrdD4.csv", "r"), delimiter=",", dtype =float)
gpUmD5 = np.genfromtxt(open("gpUmrdD5.csv", "r"), delimiter=",", dtype =float)
gpUmD = (gpUmD1 + gpUmD2 + gpUmD3 + gpUmD4 + gpUmD5)/5

gpmfD1 = np.genfromtxt(open("gpmfrdD1.csv", "r"), delimiter=",", dtype =float)
gpmfD2 = np.genfromtxt(open("gpmfrdD2.csv", "r"), delimiter=",", dtype =float)
gpmfD3 = np.genfromtxt(open("gpmfrdD3.csv", "r"), delimiter=",", dtype =float)
gpmfD4 = np.genfromtxt(open("gpmfrdD4.csv", "r"), delimiter=",", dtype =float)
gpmfD5 = np.genfromtxt(open("gpmfrdD5.csv", "r"), delimiter=",", dtype =float)
gpmfD = (gpmfD1 + gpmfD2 + gpmfD3 + gpmfD4 + gpmfD5)/5

rtmUmD1 = np.genfromtxt(open("rtmUmrdD1.csv", "r"), delimiter=",", dtype =float)
rtmUmD2 = np.genfromtxt(open("rtmUmrdD2.csv", "r"), delimiter=",", dtype =float)
rtmUmD3 = np.genfromtxt(open("rtmUmrdD3.csv", "r"), delimiter=",", dtype =float)
rtmUmD4 = np.genfromtxt(open("rtmUmrdD4.csv", "r"), delimiter=",", dtype =float)
rtmUmD5 = np.genfromtxt(open("rtmUmrdD5.csv", "r"), delimiter=",", dtype =float)
rtmUmD = (rtmUmD1 + rtmUmD2 + rtmUmD3 + rtmUmD4 + rtmUmD5)/5

rtmmfD1 = np.genfromtxt(open("rtmmfrdD1.csv", "r"), delimiter=",", dtype =float)
rtmmfD2 = np.genfromtxt(open("rtmmfrdD2.csv", "r"), delimiter=",", dtype =float)
rtmmfD3 = np.genfromtxt(open("rtmmfrdD3.csv", "r"), delimiter=",", dtype =float)
rtmmfD4 = np.genfromtxt(open("rtmmfrdD4.csv", "r"), delimiter=",", dtype =float)
rtmmfD5 = np.genfromtxt(open("rtmmfrdD5.csv", "r"), delimiter=",", dtype =float)
rtmmfD = (rtmmfD1 + rtmmfD2 + rtmmfD3 + rtmmfD4 + rtmmfD5)/5


totgpiUmB1 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB2 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB3 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB4 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB5 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB = (totgpiUmB1 + totgpiUmB2 + totgpiUmB3 + totgpiUmB4 + totgpiUmB5)/5

totthrptfmB1 = np.genfromtxt(open("totthrptfmrdB1.csv", "r"), delimiter=",", dtype =float)
totthrptfmB2 = np.genfromtxt(open("totthrptfmrdB2.csv", "r"), delimiter=",", dtype =float)
totthrptfmB3 = np.genfromtxt(open("totthrptfmrdB3.csv", "r"), delimiter=",", dtype =float)
totthrptfmB4 = np.genfromtxt(open("totthrptfmrdB4.csv", "r"), delimiter=",", dtype =float)
totthrptfmB5 = np.genfromtxt(open("totthrptfmrdB5.csv", "r"), delimiter=",", dtype =float)
totthrptfmB = (totthrptfmB1 + totthrptfmB2 + totthrptfmB3 + totthrptfmB4 + totthrptfmB5)/5

totfmUmB1 = np.genfromtxt(open("totfmUmrdB1.csv", "r"), delimiter=",", dtype =float)
totfmUmB2 = np.genfromtxt(open("totfmUmrdB2.csv", "r"), delimiter=",", dtype =float)
totfmUmB3 = np.genfromtxt(open("totfmUmrdB3.csv", "r"), delimiter=",", dtype =float)
totfmUmB4 = np.genfromtxt(open("totfmUmrdB4.csv", "r"), delimiter=",", dtype =float)
totfmUmB5 = np.genfromtxt(open("totfmUmrdB5.csv", "r"), delimiter=",", dtype =float)
totfmUmB = (totfmUmB1 + totfmUmB2 + totfmUmB3 + totfmUmB4 + totfmUmB5)/5

totthrptgpB1 = np.genfromtxt(open("totthrptgprdB1.csv", "r"), delimiter=",", dtype =float)
totthrptgpB2 = np.genfromtxt(open("totthrptgprdB2.csv", "r"), delimiter=",", dtype =float)
totthrptgpB3 = np.genfromtxt(open("totthrptgprdB3.csv", "r"), delimiter=",", dtype =float)
totthrptgpB4 = np.genfromtxt(open("totthrptgprdB4.csv", "r"), delimiter=",", dtype =float)
totthrptgpB5 = np.genfromtxt(open("totthrptgprdB5.csv", "r"), delimiter=",", dtype =float)
totthrptgpB = (totthrptgpB1 + totthrptgpB2 + totthrptgpB3 + totthrptgpB4 + totthrptgpB5)/5

totUmgpB1 = np.genfromtxt(open("totUmgprdB1.csv", "r"), delimiter=",", dtype =float)
totUmgpB2 = np.genfromtxt(open("totUmgprdB2.csv", "r"), delimiter=",", dtype =float)
totUmgpB3 = np.genfromtxt(open("totUmgprdB3.csv", "r"), delimiter=",", dtype =float)
totUmgpB4 = np.genfromtxt(open("totUmgprdB4.csv", "r"), delimiter=",", dtype =float)
totUmgpB5 = np.genfromtxt(open("totUmgprdB5.csv", "r"), delimiter=",", dtype =float)
totUmgpB = (totUmgpB1 + totUmgpB2 + totUmgpB3 + totUmgpB4 + totUmgpB5)/5

totgpshcpB1 = np.genfromtxt(open("totgpshcprdB1.csv", "r"), delimiter=",", dtype =float)
totgpshcpB2 = np.genfromtxt(open("totgpshcprdB2.csv", "r"), delimiter=",", dtype =float)
totgpshcpB3 = np.genfromtxt(open("totgpshcprdB3.csv", "r"), delimiter=",", dtype =float)
totgpshcpB4 = np.genfromtxt(open("totgpshcprdB4.csv", "r"), delimiter=",", dtype =float)
totgpshcpB5 = np.genfromtxt(open("totgpshcprdB5.csv", "r"), delimiter=",", dtype =float)
totgpshcpB = (totgpshcpB1 + totgpshcpB2 + totgpshcpB3 + totgpshcpB4 + totgpshcpB5)/5

totthrptrtmB1 = np.genfromtxt(open("totthrptrtmrdB1.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB2 = np.genfromtxt(open("totthrptrtmrdB2.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB3 = np.genfromtxt(open("totthrptrtmrdB3.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB4 = np.genfromtxt(open("totthrptrtmrdB4.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB5 = np.genfromtxt(open("totthrptrtmrdB5.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB = (totthrptrtmB1 + totthrptrtmB2 + totthrptrtmB3 + totthrptrtmB4 + totthrptrtmB5)/5

totUmrtmB1 = np.genfromtxt(open("totUmrtmrdB1.csv", "r"), delimiter=",", dtype =float)
totUmrtmB2 = np.genfromtxt(open("totUmrtmrdB2.csv", "r"), delimiter=",", dtype =float)
totUmrtmB3 = np.genfromtxt(open("totUmrtmrdB3.csv", "r"), delimiter=",", dtype =float)
totUmrtmB4 = np.genfromtxt(open("totUmrtmrdB4.csv", "r"), delimiter=",", dtype =float)
totUmrtmB5 = np.genfromtxt(open("totUmrtmrdB5.csv", "r"), delimiter=",", dtype =float)
totUmrtmB = (totUmrtmB1 + totUmrtmB2 + totUmrtmB3 + totUmrtmB4 + totUmrtmB5)/5

totrtmshcpB1 = np.genfromtxt(open("totrtmshcprdB1.csv", "r"), delimiter=",", dtype =float)
totrtmshcpB2 = np.genfromtxt(open("totrtmshcprdB2.csv", "r"), delimiter=",", dtype =float)
totrtmshcpB3 = np.genfromtxt(open("totrtmshcprdB3.csv", "r"), delimiter=",", dtype =float)
totrtmshcpB4 = np.genfromtxt(open("totrtmshcprdB4.csv", "r"), delimiter=",", dtype =float)
totrtmshcpB5 = np.genfromtxt(open("totrtmshcprdB5.csv", "r"), delimiter=",", dtype =float)
totrtmshcpB = (totrtmshcpB1 + totrtmshcpB2 + totrtmshcpB3 + totrtmshcpB4 + totrtmshcpB5)/5

gpUmB1 = np.genfromtxt(open("gpUmrdB1.csv", "r"), delimiter=",", dtype =float)
gpUmB2 = np.genfromtxt(open("gpUmrdB2.csv", "r"), delimiter=",", dtype =float)
gpUmB3 = np.genfromtxt(open("gpUmrdB3.csv", "r"), delimiter=",", dtype =float)
gpUmB4 = np.genfromtxt(open("gpUmrdB4.csv", "r"), delimiter=",", dtype =float)
gpUmB5 = np.genfromtxt(open("gpUmrdB5.csv", "r"), delimiter=",", dtype =float)
gpUmB = (gpUmB1 + gpUmB2 + gpUmB3 + gpUmB4 + gpUmB5)/5

gpmfB1 = np.genfromtxt(open("gpmfrdB1.csv", "r"), delimiter=",", dtype =float)
gpmfB2 = np.genfromtxt(open("gpmfrdB2.csv", "r"), delimiter=",", dtype =float)
gpmfB3 = np.genfromtxt(open("gpmfrdB3.csv", "r"), delimiter=",", dtype =float)
gpmfB4 = np.genfromtxt(open("gpmfrdB4.csv", "r"), delimiter=",", dtype =float)
gpmfB5 = np.genfromtxt(open("gpmfrdB5.csv", "r"), delimiter=",", dtype =float)
gpmfB = (gpmfB1 + gpmfB2 + gpmfB3 + gpmfB4 + gpmfB5)/5

rtmUmB1 = np.genfromtxt(open("rtmUmrdB1.csv", "r"), delimiter=",", dtype =float)
rtmUmB2 = np.genfromtxt(open("rtmUmrdB2.csv", "r"), delimiter=",", dtype =float)
rtmUmB3 = np.genfromtxt(open("rtmUmrdB3.csv", "r"), delimiter=",", dtype =float)
rtmUmB4 = np.genfromtxt(open("rtmUmrdB4.csv", "r"), delimiter=",", dtype =float)
rtmUmB5 = np.genfromtxt(open("rtmUmrdB5.csv", "r"), delimiter=",", dtype =float)
rtmUmB = (rtmUmB1 + rtmUmB2 + rtmUmB3 + rtmUmB4 + rtmUmB5)/5

rtmmfB1 = np.genfromtxt(open("rtmmfrdB1.csv", "r"), delimiter=",", dtype =float)
rtmmfB2 = np.genfromtxt(open("rtmmfrdB2.csv", "r"), delimiter=",", dtype =float)
rtmmfB3 = np.genfromtxt(open("rtmmfrdB3.csv", "r"), delimiter=",", dtype =float)
rtmmfB4 = np.genfromtxt(open("rtmmfrdB4.csv", "r"), delimiter=",", dtype =float)
rtmmfB5 = np.genfromtxt(open("rtmmfrdB5.csv", "r"), delimiter=",", dtype =float)
rtmmfB = (rtmmfB1 + rtmmfB2 + rtmmfB3 + rtmmfB4 + rtmmfB5)/5



# %% DTAG plotting 

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#totthrptfmpl = totthrptfm*np.ones([numyears,1])

ln1 = ax1.plot(years, totthrptgpD,'--', color = 'b',label = 'GP')
#ln2 = ax1.plot(years, totthrptfmpl,'-', color = 'r',label = 'FM' )
ln2 = ax1.plot(years, totthrptfmD,'-', color = 'r',label = 'FM' )
ln3 = ax2.plot(years, totgpshcpD,'-.', color = 'g',label = 'Sh.')

ln4 = ax1.plot(years, totthrptrtmD,':', color = 'b',label = 'RTM')
#ln5 = ax2.plot(years, totrtmshcp,'-.', color = 'g',label = 'Sh. RTM')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
ax2.set_ylabel("Shannon limit (Tb/s)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptrdaveD.pdf', dpi=200,bbox_inches='tight')
plt.show()

totthrptdiffgpD = ((totthrptgpD - totthrptfmD)/totthrptfmD)*100
totthrptdiffrtmD = ((totthrptrtmD - totthrptfmD)/totthrptfmD)*100
totthrptdiffshD = ((totgpshcpD - totthrptfmD)/totthrptfmD)*100

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ln1 = ax1.plot(years, totthrptdiffgpD,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptdiffrtmD,':', color = 'r',label = 'RTM' )
ln3 = ax2.plot(years, totthrptdiffshD,'-', color = 'g',label = 'Sh.')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdaveD.pdf', dpi=200,bbox_inches='tight')
plt.show()
    

# %% BT plotting 

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#totthrptfmpl = totthrptfm*np.ones([numyears,1])

ln1 = ax1.plot(years, totthrptgpB,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptfmB,'-', color = 'r',label = 'FM' )
ln3 = ax2.plot(years, totgpshcpB,'-.', color = 'g',label = 'Sh.')

ln4 = ax1.plot(years, totthrptrtmB,':', color = 'b',label = 'RTM')
#ln5 = ax2.plot(years, totrtmshcp,'-.', color = 'g',label = 'Sh. RTM')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
ax2.set_ylabel("Shannon limit (Tb/s)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptrdaveB.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %%

totthrptdiffgpB = ((totthrptgpB - totthrptfmB)/totthrptfmB)*100
totthrptdiffrtmB = ((totthrptrtmB - totthrptfmB)/totthrptfmB)*100
totthrptdiffshB = ((totgpshcpB - totthrptfmB)/totthrptfmB)*100

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ln1 = ax1.plot(years, totthrptdiffgpB,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptdiffrtmB,':', color = 'r',label = 'RTM' )
ln3 = ax2.plot(years, totthrptdiffshB,'-', color = 'g',label = 'Sh.')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdaveB.pdf', dpi=200,bbox_inches='tight')
plt.show()
    





   




