# GPreachrings - GP reach assignment algorithm and ring network testbed 
# Uses ring topolpgies created by connecting the outer nodes of the BT (UK) and DTAG (GER) networks
# Throughput calculated for clockwise and anticlockwise connection for each node pair
# AUTHOR: Josh Nevin


# %% ################## imports ####################
import numpy as np
import matplotlib.pyplot as plt
import time 
from scipy import special
#from scipy.stats import iqr
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF
#from GHquad import GHquad
#from NFmodelGNPy import nf_model
from NFmodelGNPy import lin2db
from NFmodelGNPy import db2lin
#from GNf import GNmain
#import random
from dijkstra import dijkstra
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.preprocessing import StandardScaler
import cProfile
#from scipy.special import erfc

MSdata = True
if MSdata:
    def mssdcalc(f, clipind):
        clipind = int(clipind)
        numsamples = 32000 #32252
        msd = np.empty([numsamples,1])
        next(f)
        for i in range(numsamples):
            msdl = f.readline()
            msd[i] = msdl[20:clipind].strip()
        #msber = 0.5*erfc(msd/(2**0.5))
        #msvar = np.var(msber)
        msvar = np.var(msd)
        mssd = msvar**0.5
        return mssd

    def msdread(f, clipind):
        clipind = int(clipind)
        msd = []
        cdcheck = []
        next(f)
        stillparse = True
        i = 0
        while stillparse:
            msdl = f.readline()
            #print(msdl[37:44].strip())
            if len(msdl) != 0:
                cdcheck.append(float(msdl[38:44].strip(' 0\t')))
                #print(float(msdl[38:44].strip(' 0\t')))
                if i != 0:
                    if cdcheck[i] - cdcheck[i-1] > 5:
                        continue
                msd.append(float(msdl[20:clipind].strip()))
            else:
                stillparse = False
            i = i + 1
        numsamples = len(msd)
        sds = np.var(msd[0:672])**0.5
        sde = np.var(msd[numsamples-672:numsamples])**0.5
        return msd, sds, sde
    
    
    numchannels = 150
    numchannels2 = 190
    numchannels3 = 270
    numchannels4 = 240
    sg15sds = np.empty([numchannels,1])  # segments 1 - 5 
    sg711sds = np.empty([numchannels2,1]) # segments 7 - 11 
    sg5055sds = np.empty([numchannels3,1]) # segments 50 - 55 
    sg95100sds = np.empty([numchannels4,1]) # segments 95 - 100 
    sg15sde = np.empty([numchannels,1])  # segments 1 - 5 
    sg711sde = np.empty([numchannels2,1]) # segments 7 - 11 
    sg5055sde = np.empty([numchannels3,1]) # segments 50 - 55 
    sg95100sde = np.empty([numchannels4,1]) # segments 95 - 100 

    #testseg1,testseg1sds, testseg1sde = msdread(open("channel_1_segment_1.txt", "r"), 25)
    #testseg3, testseg3sds, testseg3sde = msdread(open("channel_65_segment_3.txt", "r"), 25)
    testseg7, _, _ = msdread(open("channel_" + str(191) + "_segment_7.txt", "r"), 25)
    testseg7, _, _ = msdreadfull(open("channel_" + str(191) + "_segment_7.txt", "r"), 25)

    """ for i in range(numchannels):
        if i < 40:
            _, sg15sds[i], sg15sde[i]  = msdread(open("channel_" + str(i+1) + "_segment_1.txt", "r"), 25)
        elif i < 60:
            _, sg15sds[i], sg15sde[i]  = msdread(open("channel_" + str(i+1) + "_segment_2.txt", "r"), 25)
        elif i < 100:
            _, sg15sds[i], sg15sde[i]  = msdread(open("channel_" + str(i+1) + "_segment_3.txt", "r"), 25)
        elif i < 130:
            _, sg15sds[i], sg15sde[i]  = msdread(open("channel_" + str(i+1) + "_segment_4.txt", "r"), 24) 
        else:
            _, sg15sds[i], sg15sde[i]  = msdread(open("channel_" + str(i+1) + "_segment_5.txt", "r"), 25)
    
    for i in range(numchannels2):
        if i < 240-numchannels2:
            #_, sg711sds[i], sg711sde[i]  = msdread(open("channel_" + str(i+191) + "_segment_7.txt", "r"), 25)
            continue
        elif i < 290-numchannels2:
            _, sg711sds[i], sg711sde[i]  = msdread(open("channel_" + str(i+191) + "_segment_8.txt", "r"), 25)
        elif i < 330-numchannels2:
            _, sg711sds[i], sg711sde[i]  = msdread(open("channel_" + str(i+191) + "_segment_9.txt", "r"), 25)
        elif i < 360-numchannels2:
            _, sg711sds[i], sg711sde[i]  = msdread(open("channel_" + str(i+191) + "_segment_10.txt", "r"), 25) 
        else:
            _, sg711sds[i], sg711sde[i]  = msdread(open("channel_" + str(i+191) + "_segment_11.txt", "r"), 25) """


    meansg15sds = np.mean(sg15sds)
    meansg15sde = np.mean(sg15sde)
    meansg711sds = np.mean(sg711sds)
    meansg711sde = np.mean(sg711sde)


    """ for i in range(numchannels):
        #print(i)
        if i < 40: 
            sg1sd[i] = mssdcalc(open("channel_" + str(i+1) + "_segment_1.txt", "r"), 25)

        elif i < 60:
            sg1sd[i] = mssdcalc(open("channel_" + str(i+1) + "_segment_2.txt", "r"), 25)
        elif i < 100:
            sg1sd[i] = mssdcalc(open("channel_" + str(i+1) + "_segment_3.txt", "r"), 25)
        elif i < 130:
            sg1sd[i] = mssdcalc(open("channel_" + str(i+1) + "_segment_4.txt", "r"), 24)  
        else:
            sg1sd[i] = mssdcalc(open("channel_" + str(i+1) + "_segment_5.txt", "r"), 25)

    for i in range(numchannels2):
        if i < 240-numchannels2:
            sg2sd[i] = mssdcalc(open("channel_" + str(i+191) + "_segment_7.txt", "r"), 25)
        elif i < 290-numchannels2:
            sg2sd[i] = mssdcalc(open("channel_" + str(i+191) + "_segment_8.txt", "r"), 25)
        elif i < 330-numchannels2:
            sg2sd[i] = mssdcalc(open("channel_" + str(i+191) + "_segment_9.txt", "r"), 25)
        elif i < 360-numchannels2:
            sg2sd[i] = mssdcalc(open("channel_" + str(i+191) + "_segment_10.txt", "r"), 25)
        else:
            sg2sd[i] = mssdcalc(open("channel_" + str(i+191) + "_segment_11.txt", "r"), 25)
            
    for i in range(numchannels3):
        if i < 1840-1790:
            sg3sd[i] = mssdcalc(open("channel_" + str(i+1791) + "_segment_50.txt", "r"), 25)
        elif i < 1890-1790:
            sg3sd[i] = mssdcalc(open("channel_" + str(i+1791) + "_segment_51.txt", "r"), 25)
        elif i < 1930-1790:
            sg3sd[i] = mssdcalc(open("channel_" + str(i+1791) + "_segment_52.txt", "r"), 24)
        elif i < 1980-1790:
            sg3sd[i] = mssdcalc(open("channel_" + str(i+1791) + "_segment_53.txt", "r"), 25)
        elif i < 2010-1790:
            sg3sd[i] = mssdcalc(open("channel_" + str(i+1791) + "_segment_54.txt", "r"), 25)
        else:
            sg3sd[i] = mssdcalc(open("channel_" + str(i+1791) + "_segment_55.txt", "r"), 25)

    for i in range(numchannels4):
        if i < 3310-3250:
            sg4sd[i] = mssdcalc(open("channel_" + str(i+3251) + "_segment_95.txt", "r"), 25)
        elif i < 3360-3250:
            sg4sd[i] = mssdcalc(open("channel_" + str(i+3251) + "_segment_96.txt", "r"), 25)
        elif i < 3410-3250:
            sg4sd[i] = mssdcalc(open("channel_" + str(i+3251) + "_segment_97.txt", "r"), 24)
        elif i < 3420-3250:
            sg4sd[i] = mssdcalc(open("channel_" + str(i+3251) + "_segment_98.txt", "r"), 25)
        elif i < 3450-3250:
            sg4sd[i] = mssdcalc(open("channel_" + str(i+3251) + "_segment_99.txt", "r"), 25)
        else:
            sg4sd[i] = mssdcalc(open("channel_" + str(i+3251) + "_segment_100.txt", "r"), 25)
        
    sg1sdmean = np.mean(sg1sd)
    sg2sdmean = np.mean(sg2sd)
    sg3sdmean = np.mean(sg3sd)
    sg4sdmean = np.mean(sg4sd)
    sgtotmean = (sg1sdmean + sg2sdmean + sg3sdmean + sg4sdmean)/4 """

# %% section 1: find the shortest and second shortest paths for each node pair and start to fill network 

nodesT = ['1','2','3','4','5','6','7','8','9','10','11']

graphT = {'1':{'2':240,'11':240},'2':{'1':240,'3':240},'3':{'2':240,'4':240},    
         '4':{'3':240,'5':240},'5':{'4':240,'6':240},'6':{'5':240,'7':240}, '7':{'6':240,'8':240},
         '8':{'7':240,'9':240}, '9':{'8':240,'10':240}, '10':{'9':240,'11':240}, '11':{'10':240,'1':240}
         }
edgesT = {'1':{'2':0,'11':1},'2':{'1':2,'3':3},'3':{'2':4,'4':5},    
         '4':{'3':6,'5':7},'5':{'4':8,'6':9},'6':{'5':10,'7':11}, '7':{'6':12,'8':13},
         '8':{'7':14,'9':15}, '9':{'8':16,'10':17}, '10':{'9':18,'11':19}, '11':{'10':20,'1':21}
         }
numnodesT = 11
numedgesT = 22
LspansT = 80

nodesB = ['1','2','3','4','5','6','7','8','9','10','11','12','13']

graphB = {'1':{'2':720,'13':80},'2':{'1':720,'3':160},'3':{'2':160,'4':240},    
         '4':{'3':240,'5':160},'5':{'4':160,'6':80},'6':{'5':80,'7':240}, '7':{'6':240,'8':80},
         '8':{'7':80,'9':400}, '9':{'8':400,'10':160}, '10':{'9':160,'11':80}, '11':{'10':80,'12':160},
         '12':{'11':160,'13':240}, '13':{'12':240,'1':80}
         }
edgesB = {'1':{'2':0,'13':1},'2':{'1':2,'3':3},'3':{'2':4,'4':5},    
         '4':{'3':6,'5':7},'5':{'4':8,'6':9},'6':{'5':10,'7':11}, '7':{'6':12,'8':13},
         '8':{'7':14,'9':15}, '9':{'8':16,'10':17}, '10':{'9':18,'11':19}, '11':{'10':20,'12':21},
         '12':{'11':22,'13':23}, '13':{'12':24,'1':25}
         }
numnodesB = 13
numedgesB = 26
LspansB = 80

nodesD = ['1','2','3','4','5','6','7','8','9','10','11','12']

graphD = {'1':{'2':400,'12':160},'2':{'1':400,'3':240},'3':{'2':240,'4':320},    
         '4':{'3':320,'5':240},'5':{'4':240,'6':160},'6':{'5':160,'7':80}, '7':{'6':80,'8':240},
         '8':{'7':240,'9':240}, '9':{'8':240,'10':80}, '10':{'9':80,'11':80}, '11':{'10':80,'12':320}, '12':{'11':320,'1':160}
         }
edgesD = {'1':{'2':0,'12':1},'2':{'1':2,'3':3},'3':{'2':4,'4':5},    
         '4':{'3':6,'5':7},'5':{'4':8,'6':9},'6':{'5':10,'7':11}, '7':{'6':12,'8':13},
         '8':{'7':14,'9':15}, '9':{'8':16,'10':17}, '10':{'9':18,'11':19}, '11':{'10':20,'12':21}, '12':{'11':22,'1':23}
         }
numnodesD = 12
numedgesD = 24
LspansD = 80

graphA = graphD
if graphA == graphT:
    numnodesA = numnodesT
    numedgesA = numedgesT
    nodesA = nodesT
    LspansA = LspansT
    edgesA = edgesT

if graphA == graphB:
    numnodesA = numnodesB
    numedgesA = numedgesB
    nodesA = nodesB
    LspansA = LspansB
    edgesA = edgesB
    
if graphA == graphD:
    numnodesA = numnodesD
    numedgesA = numedgesD
    nodesA = nodesD
    LspansA = LspansD
    edgesA = edgesD

def removekey(d, keysrc, keydes): # function for removing key from dict - used to remove blocked links 
    r = dict(d)                     # removes the link between nodes 'keysrc' and 'keydes'
    del r.get(keysrc)[keydes]
    return r    

def findroutes(nodes, secondpath):
    dis = []
    path = []
    numnodes = len(nodes)
    if graphA == graphT:
        for i in range(numnodes):
            for j in range(numnodes): 
                if i == j:
                    continue
                d, p = dijkstra({'1':{'2':240,'11':240},'2':{'1':240,'3':240},'3':{'2':240,'4':240},    
                                '4':{'3':240,'5':240},'5':{'4':240,'6':240},'6':{'5':240,'7':240}, '7':{'6':240,'8':240},
                                 '8':{'7':240,'9':240}, '9':{'8':240,'10':240}, '10':{'9':240,'11':240}, '11':{'10':240,'1':240}
                                 } , nodes[i], nodes[j])
                dis.append(d)
                path.append(p)
                if secondpath:
                    shgraph = removekey({'1':{'2':240,'11':240},'2':{'1':240,'3':240},'3':{'2':240,'4':240},    
                                         '4':{'3':240,'5':240},'5':{'4':240,'6':240},'6':{'5':240,'7':240}, '7':{'6':240,'8':240},
                                         '8':{'7':240,'9':240}, '9':{'8':240,'10':240}, '10':{'9':240,'11':240}, '11':{'10':240,'1':240}
                                         }, p[0],p[1])
                    d2, p2 = dijkstra(shgraph , nodes[i], nodes[j])
                    dis.append(d2)
                    path.append(p2)
    if graphA == graphB:
        for i in range(numnodes):
            for j in range(numnodes): 
                if i == j:
                    continue
                d, p = dijkstra({'1':{'2':720,'13':80},'2':{'1':720,'3':160},'3':{'2':160,'4':240},    
                                '4':{'3':240,'5':160},'5':{'4':160,'6':80},'6':{'5':80,'7':240}, '7':{'6':240,'8':80},
                                '8':{'7':80,'9':400}, '9':{'8':400,'10':160}, '10':{'9':160,'11':80}, '11':{'10':80,'12':160},
                                 '12':{'11':160,'13':240}, '13':{'12':240,'1':80}
                                 } , nodes[i], nodes[j])
                dis.append(d)
                path.append(p)
                if secondpath:
                    shgraph = removekey({'1':{'2':720,'13':80},'2':{'1':720,'3':160},'3':{'2':160,'4':240},    
                                         '4':{'3':240,'5':160},'5':{'4':160,'6':80},'6':{'5':80,'7':240}, '7':{'6':240,'8':80},
                                         '8':{'7':80,'9':400}, '9':{'8':400,'10':160}, '10':{'9':160,'11':80}, '11':{'10':80,'12':160},
                                         '12':{'11':160,'13':240}, '13':{'12':240,'1':80}
                                          }, p[0],p[1])
                    d2, p2 = dijkstra(shgraph , nodes[i], nodes[j])
                    dis.append(d2)
                    path.append(p2)
    if graphA == graphD:
        for i in range(numnodes):
            for j in range(numnodes): 
                if i == j:
                    continue
                d, p = dijkstra({'1':{'2':400,'12':160},'2':{'1':400,'3':240},'3':{'2':240,'4':320},    
                                 '4':{'3':320,'5':240},'5':{'4':240,'6':160},'6':{'5':160,'7':80}, '7':{'6':80,'8':240},
                                 '8':{'7':240,'9':240}, '9':{'8':240,'10':80}, '10':{'9':80,'11':80}, '11':{'10':80,'12':320}, '12':{'11':320,'1':160}
                                 } , nodes[i], nodes[j])
                dis.append(d)
                path.append(p)
                if secondpath:
                    shgraph = removekey({'1':{'2':400,'12':160},'2':{'1':400,'3':240},'3':{'2':240,'4':320},    
                                         '4':{'3':320,'5':240},'5':{'4':240,'6':160},'6':{'5':160,'7':80}, '7':{'6':80,'8':240},
                                         '8':{'7':240,'9':240}, '9':{'8':240,'10':80}, '10':{'9':80,'11':80}, '11':{'10':80,'12':320}, '12':{'11':320,'1':160}
                                         }, p[0],p[1])
                    d2, p2 = dijkstra(shgraph , nodes[i], nodes[j])
                    dis.append(d2)
                    path.append(p2)
    return dis, path
        
pthdists, pths = findroutes(nodesA,True)             
numpths = len(pthdists)

#  section 2: find the number of wavelengths on each link in the topology 
     
def getlinklen(shpath,graph,edges):  # takes nodes traversed as input and returns the lengths of each link and the edge indices 
    linklen = np.empty([len(shpath)-1,1])
    link = []
    for i in range(len(shpath)-1):
        linklen[i] = float((graph.get(shpath[i])).get(shpath[i+1]))
        link.append((edges.get(shpath[i])).get(shpath[i+1]))
    return linklen, link                

edgeinds = [] # indices of each edge traversed for each path 
edgelens = [] # lengths of each edge traversed for each path 
numlamlk = np.zeros([numedgesA,1])
for i in range(len(pths)):
    edgeinds.append(getlinklen(pths[i], graphA, edgesA)[1])  # transparent network: only need total distance for each path 
    numlamlk[edgeinds[i]] = numlamlk[edgeinds[i]] + 1
    edgelens.append(getlinklen(pths[i], graphA, edgesA)[0])  # transparent network: only need total distance for each path 


test, _ = getlinklen(pths[4], graphA, edgesA)

#  section 3: ageing effects and margin calculation 

PchdBm = np.linspace(-6,6,500)  # 500 datapoints for higher resolution of Pch
TRxb2b = 26 # fron UCL paper: On the limits of digital back-propagation in the presence of transceiver noise, Lidia Galdino et al.
numpoints = 100

alpha = 0.2
NLco = 1.27
Disp = 16.7

OSNRmeasBW = 12.478 # OSNR measurement BW [GHz]
Rs = 32 # Symbol rate [Gbd]
Bchrs = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
testlen = 1000.0     # all ageing effects modelled using values in: Faster return of investment in WDM networks when elastic transponders dynamically fit ageing of link margins, Pesic et al.
years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)
sd = np.linspace(0.04, 0.08, np.size(years)) # added SNR uncertainty SD - assumed to double over lifetime

NF = np.linspace(4.5,5.5,np.size(years)) # define the NF ageing of the amplifiers 
alpha = 0.2 + 0.00164*years # define the fibre ageing due to splice losses over time 
trxaging = (1 + 0.05*years).reshape(np.size(years),1)*(OSNRmeasBW/Bchrs) # define TRx ageing 
oxcaging = (0.03 + 0.007*years).reshape(np.size(years),1)*(OSNRmeasBW/Bchrs) # define filter ageing for one filter

# find the worst-case margin required                         
fmD = sd[-1]*5 # static D margin is defined as 5xEoL SNR uncertainty SD that is added
fmDGI = sd[0]*5 # inital D margin used for GP approach (before LP establishment)

#  section 4: find the SNR over each path, accounting for the varying number of wavelengths on each link 

def SNRgen(pathind, yearind, nyqch):  # function for generating a new SNR value to test if uncertainty is dealt with
        Ls = LspansA
        D = Disp          # rather than the worst-case number 
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        allin = np.log((10**(alpha[yearind]/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        links = edgeinds[pathind] # links traversed by path
        numlinks = len(links) # number of links traversed 
        Gnlisp = np.empty([numlinks,1])
        for i in range(numlinks):
            numlam = numlamlk[links[i]][0] # select number wavelengths on given link
            #print(numlam)
            if nyqch:
                NchNy = numlam
                BWNy = (NchNy*Rs)/1e3 
            else:
                NchRS = numlam
                Df = 50 # 50 GHz grid 
                BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
           # ===================== find Popt for one span ==========================
            numpch = len(PchdBm)
            Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
            if nyqch:
                Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
            else:
                Gwdmsw = Pchsw/(BchRS*1e9)
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
            G = alpha[yearind]*Ls
            NFl = 10**(NF[yearind]/10) 
            Gl = 10**(G/10) 
            if nyqch:
                Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
            else:
                Pasesw = NFl*h*f*(Gl - 1)*BchRS*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*BchRS*1e9)
            Popt = PchdBm[np.argmax(snrsw)]  
        # =======================================================================
            totnumspans = int(pthdists[pathind]/Ls) # total number of spans traversed for the path 
            numspans = int(edgelens[pathind][i][0]/Ls) # number of spans traversed for each link in the path
            if nyqch:
                Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisp[i] = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
            else:
                Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
                Gnlisp[i] = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans                                                                             
            
        Gnli = np.sum(Gnlisp)
        if nyqch:
            Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*Rs*1e9*totnumspans
        else:
            Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*BchRS*1e9*totnumspans
        Pch = 1e-3*10**(Popt/10) 
        if nyqch:
            snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + ((numlinks - 1)*2 + 2)*oxcaging[yearind])
        else:
            snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxaging[yearind] + ((numlinks - 1)*2 + 2)*oxcaging[yearind])
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]
        return lin2db(snr) + np.random.normal(0,sdnorm,numpoints) 


def fmsnr(pathind, yearind, nyqch):  # function for generating a new SNR value to test if uncertainty is dealt with
        Ls = LspansA
        D = Disp          # rather than the worst-case number 
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        allin = np.log((10**(alpha[yearind]/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        links = edgeinds[pathind] # links traversed by path
        numlinks = len(links) # number of links traversed 
        Gnlisp = np.empty([numlinks,1])
        for i in range(numlinks):
            numlam = numlamlk[links[i]][0] # select number wavelengths on given link
            #print(numlam)
            if nyqch:
                NchNy = numlam
                BWNy = (NchNy*Rs)/1e3 
            else:
                NchRS = numlam
                Df = 50 # 50 GHz grid 
                BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
           # ===================== find Popt for one span ==========================
            numpch = len(PchdBm)
            Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
            if nyqch:
                Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
            else:
                Gwdmsw = Pchsw/(BchRS*1e9)
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
            G = alpha[yearind]*Ls
            NFl = 10**(NF[yearind]/10) 
            Gl = 10**(G/10) 
            if nyqch:
                Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
            else:
                Pasesw = NFl*h*f*(Gl - 1)*BchRS*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*BchRS*1e9)
            Popt = PchdBm[np.argmax(snrsw)]  
        # =======================================================================
            totnumspans = int(pthdists[pathind]/Ls) # total number of spans traversed for the path 
            numspans = int(edgelens[pathind][i][0]/Ls) # number of spans traversed for each link in the path
            if nyqch:
                Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisp[i] = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
            else:
                Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
                Gnlisp[i] = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans                                                                             
            
        Gnli = np.sum(Gnlisp)
        if nyqch:
            Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*Rs*1e9*totnumspans
        else:
            Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*BchRS*1e9*totnumspans
        Pch = 1e-3*10**(Popt/10) 
        if nyqch:
            snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + ((numlinks - 1)*2 + 2)*oxcaging[yearind])
        else:
            snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxaging[yearind] + ((numlinks - 1)*2 + 2)*oxcaging[yearind])
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        return lin2db(snr)  

def SNRnew(pathind, yearind, nyqch):  # function for generating a new SNR value to test if uncertainty is dealt with
        Ls = LspansA
        D = Disp          # rather than the worst-case number 
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        allin = np.log((10**(alpha[yearind]/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        links = edgeinds[pathind] # links traversed by path
        numlinks = len(links) # number of links traversed 
        Gnlisp = np.empty([numlinks,1])
        for i in range(numlinks):
            numlam = numlamlk[links[i]][0] # select number wavelengths on given link
            #print(numlam)
            if nyqch:
                NchNy = numlam
                BWNy = (NchNy*Rs)/1e3 
            else:
                NchRS = numlam
                Df = 50 # 50 GHz grid 
                BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
           # ===================== find Popt for one span ==========================
            numpch = len(PchdBm)
            Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
            if nyqch:
                Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
            else:
                Gwdmsw = Pchsw/(BchRS*1e9)
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
            G = alpha[yearind]*Ls
            NFl = 10**(NF[yearind]/10) 
            Gl = 10**(G/10) 
            if nyqch:
                Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
            else:
                Pasesw = NFl*h*f*(Gl - 1)*BchRS*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*BchRS*1e9)
            Popt = PchdBm[np.argmax(snrsw)]  
        # =======================================================================
            totnumspans = int(pthdists[pathind]/Ls) # total number of spans traversed for the path 
            numspans = int(edgelens[pathind][i][0]/Ls) # number of spans traversed for each link in the path
            if nyqch:
                Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisp[i] = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
            else:
                Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
                Gnlisp[i] = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans                                                                             
            
        Gnli = np.sum(Gnlisp)
        if nyqch:
            Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*Rs*1e9*totnumspans
        else:
            Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*BchRS*1e9*totnumspans
        Pch = 1e-3*10**(Popt/10) 
        if nyqch:
            snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + ((numlinks - 1)*2 + 2)*oxcaging[yearind])
        else:
            snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxaging[yearind] + ((numlinks - 1)*2 + 2)*oxcaging[yearind])
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]
        return lin2db(snr) + np.random.normal(0,sdnorm,1) 


testsnrgen = SNRgen(4, 0, False)
testfmsnr = fmsnr(4, 0, False)
testsnrnew = SNRnew(4, 0, False)

#  section 5: implement fixed margin reach determination and intial GP reach (planning stage - before switch-on)

# define the FEC thresholds - all correspond to BER of 2e-2 (2% FEC) - given by MATLAB bertool 
FT2 = 3.243 
FT4 = 6.254 
FT8 = 10.697
FT16 = 12.707
FT32 = 16.579
FT64 = 18.432
FT128 = 22.185

def GPtrain(x,y):
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
    kernel = C(0.03, (1e-3, 1e1)) * RBF(0.01, (1e-2, 1e2)) 
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 20, normalize_y=False, alpha=np.var(y))
    gpr.fit(x, y)
    #print("Optimised kernel: %s" % gpr.kernel_)
    ystar, sigma = gpr.predict(x, return_std=True )
    sigma = np.reshape(sigma,(np.size(sigma), 1)) 
    sigma = (sigma**2 + 1)**0.5  
    ystarp = ystar + sigma
    ystari = scaler.inverse_transform(ystar)
    ystarpi = scaler.inverse_transform(ystarp)
    sigmai = np.mean(ystarpi - ystari)
    return ystari, sigmai


def initreach(pathind):
    
    gnSNRF = fmsnr(pathind, -1, False)
    gnSNRG = fmsnr(pathind, 0, False)
    # fixed margin case
    if gnSNRF - fmD > FT128:
        MF = 128
        UmF = gnSNRF - fmD - FT128
    elif gnSNRF - fmD > FT64:
        MF = 64
        UmF = gnSNRF - fmD - FT64
    elif gnSNRF - fmD > FT32:
        MF = 32
        UmF = gnSNRF - fmD - FT32
    elif gnSNRF - fmD > FT16:
        MF = 16
        UmF = gnSNRF - fmD - FT16
    elif gnSNRF - fmD > FT8:
        MF = 8
        UmF = gnSNRF - fmD - FT8
    elif gnSNRF - fmD > FT4:
        MF = 4
        UmF = gnSNRF - fmD - FT4
    elif gnSNRF - fmD > FT2:
        MF = 2
        UmF = gnSNRF - fmD - FT2
    else:
        print("not able to establish a link")

    # GP case   
    if gnSNRG - fmDGI > FT128:
        MFG = 128
        UmG = gnSNRG - fmDGI - FT128
    elif gnSNRG - fmDGI > FT64:
        MFG = 64
        UmG = gnSNRG - fmDGI - FT64
    elif gnSNRG - fmDGI > FT32:
        MFG = 32
        UmG = gnSNRG - fmDGI - FT32
    elif gnSNRG - fmDGI > FT16:
        MFG = 16
        UmG = gnSNRG - fmDGI - FT16
    elif gnSNRG - fmDGI > FT8:
        MFG = 8
        UmG = gnSNRG - fmDGI - FT8
    elif gnSNRG - fmDGI > FT4:
        MFG = 4
        UmG = gnSNRG - fmDGI - FT4
    elif gnSNRG - fmDGI > FT2:
        MFG = 2
        UmG = gnSNRG - fmDGI - FT2
    else:
        print("not able to establish a link")
    return MF, MFG, UmF, UmG

fmmf = np.empty([numpths,1]) # fixed margin modulation format 
gpimf = np.empty([numpths,1]) # GP initial modulation format
fmUm = np.empty([numpths,1]) # fixed margin U margins 
gpiUm = np.empty([numpths,1]) # GP initial U margins 
for i in range(numpths):
    fmmf[i], gpimf[i], fmUm[i], gpiUm[i] = initreach(i)
    
# %% section 6: implement GP reach algorithm 

def GPreach(pathind, numsig, yearind, nyqch):
    truesnr = SNRgen(pathind,yearind, nyqch)
    meansnr = np.mean(truesnr)
    x = np.linspace(0,numpoints,numpoints)
    prmn, sigma = GPtrain(x,truesnr)
    prmn = np.mean(prmn)
    if (prmn - FT128)/sigma > numsig:
        marG = prmn - FT128
        FT = FT128
        mfG = 128
    elif (prmn - FT64)/sigma > numsig:
        marG = prmn - FT64
        FT = FT64
        mfG = 64
    elif (prmn - FT32)/sigma > numsig:
        marG = prmn - FT32
        FT = FT32
        mfG = 32
    elif (prmn - FT16)/sigma > numsig:
        marG = prmn - FT16
        FT = FT16
        mfG = 16
    elif (prmn - FT8)/sigma > numsig:
        marG = prmn - FT8
        FT = FT8
        mfG = 8
    elif (prmn - FT4)/sigma > numsig:
        marG = prmn - FT4
        FT = FT4
        mfG = 4
    elif (prmn - FT2)/sigma > numsig:
        marG = prmn - FT2
        FT = FT2
        mfG = 2
    else:
        print("not able to establish a link")
# =============================================================================
#     if nyqch:
#         C = 32*np.log2(1 + db2lin(prmn))
#     else:
#         C = 41.6*np.log2(1 + db2lin(prmn))
# =============================================================================
    C = 2*np.log2(1+db2lin(meansnr))  # Shannon capacity of AWGN channel under and average power constraint in bits/symb
    if SNRnew(pathind,yearind, nyqch) > FT:
        estbl = 1
        Um = prmn - numsig*sigma - FT
    else:
        estbl = 0
    return mfG, estbl, Um, C, marG    
    

gpmf = np.empty([numpths,numyears])
gpestbl = np.empty([numpths,numyears])
gpUm = np.empty([numpths,numyears])
gpshcp = np.empty([numpths,numyears])
gpmar = np.empty([numpths,numyears])

numsig = 5
start = time.time()
for i in range(numpths):
    for j in range(numyears):  # change this later - will need to implement a time loop for the whole script, put it all in a big function
        gpmf[i][j], gpestbl[i][j], gpUm[i][j], gpshcp[i][j], gpmar[i][j] = GPreach(i, numsig, j, False)
    print("completed path " + str(i))
end = time.time()

print("GP algorithm took " + str((end-start)/60) + " minutes")

# %% section 7: determine throughput for the ring network 

if graphA == graphT:
    np.savetxt('gpmfT_' + str(numsig) + '.csv', gpmf, delimiter=',') 
    np.savetxt('gpUmT_' + str(numsig) + '.csv', gpUm, delimiter=',') 
    np.savetxt('gpshcpT_' + str(numsig) + '.csv', gpshcp, delimiter=',') 
    np.savetxt('gpmarT_' + str(numsig) + '.csv', gpmar, delimiter=',')
if graphA == graphD:
    np.savetxt('gpmfD_' + str(numsig) + '.csv', gpmf, delimiter=',') 
    np.savetxt('gpUmD_' + str(numsig) + '.csv', gpUm, delimiter=',') 
    np.savetxt('gpshcpD_' + str(numsig) + '.csv', gpshcp, delimiter=',') 
    np.savetxt('gpmarD_' + str(numsig) + '.csv', gpmar, delimiter=',')
if graphA == graphB:
    np.savetxt('gpmfB_' + str(numsig) + '.csv', gpmf, delimiter=',') 
    np.savetxt('gpUmB_' + str(numsig) + '.csv', gpUm, delimiter=',') 
    np.savetxt('gpshcpB_' + str(numsig) + '.csv', gpshcp, delimiter=',') 
    np.savetxt('gpmarB_' + str(numsig) + '.csv', gpmar, delimiter=',')

# %% import data
importdata = False
if importdata:
    if graphA == graphD:
        gpmf = np.genfromtxt(open("gpmfD.csv", "r"), delimiter=",", dtype =float)
        gpUm = np.genfromtxt(open("gpUmD.csv", "r"), delimiter=",", dtype =float)
        gpshcp = np.genfromtxt(open("gpshcpD.csv", "r"), delimiter=",", dtype =float)
        
    if graphA == graphB:
        gpmf = np.genfromtxt(open("gpmfB.csv", "r"), delimiter=",", dtype =float)
        gpUm = np.genfromtxt(open("gpUmB.csv", "r"), delimiter=",", dtype =float)
        gpshcp = np.genfromtxt(open("gpshcpB.csv", "r"), delimiter=",", dtype =float)
        

# %%

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

def thrptcalc(gpmf, gpUm, gpshcp, gpmar):
    ratesgp = np.empty([numpths,numyears])
    FECOH = 0.2
    for i in range(numpths):
        for j in range(numyears):
            ratesgp[i][j] = rateconv(gpmf[i][j])
    totthrptgp = np.sum(ratesgp, axis=0)/1e3
    totgpshcp = np.sum(gpshcp,axis=0).reshape(numyears,1)*Rs*(1-FECOH)*2/1e3
    totUmgp = np.sum(gpUm,axis=0).reshape(numyears,1)
    medmargp = np.sum(gpmar, axis=0).reshape(numyears,1)
    return totthrptgp, totUmgp, totgpshcp, medmargp

totthrptgp, totUmgp, totgpshcp, medmargp = thrptcalc(gpmf, gpUm, gpshcp, gpmar)


# %%

if graphA == graphD:
    np.savetxt('totthrptgpD_' + str(numsig) + '.csv', totthrptgp, delimiter=',') 
    np.savetxt('medmargpD_' + str(numsig) + '.csv', medmargp, delimiter=',') 
if graphA == graphB:
    np.savetxt('totthrptgpB.csv', totthrptgp, delimiter=',') 
    np.savetxt('medmargpB_' + str(numsig) + '.csv', medmargp, delimiter=',') 
if graphA == graphT:
    np.savetxt('totthrptgpT.csv', totthrptgp, delimiter=',') 
    np.savetxt('medmargpT_' + str(numsig) + '.csv', medmargp, delimiter=',') 



# %% plotting 

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)


fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totthrptgp,'--', color = 'b',label = 'GP')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('totalthrptnoloadingT.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %%

fig, ax1 = plt.subplots()

lpind = 0
ln1 = ax1.plot(years, gpmar[lpind],'--', color = 'b',label = 'GP')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("assigned margin (dB)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
    
lns = ln1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('margin.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %%






