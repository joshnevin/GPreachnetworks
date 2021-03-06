# GPreachringsrand - GP reach assignment algorithm and ring network testbed with network traffic loading
# Uses ring topolpgies created by connecting the outer nodes of the BT (UK) and DTAG (GER) networks
# Throughput calculated for clockwise and anticlockwise connection for randomly selected node pairs
# with the number of node pairs connected increasing by 20% per year 
# AUTHOR: Josh Nevin


# %% ################## imports ####################
import numpy as np
import matplotlib.pyplot as plt
import time 
import random
#from scipy import special
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
#import cProfile
#from scipy.special import erfc

# section 1: define topologies 

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

graphA = graphD # select the topology under test 
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

#  section 2: ageing effects and margin calculation 

PchdBm = np.linspace(-6,6,500)  # 500 datapoints for higher resolution of Pch
TRxb2b = 26 # fron UCL paper: On the limits of digital back-propagation in the presence of transceiver noise, Lidia Galdino et al.
numpoints = 100


NLco = 1.27
Disp = 16.7

OSNRmeasBW = 12.478 # OSNR measurement BW [GHz]
Rs = 32 # Symbol rate [Gbd]
Bchrs = 41.6
testlen = 1000.0     # all ageing effects modelled using values in: Faster return of investment in WDM networks when elastic transponders dynamically fit ageing of link margins, Pesic et al.
years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)
sd = np.linspace(0.04, 0.08, np.size(years)) # added SNR uncertainty SD - assumed to double over lifetime

NF = np.linspace(4.5,5.5,np.size(years)) # define the NF ageing of the amplifiers 
alpha = 0.2 + 0.00164*years # define the fibre ageing due to splice losses over time 
trxaging = (1 + 0.05*years).reshape(np.size(years),1)*(OSNRmeasBW/Bchrs) # define TRx ageing 
oxcaging = (0.03 + 0.007*years).reshape(np.size(years),1)*(OSNRmeasBW/Bchrs) # define filter ageing, assuming two filters per link, one at Tx and one at Rx

# find the worst-case margin required                         
fmD = sd[-1]*5 # static D margin is defined as 5xEoL SNR uncertainty SD that is added
fmDGI = sd[0]*5 # inital D margin used for GP approach (before LP establishment)

 # define the FEC thresholds - all correspond to BER of 2e-2 (2% FEC) - given by MATLAB bertool 
FT2 = 3.243 
FT4 = 6.254 
FT8 = 10.697
FT16 = 12.707
FT32 = 16.579
FT64 = 18.432
FT128 = 22.185

#  generate initial random requests 

def removekey(d, keysrc, keydes): # function for removing key from dict - used to remove blocked links 
    r = dict(d)                     # removes the link between nodes 'keysrc' and 'keydes'
    del r.get(keysrc)[keydes]
    return r    

# function for generating a real time SNR values with uncertainty
    
def SNRgen(pathind, yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths):  
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

# function for generating a fixed margin SNR value
      
def fmsnr(pathind, yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths):  
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
            # ======================================================================
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

# function for generating a new SNR value to test if uncertainty is dealt with

def SNRnew(pathind, yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths):  # function for generating a new SNR value to test if uncertainty is dealt with
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
    Gnlisp = np.zeros([numlinks,1])
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


def GPtrain(x,y):  # train standard GP using SKLearn - hyper priors controlled by bounds in kernel
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
        sigma = (sigma**2 + 1)**0.5 # weird definition of sigma in SKLearn - verified with my own implementation of R+W algorithm 2.1 - see heteroscedastickernel2.py first cell
        ystarp = ystar + sigma
        ystari = scaler.inverse_transform(ystar)
        ystarpi = scaler.inverse_transform(ystarp)
        sigmai = np.mean(ystarpi - ystari)
        return ystari, sigmai
# implementation of GP reach algorithm - finds reach to given confidence level in the face of uncertainty
def GPreach(pathind, numsig, yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths):
    truesnr = SNRgen(pathind,yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths)
    #print(truesnr)
    x = np.linspace(0,numpoints,numpoints)
    prmn, sigma = GPtrain(x,truesnr)
    prmn = np.mean(prmn)
    #print(prmn)
    if (prmn - FT128)/sigma > numsig:
        FT = FT128
        mfG = 128
    elif (prmn - FT64)/sigma > numsig:
        FT = FT64
        mfG = 64
    elif (prmn - FT32)/sigma > numsig:
        FT = FT32
        mfG = 32
    elif (prmn - FT16)/sigma > numsig:
        FT = FT16
        mfG = 16
    elif (prmn - FT8)/sigma > numsig:
        FT = FT8
        mfG = 8
    elif (prmn - FT4)/sigma > numsig:
        FT = FT4
        mfG = 4
    elif (prmn - FT2)/sigma > numsig:
        FT = FT2
        mfG = 2
    else:
        print("not able to establish a link")
    C = 2*np.log2(1+db2lin(prmn))  # Shannon capacity of AWGN channel under and average power constraint in bits/symb
    newsnrval = SNRnew(pathind,yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths)
    if newsnrval > FT:
        estbl = 1
        Um = prmn - numsig*sigma - FT
    else:
        estbl = 0
        #print(FT)
    return mfG, estbl, Um, C    
        
def rtmreach(pathind, numsig, yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths): # RTM = real-time + (D) margin
    truesnr = SNRgen(pathind,yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths)
    meansnr = np.mean(truesnr)
    if meansnr - fmD > FT128:
        FT = FT128
        mfG = 128
    elif meansnr - fmD > FT64:
        FT = FT64
        mfG = 64
    elif meansnr - fmD > FT32:
        FT = FT32
        mfG = 32
    elif meansnr - fmD > FT16:
        FT = FT16
        mfG = 16
    elif meansnr - fmD > FT8:
        FT = FT8
        mfG = 8
    elif meansnr - fmD > FT4:
        FT = FT4
        mfG = 4
    elif meansnr - fmD > FT2:
        FT = FT2
        mfG = 2
    else:
        print("not able to establish a link")

    C = 2*np.log2(1+db2lin(meansnr)) # this yields capacity in bits/sym
    if SNRnew(pathind,yearind, nyqch, edgeinds, edgelens, numlamlk, pthdists, pths) > FT:
        estbl = 1
        Um = meansnr - fmD - FT
    else:
        estbl = 0
    return mfG, estbl, Um, C

def fmreach(pathind, edgeinds, edgelens, numlamlk, pthdists, pths):
    
    gnSNRF = fmsnr(pathind, -1, False, edgeinds, edgelens, numlamlk, pthdists, pths)
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
    return MF, UmF


def gpireach(pathind, edgeinds, edgelens, numlamlk, pthdists, pths):
    
    gnSNRG = fmsnr(pathind, 0, False, edgeinds, edgelens, numlamlk, pthdists, pths)
    
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
    return MFG, UmG

def findroutesrand(nodes, secondpath, numreq):
    dis = []
    path = []
    if graphA == graphT:
        for _ in range(numreq):           
            srcnd, desnd = requestgen(graphA)
            d, p = dijkstra({'1':{'2':240,'11':240},'2':{'1':240,'3':240},'3':{'2':240,'4':240},    
                            '4':{'3':240,'5':240},'5':{'4':240,'6':240},'6':{'5':240,'7':240}, '7':{'6':240,'8':240},
                            '8':{'7':240,'9':240}, '9':{'8':240,'10':240}, '10':{'9':240,'11':240}, '11':{'10':240,'1':240}
                            } , srcnd, desnd)
            dis.append(d)
            path.append(p)
            if secondpath:
                shgraph = removekey({'1':{'2':240,'11':240},'2':{'1':240,'3':240},'3':{'2':240,'4':240},    
                                     '4':{'3':240,'5':240},'5':{'4':240,'6':240},'6':{'5':240,'7':240}, '7':{'6':240,'8':240},
                                     '8':{'7':240,'9':240}, '9':{'8':240,'10':240}, '10':{'9':240,'11':240}, '11':{'10':240,'1':240}
                                     }, p[0],p[1])  # remove the first link of the shortest path, between the first two nodes traversed
                d2, p2 = dijkstra(shgraph , srcnd, desnd)
                dis.append(d2)
                path.append(p2)
    if graphA == graphB:
        for _ in range(numreq):           
            srcnd, desnd = requestgen(graphA)
            d, p = dijkstra({'1':{'2':720,'13':80},'2':{'1':720,'3':160},'3':{'2':160,'4':240},    
                            '4':{'3':240,'5':160},'5':{'4':160,'6':80},'6':{'5':80,'7':240}, '7':{'6':240,'8':80},
                            '8':{'7':80,'9':400}, '9':{'8':400,'10':160}, '10':{'9':160,'11':80}, '11':{'10':80,'12':160},
                             '12':{'11':160,'13':240}, '13':{'12':240,'1':80}
                             } , srcnd, desnd)
            dis.append(d)
            path.append(p)
            if secondpath:
                shgraph = removekey({'1':{'2':720,'13':80},'2':{'1':720,'3':160},'3':{'2':160,'4':240},    
                                    '4':{'3':240,'5':160},'5':{'4':160,'6':80},'6':{'5':80,'7':240}, '7':{'6':240,'8':80},
                                     '8':{'7':80,'9':400}, '9':{'8':400,'10':160}, '10':{'9':160,'11':80}, '11':{'10':80,'12':160},
                                     '12':{'11':160,'13':240}, '13':{'12':240,'1':80}
                                      }, p[0],p[1])  # remove the first link of the shortest path, between the first two nodes traversed
                d2, p2 = dijkstra(shgraph , srcnd, desnd)
                dis.append(d2)
                path.append(p2)
    if graphA == graphD:
        for i in range(numreq):           
            srcnd, desnd = requestgen(graphA)
            d, p = dijkstra({'1':{'2':400,'12':160},'2':{'1':400,'3':240},'3':{'2':240,'4':320},    
                             '4':{'3':320,'5':240},'5':{'4':240,'6':160},'6':{'5':160,'7':80}, '7':{'6':80,'8':240},
                             '8':{'7':240,'9':240}, '9':{'8':240,'10':80}, '10':{'9':80,'11':80}, '11':{'10':80,'12':320}, '12':{'11':320,'1':160}
                             } , srcnd, desnd)
            dis.append(d)
            path.append(p)
            if secondpath:
                shgraph = removekey({'1':{'2':400,'12':160},'2':{'1':400,'3':240},'3':{'2':240,'4':320},    
                                     '4':{'3':320,'5':240},'5':{'4':240,'6':160},'6':{'5':160,'7':80}, '7':{'6':80,'8':240},
                                     '8':{'7':240,'9':240}, '9':{'8':240,'10':80}, '10':{'9':80,'11':80}, '11':{'10':80,'12':320}, '12':{'11':320,'1':160}
                                     }, p[0],p[1])  # remove the first link of the shortest path, between the first two nodes traversed
                d2, p2 = dijkstra(shgraph , srcnd, desnd)
                dis.append(d2)
                path.append(p2)
    return dis, path

def requestgen(graph):
            src = random.choice(list(graph.keys()))
            des = random.choice(list(graph.keys()))
            while des == src:
                des = random.choice(list(graph.keys()))
            return src, des

def getlinklen(shpath,graph,edges):  # takes nodes traversed as input and returns the lengths of each link and the edge indices 
    linklen = np.empty([len(shpath)-1,1])
    link = []
    for i in range(len(shpath)-1):
        linklen[i] = float((graph.get(shpath[i])).get(shpath[i+1]))
        link.append((edges.get(shpath[i])).get(shpath[i+1]))
    return linklen, link                

def randpthsgen(initreq, numreqcum):
    
    pthdists, pths = findroutesrand(nodesA, True, initreq)  
    
    for i in range(numyears-1):
        
        pthdist, pth = findroutesrand(nodesA, True, numreqcum[i])  
        for i in range(len(pthdist)):
            pthdists.append(pthdist[i])
            pths.append(pth[i])
    
    return pthdists, pths
    
numreq = [int(20*(1.2)**(x)) for x in years]
numreqcum = [numreq[i+1] - numreq[i] for i in range(numyears-1)]


def thrptcalcinitgpi(gpimf, gpiUm, numpths):
    ratesgpi = np.empty([numpths,1])
    for i in range(numpths):
        ratesgpi[i] = rateconv(gpimf[i][0])
    totthrptgpi = np.sum(ratesgpi, axis=0)/1e3
    #totthrptdiffi = totthrptgpi - totthrptfm
    totgpiUm = np.sum(gpiUm, axis=0)
    return totthrptgpi, totgpiUm

def thrptcalcinitfm(fmmf, fmUm, numpths):
    ratesfm = np.empty([numpths,1])
    for i in range(numpths):
        ratesfm[i] = rateconv(fmmf[i][0])
    totthrptfm = np.sum(ratesfm, axis=0)/1e3
    totfmUm = np.sum(fmUm, axis=0)
    return totthrptfm, totfmUm


def thrptcalc(gpmf, gpUm, gpshcp, rtmmf, rtmUm, rtmshcp, numpths):
    ratesgp = np.empty([numpths,1])
    ratesrtm = np.empty([numpths,1])
    FECOH = 0.2
    for i in range(numpths):
        ratesgp[i] = rateconv(gpmf[i][0])
        ratesrtm[i] = rateconv(rtmmf[i][0])
    totthrptgp = np.sum(ratesgp, axis=0)/1e3
    totthrptrtm = np.sum(ratesrtm, axis=0)/1e3
    totgpshcp = np.sum(gpshcp,axis=0)*Rs*(1-FECOH)*2/1e3
    totrtmshcp = np.sum(rtmshcp,axis=0)*Rs*(1-FECOH)*2/1e3

    totUmgp = np.sum(gpUm,axis=0)
    totUmrtm = np.sum(rtmUm,axis=0)
 
    return totthrptgp, totUmgp, totgpshcp, totthrptrtm, totUmrtm,  totrtmshcp

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

def randyearloop(initreq, numreqcum):
    
    gpUmsave = []
    gpmfsave = []
    rtmUmsave = []
    rtmmfsave = []
    
    numlamwc = 80*np.ones([numedgesA,1])
    
    totfmUm = np.empty([numyears,1])
    totthrptfm = np.empty([numyears,1])
    totthrptgp = np.empty([numyears,1])
    totUmgp = np.empty([numyears,1])
    #totthrptdiffgp = np.empty([numyears,1])
    totgpshcp = np.empty([numyears,1])
    totthrptrtm = np.empty([numyears,1])
    totUmrtm = np.empty([numyears,1])
    #totthrptdiffrtm = np.empty([numyears,1])
    totrtmshcp = np.empty([numyears,1])
    #totthrptdiffsh = np.empty([numyears,1])
    
    for i in range(numyears):
        
        if i == 0: 
            pthdists, pths = findroutesrand(nodesA, True, initreq)  
        else:
            pthdist, pth = findroutesrand(nodesA, True, numreqcum[i-1])  
            for k in range(len(pthdist)):
                pthdists.append(pthdist[k])
                pths.append(pth[k])

        numpths = len(pthdists)  
        edgeinds = [] # indices of each edge traversed for each path 
        edgelens = [] # lengths of each edge traversed for each path 
        numlamlk = np.zeros([numedgesA,1])
        for j in range(len(pths)):
            edgeinds.append(getlinklen(pths[j], graphA, edgesA)[1])  # transparent network: only need total distance for each path 
            numlamlk[edgeinds[j]] = numlamlk[edgeinds[j]] + 1
            edgelens.append(getlinklen(pths[j], graphA, edgesA)[0])  # transparent network: only need total distance for each path 
        if i == 0:
            gpimf = np.empty([numpths,1]) # GP initial modulation format
            gpiUm = np.empty([numpths,1]) # GP initial U margins 
            for l in range(numpths):
                gpimf[l],  gpiUm[l] = gpireach(l, edgeinds, edgelens, numlamlk, pthdists, pths)
            totthrptgpi, totgpiUm = thrptcalcinitgpi(gpimf, gpiUm, numpths)
            
        #  section 6: implement GP reach algorithm
        fmmf = np.empty([numpths,1]) # fixed margin modulation format 
        fmUm = np.empty([numpths,1]) # fixed margin U margins 
        gpmf = np.empty([numpths,1])
        gpestbl = np.empty([numpths,1])
        gpUm = np.empty([numpths,1])
        gpshcp = np.empty([numpths,1])
        rtmmf = np.empty([numpths,1])
        rtmestbl = np.empty([numpths,1])
        rtmUm = np.empty([numpths,1])
        rtmshcp = np.empty([numpths,1])
        start = time.time()
        for j in range(numpths):
              # change this later - will need to implement a time loop for the whole script, put it all in a big function
            fmmf[j],  fmUm[j] = fmreach(j, edgeinds, edgelens, numlamwc, pthdists, pths)  
            gpmf[j], gpestbl[j], gpUm[j], gpshcp[j] = GPreach(j, 5, i, False, edgeinds, edgelens, numlamlk, pthdists, pths)
            rtmmf[j], rtmestbl[j], rtmUm[j], rtmshcp[j] = rtmreach(j, 5, i, False, edgeinds, edgelens, numlamlk, pthdists, pths)
        end = time.time()
        
        print("Iter " + str(i) + " took " + str( '%.3f' % ((end-start)/60)  ) + " minutes")
        
        gpUmsave.append(gpUm)
        gpmfsave.append(gpmf)
        rtmUmsave.append(rtmUm)
        rtmmfsave.append(rtmmf)
    
        totthrptgp[i], totUmgp[i], totgpshcp[i], totthrptrtm[i], totUmrtm[i], totrtmshcp[i] = thrptcalc(gpmf, gpUm, gpshcp, rtmmf, rtmUm, rtmshcp, numpths)
        totthrptfm[i], totfmUm[i] = thrptcalcinitfm(fmmf, fmUm, numpths)
    
    return totthrptgpi, totgpiUm, totthrptfm, totfmUm, totthrptgp, totUmgp, totgpshcp, totthrptrtm, totUmrtm, totrtmshcp,  gpUmsave, gpmfsave, rtmUmsave, rtmmfsave
    
# %% Run algorithm

totthrptgpi, totgpiUm, totthrptfm, totfmUm, totthrptgp, totUmgp, totgpshcp, totthrptrtm, totUmrtm, totrtmshcp,  gpUmsave, gpmfsave, rtmUmsave, rtmmfsave = randyearloop(20, numreqcum)

# %% section 7: determine throughput for the ring network 

gpthrptben = totthrptgp - totthrptrtm

totthrptdiffgp = ((totthrptgp - totthrptfm)/totthrptfm)*100

totthrptdiffrtm = ((totthrptrtm - totthrptfm)/totthrptfm)*100

totthrptdiffsh = ((totgpshcp - totthrptfm)/totthrptfm)*100

randiter = 5

if graphA == graphT:
    np.savetxt('totthrptgpirdT' + str(randiter) + '.csv', totthrptgpi, delimiter=',') 
    np.savetxt('totgpiUmrdT' + str(randiter) + '.csv', totgpiUm, delimiter=',') 
    np.savetxt('totthrptfmrdT' + str(randiter) + '.csv', totthrptfm, delimiter=',') 
    np.savetxt('totfmUmrdT' + str(randiter) + '.csv', totfmUm, delimiter=',') 
    np.savetxt('totthrptgprdT' + str(randiter) + '.csv', totthrptgp, delimiter=',') 
    np.savetxt('totUmgprdT' + str(randiter) + '.csv', totUmgp, delimiter=',') 
    np.savetxt('totgpshcprdT' + str(randiter) + '.csv', totgpshcp, delimiter=',') 
    np.savetxt('totthrptrtmrdT' + str(randiter) + '.csv', totthrptrtm, delimiter=',') 
    np.savetxt('totUmrtmrdT' + str(randiter) + '.csv', totUmrtm, delimiter=',') 
    np.savetxt('totrtmshcprdT' + str(randiter) + '.csv', totrtmshcp, delimiter=',') 
    #np.savetxt('gpUmsaverdT.csv', gpUmsave, delimiter=',') 
    #np.savetxt('gpmfsaverdT.csv', gpmfsave, delimiter=',') 
    #np.savetxt('rtmUmsaverdT.csv', rtmUmsave, delimiter=',') 
    #np.savetxt('rtmmfsaverdT.csv', rtmmfsave, delimiter=',') 

if graphA == graphD:
    np.savetxt('totthrptgpirdD' + str(randiter) + '.csv', totthrptgpi, delimiter=',') 
    np.savetxt('totgpiUmrdD' + str(randiter) + '.csv', totgpiUm, delimiter=',') 
    np.savetxt('totthrptfmrdD' + str(randiter) + '.csv', totthrptfm, delimiter=',') 
    np.savetxt('totfmUmrdD' + str(randiter) + '.csv', totfmUm, delimiter=',') 
    np.savetxt('totthrptgprdD' + str(randiter) + '.csv', totthrptgp, delimiter=',') 
    np.savetxt('totUmgprdD' + str(randiter) + '.csv', totUmgp, delimiter=',') 
    np.savetxt('totgpshcprdD' + str(randiter) + '.csv', totgpshcp, delimiter=',') 
    np.savetxt('totthrptrtmrdD' + str(randiter) + '.csv', totthrptrtm, delimiter=',') 
    np.savetxt('totUmrtmrdD' + str(randiter) + '.csv', totUmrtm, delimiter=',') 
    np.savetxt('totrtmshcprdD' + str(randiter) + '.csv', totrtmshcp, delimiter=',') 
    #np.savetxt('gpUmsaverdD.csv', gpUmsave, delimiter=',') 
    #np.savetxt('gpmfsaverdD.csv', gpmfsave, delimiter=',') 
    #np.savetxt('rtmUmsaverdD.csv', rtmUmsave, delimiter=',') 
    #np.savetxt('rtmmfsaverdD.csv', rtmmfsave, delimiter=',') 

if graphA == graphB:
    np.savetxt('totthrptgpirdB' + str(randiter) + '.csv', totthrptgpi, delimiter=',') 
    np.savetxt('totgpiUmrdB' + str(randiter) + '.csv', totgpiUm, delimiter=',') 
    np.savetxt('totthrptfmrdB' + str(randiter) + '.csv', totthrptfm, delimiter=',') 
    np.savetxt('totfmUmrdB' + str(randiter) + '.csv', totfmUm, delimiter=',') 
    np.savetxt('totthrptgprdB' + str(randiter) + '.csv', totthrptgp, delimiter=',') 
    np.savetxt('totUmgprdB' + str(randiter) + '.csv', totUmgp, delimiter=',') 
    np.savetxt('totgpshcprdB' + str(randiter) + '.csv', totgpshcp, delimiter=',') 
    np.savetxt('totthrptrtmrdB' + str(randiter) + '.csv', totthrptrtm, delimiter=',') 
    np.savetxt('totUmrtmrdB' + str(randiter) + '.csv', totUmrtm, delimiter=',') 
    np.savetxt('totrtmshcprdB' + str(randiter) + '.csv', totrtmshcp, delimiter=',') 
    #np.savetxt('gpUmsaverdB.csv', gpUmsave, delimiter=',') 
    #np.savetxt('gpmfsaverdB.csv', gpmfsave, delimiter=',') 
    #np.savetxt('rtmUmsaverdB.csv', rtmUmsave, delimiter=',') 
    #np.savetxt('rtmmfsaverdB.csv', rtmmfsave, delimiter=',')  
    
#lkind = 25   
lkind = 4

gpmf = np.empty([numyears,1])
gpUm = np.empty([numyears,1])
for i in range(numyears):
    gpmf[i] = gpmfsave[i][lkind]
    gpUm[i] = gpUmsave[i][lkind]

rtmmf = np.empty([numyears,1])
rtmUm = np.empty([numyears,1])
for i in range(numyears):
    rtmmf[i] = rtmmfsave[i][lkind]
    rtmUm[i] = rtmUmsave[i][lkind]

if graphA == graphT:
    np.savetxt('gpUmrdT' + str(randiter) + '.csv', gpUm, delimiter=',') 
    np.savetxt('gpmfrdT' + str(randiter) + '.csv', gpmf, delimiter=',') 
    np.savetxt('rtmUmrdT' + str(randiter) + '.csv', rtmUm, delimiter=',') 
    np.savetxt('rtmmfrdT' + str(randiter) + '.csv', rtmmf, delimiter=',') 
if graphA == graphD:
    np.savetxt('gpUmrdD' + str(randiter) + '.csv', gpUm, delimiter=',') 
    np.savetxt('gpmfrdD' + str(randiter) + '.csv', gpmf, delimiter=',') 
    np.savetxt('rtmUmrdD' + str(randiter) + '.csv', rtmUm, delimiter=',') 
    np.savetxt('rtmmfrdD' + str(randiter) + '.csv', rtmmf, delimiter=',') 
if graphA == graphB:
    np.savetxt('gpUmrdB' + str(randiter) + '.csv', gpUm, delimiter=',') 
    np.savetxt('gpmfrdB' + str(randiter) + '.csv', gpmf, delimiter=',') 
    np.savetxt('rtmUmrdB' + str(randiter) + '.csv', rtmUm, delimiter=',') 
    np.savetxt('rtmmfrdB' + str(randiter) + '.csv', rtmmf, delimiter=',') 

# %% plotting 

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

if graphA == graphB:
    suffix = "B"
elif graphA == graphD:
    suffix = "D"
elif graphA == graphT:
    suffix = "T"

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#totthrptfmpl = totthrptfm*np.ones([numyears,1])

ln1 = ax1.plot(years, totthrptgp,'--', color = 'b',label = 'GP')
#ln2 = ax1.plot(years, totthrptfmpl,'-', color = 'r',label = 'FM' )
ln2 = ax1.plot(years, totthrptfm,'-', color = 'r',label = 'FM' )
ln3 = ax2.plot(years, totgpshcp,'-.', color = 'g',label = 'Sh.')

ln4 = ax1.plot(years, totthrptrtm,':', color = 'b',label = 'RTM')
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
plt.savefig('totalthrptrd' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

totthrptfmpl = totthrptfm*np.ones([numyears,1])

ln1 = ax1.plot(years, totthrptdiffgp,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptdiffrtm,':', color = 'r',label = 'RTM' )
ln3 = ax2.plot(years, totthrptdiffsh,'-', color = 'g',label = 'Sh.')
    
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
plt.savefig('totalthrptdiffrd' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()
    
# %%


#imodfN = [int(i) for i in modfN[linkind1]]
#imodfrtN = [int(i) for i in modfrtN[linkind1]]

#imfplN = [int(i) for i in mfplN[linkind1]]
y2lb = ['4','5']

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ln1 = ax1.plot(years, gpUm,'--',color = 'b',label = 'GP U')
ln2 = ax1.plot(years, rtmUm,'--',color = 'g',label = 'RTM U')
#ln6 = ax1.plot(years, fmUmpl[linkind1],'--',color = 'r', label = "FM U")

ln3 = ax2.plot(years, gpmf,'-',color = 'b',label = 'GP SE')
ln4 = ax2.plot(years, rtmmf,'-',color = 'g',label = 'RTM SE')
#ln3 = ax2.plot(years, fmmfpl[linkind1],'-',color = 'r',label = 'FM SE')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("U margin (dB)")
ax2.set_ylabel("spectral efficiency (bits/sym)")

ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([-1, 4])
ax2.set_ylim([15.7, 32.3])

ax2.set_yticks([16,32])
ax2.set_yticklabels(y2lb)
lns = ln1+ln2 +ln3+ln4 # +ln5+ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('UmarginGPbenefitrd' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% heteroscedastic data generation: normal increase in SNR st. dev. and ageing, linear loading 
hetdatagen = True
if hetdatagen:
    def datatshet(edgelen, Lspans, numlam, NF,sd, alpha, yearind, nyqch):
            Ls = Lspans
            D = Disp
            gam = NLco
            lam = 1550 # operating wavelength centre [nm]
            f = 299792458/(lam*1e-9) # operating frequency [Hz]
            c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
            Rs = 32 # symbol rate [GBaud]
            h = 6.63*1e-34  # Planck's constant [Js]
            if nyqch:
                NchNy = numlam
                BWNy = (NchNy*Rs)/1e3 
            else:
                NchRS = numlam
                Df = 50 # 50 GHz grid 
                BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
            allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
            beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
            Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
            Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
            numspans = int(edgelen/Lspans)
            
            numpch = len(PchdBm)
            Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
            if nyqch:
                Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
            else:
                Gwdmsw = Pchsw/(BchRS*1e9)
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
            G = alpha*Ls
            NFl = 10**(NF/10) 
            Gl = 10**(G/10) 
             # [W] the ASE noise power in one Nyquist channel across all spans # NEEDS FIXING - CURRENTLY FOR NYQUIST CASE
            if nyqch:
                Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
            else:
                Pasesw = NFl*h*f*(Gl - 1)*BchRS*1e9
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*BchRS*1e9) # NEEDS FIXING - CURRENTLY FOR NYQUIST CASE
            Popt = PchdBm[np.argmax(snrsw)]     
            
            if nyqch:
                Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
                Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
            else:
                Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
                Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans
                Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*BchRS*1e9*numspans
            Pch = 1e-3*10**(Popt/10) 
            if nyqch:            
                snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxagingh[yearind] + 2*oxcagingh[yearind]) # subtract static ageing effects
                #snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaginghextdeg[yearind] + oxcagingh[yearind]) # subtract static ageing effects
            else:            
                snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxagingh[yearind] + 2*oxcagingh[yearind]) # subtract static ageing effects
                #snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxaginghextdeg[yearind] + oxcagingh[yearind]) # subtract static ageing effects
            snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1) # add TRx B2B noise 
            #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
            sdnorm = sd # noise on each link is assumed to be proportional to the link length 
            return lin2db(snr) + np.random.normal(0,sdnorm,1) 
        
    
    numyrsh = 200
    yearsh = np.linspace(0,10,numyrsh)
    #numlamh = np.linspace(20,80,numyrsh,dtype=int)
    numlamh = [int(13*1.2**(i/10)) for i in range(numyrsh)]
    NFh = np.linspace(4.5,5.5,numyrsh)
    sdh = np.linspace(0.04,0.08,numyrsh)  # normal ageing 
    #sdh1 = np.linspace(0.04,0.06,int(numyrsh/2)+1)  # increase in ageing rate 
    #sdh2 = np.linspace(0.0604,0.1,int(numyrsh/2)-1) # increase in ageing rate 

    #sdh = np.append(sdh1,sdh2)
   
    #sdh = 0.04 + 0.0004*yearsh**2
    #sdh = 0.04 + (0.04/10**0.5)*yearsh**0.5
    #sdh = 0.04 + (0.36/10**0.5)*yearsh**0.5
    #sdh = 0.04 + 8e-4*yearsh**2
    alphah = 0.2 + 0.00164*yearsh
    hetdata = np.empty([numyrsh,numedgesA])
    trxagingh = (1 + 0.05*yearsh).reshape(np.size(yearsh),1) 
    #trxaginghextdeg1 = [((1 + 0.05*yearsh[i])*2) for i in range(int(numyrsh/2))] 
    #trxaginghextdeg2 = [((trxaginghextdeg1[-1]/2 + 0.1*yearsh[i])*2) for i in range(int(numyrsh/2))] 
    #trxaginghextdeg = np.append(trxaginghextdeg1, trxaginghextdeg2)
    oxcagingh = (0.03 + 0.007*yearsh).reshape(np.size(yearsh),1)
    #linkPopt = np.empty([numyears,1])

    plt.plot(yearsh, sdh, label = 'linear')
    #plt.plot(yearsh, sdh2, label = 'quadratic')
    #plt.plot(yearsh, sdh3, label = 'square root')
    plt.xlabel("time (years)")
    plt.ylabel("$\sigma$(dB)")
    plt.legend()
    plt.savefig('sigmavstime.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.plot(yearsh, trxagingh, label = 'linear')
    plt.xlabel("time (years)")
    plt.ylabel("TRx ageing penalty (dB)")
    plt.legend()
    #plt.savefig('sigmavstime.pdf', dpi=200,bbox_inches='tight')
    plt.show()
 
    testlens = [400,800,1200,1600]
    hetdata = np.empty([len(testlens),numyrsh])
    for i in range(len(testlens)):
        for j in range(numyrsh):
            hetdata[i][j] = datatshet(testlens[i], 80, numlamh[j], NFh[j], sdh[j], alphah[j], j, False)
    #np.savetxt('hetdata.csv', hetdata, delimiter=',')
    np.savetxt('hetdata20.csv', hetdata, delimiter=',')
    #np.savetxt('hetdataextdeg.csv', hetdata, delimiter=',')

# %% heteroscedastic data gen for 1 year with a sudden increase in the TRx penalty 

if hetdatagen:
    def datatshetsh(edgelen, Lspans, numlam, NF,sd, alpha, yearind, nyqch):
            Ls = Lspans
            D = Disp
            gam = NLco
            lam = 1550 # operating wavelength centre [nm]
            f = 299792458/(lam*1e-9) # operating frequency [Hz]
            c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
            Rs = 32 # symbol rate [GBaud]
            h = 6.63*1e-34  # Planck's constant [Js]
            if nyqch:
                NchNy = numlam
                BWNy = (NchNy*Rs)/1e3 
            else:
                NchRS = numlam
                Df = 50 # 50 GHz grid 
                BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
            allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
            beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
            Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
            Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
            numspans = int(edgelen/Lspans)
            
            numpch = len(PchdBm)
            Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
            if nyqch:
                Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
            else:
                Gwdmsw = Pchsw/(BchRS*1e9)
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
            G = alpha*Ls
            NFl = 10**(NF/10) 
            Gl = 10**(G/10) 
             # [W] the ASE noise power in one Nyquist channel across all spans # NEEDS FIXING - CURRENTLY FOR NYQUIST CASE
            if nyqch:
                Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
            else:
                Pasesw = NFl*h*f*(Gl - 1)*BchRS*1e9
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*BchRS*1e9) # NEEDS FIXING - CURRENTLY FOR NYQUIST CASE
            Popt = PchdBm[np.argmax(snrsw)]     
            
            if nyqch:
                Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
                Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
            else:
                Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
                Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans
                Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*BchRS*1e9*numspans
            Pch = 1e-3*10**(Popt/10) 
            if nyqch:            
                #snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxagingh[yearind] + oxcagingh[yearind]) # subtract static ageing effects
                snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaginghextdeg[yearind] + 2*oxcagingh[yearind]) # subtract static ageing effects
            else:            
                #snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxagingh[yearind] + oxcagingh[yearind]) # subtract static ageing effects
                snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxaginghextdeg[yearind] + 2*oxcagingh[yearind]) # subtract static ageing effects
            snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1) # add TRx B2B noise 
            #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
            sdnorm = sd # noise on each link is assumed to be proportional to the link length 
            return lin2db(snr) + np.random.normal(0,sdnorm,1) 
        
    
    numyrsh = 200
    yearshst = np.linspace(0,1,numyrsh)
    #numlamhst = np.linspace(20,26,numyrsh,dtype=int)
    numlamhst = [int(20*1.2**(i/numyrsh)) for i in range(numyrsh)]
    NFhst = np.linspace(4.5,4.6,numyrsh)
    #sdh = np.linspace(0.04,0.08,numyrsh)  # normal ageing 
    sdh1 = np.linspace(0.04,0.042,int(numyrsh/2)+1)  # increase in ageing rate 
    sdh2 = np.linspace(0.08,0.082,int(numyrsh/2)-1) # increase in ageing rate 

    sdh = np.append(sdh1,sdh2)
   
    #sdh = 0.04 + 0.0004*yearsh**2
    #sdh = 0.04 + (0.04/10**0.5)*yearsh**0.5
    #sdh = 0.04 + (0.36/10**0.5)*yearsh**0.5
    #sdh = 0.04 + 8e-4*yearsh**2
    alphahst = 0.2 + 0.00164*yearshst
    hetdata = np.empty([numyrsh,numedgesA])
    #trxagingh = ((1 + 0.05*yearshst)*2).reshape(np.size(yearshst),1) 
    trxaginghextdeg1 = [(1 + 0.05*yearshst[i]) for i in range(int(numyrsh/2))] 
    trxaginghextdeg2 = [((trxaginghextdeg1[-1]*2 + 0.05*yearshst[i])) for i in range(int(numyrsh/2))] 
    trxaginghextdeg = np.append(trxaginghextdeg1, trxaginghextdeg2)
    oxcagingh = ((0.03 + 0.007*yearshst)*2).reshape(np.size(yearshst),1)
    #linkPopt = np.empty([numyears,1])

    plt.plot(yearshst, sdh, label = 'linear')
    #plt.plot(yearsh, sdh2, label = 'quadratic')
    #plt.plot(yearsh, sdh3, label = 'square root')
    plt.xlabel("time (years)")
    plt.ylabel("$\sigma$(dB)")
    plt.legend()
    plt.savefig('sigmavstime.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.plot(yearshst, trxaginghextdeg, label = 'linear')
    plt.xlabel("time (years)")
    plt.ylabel("TRx ageing penalty (dB)")
    plt.legend()
    plt.savefig('TRxageing.pdf', dpi=200,bbox_inches='tight')
    plt.show()
 
    testlens = [400,800,1200,1600]
    hetdata = np.empty([len(testlens),numyrsh])
    for i in range(len(testlens)):
        for j in range(numyrsh):
            hetdata[i][j] = datatshetsh(testlens[i], 80, numlamhst[j], NFhst[j], sdh[j], alphahst[j], j, False)
    #np.savetxt('hetdata.csv', hetdata, delimiter=',')
    np.savetxt('hetdataextdegst.csv', hetdata, delimiter=',')

# %% heteroscedastic data for different variance links 

if hetdatagen:
    def datatshetdv(edgelen, Lspans, numlam, NF,sd, alpha, yearind, nyqch):
            Ls = Lspans
            D = Disp
            gam = NLco
            lam = 1550 # operating wavelength centre [nm]
            f = 299792458/(lam*1e-9) # operating frequency [Hz]
            c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
            Rs = 32 # symbol rate [GBaud]
            h = 6.63*1e-34  # Planck's constant [Js]
            if nyqch:
                NchNy = numlam
                BWNy = (NchNy*Rs)/1e3 
            else:
                NchRS = numlam
                Df = 50 # 50 GHz grid 
                BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
            allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
            beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
            Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
            Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
            numspans = int(edgelen/Lspans)
            
            numpch = len(PchdBm)
            Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
            if nyqch:
                Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
            else:
                Gwdmsw = Pchsw/(BchRS*1e9)
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
            G = alpha*Ls
            NFl = 10**(NF/10) 
            Gl = 10**(G/10) 
             # [W] the ASE noise power in one Nyquist channel across all spans # NEEDS FIXING - CURRENTLY FOR NYQUIST CASE
            if nyqch:
                Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
            else:
                Pasesw = NFl*h*f*(Gl - 1)*BchRS*1e9
                snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*BchRS*1e9) # NEEDS FIXING - CURRENTLY FOR NYQUIST CASE
            Popt = PchdBm[np.argmax(snrsw)]     
            
            if nyqch:
                Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
                Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
            else:
                Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
                Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans
                Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*BchRS*1e9*numspans
            Pch = 1e-3*10**(Popt/10) 
            if nyqch:            
                #snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxagingh[yearind] + oxcagingh[yearind]) # subtract static ageing effects
                snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxagingh[yearind] + 2*oxcagingh[yearind]) # subtract static ageing effects
            else:            
                #snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxagingh[yearind] + oxcagingh[yearind]) # subtract static ageing effects
                snr = (Pch/(Pase + Gnli*BchRS*1e9)) - db2lin(trxagingh[yearind] + 2*oxcagingh[yearind]) # subtract static ageing effects
            snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1) # add TRx B2B noise 
            #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
            sdnorm = sd 
            return lin2db(snr) + np.random.normal(0,sdnorm,1) 
        
    
    numyrsh = 200
    yearshst = np.linspace(0,5,numyrsh)
    #numlamhst = [int(13*1.2**(i/20)) for i in range(100)]
    numlamhst = [int(13*1.2**(i/40)) for i in range(200)]
    NFhst = np.linspace(4.5,5.0,numyrsh)
    numlps = 3

    sdh1 = np.linspace(0.04,0.06,numyrsh)  # normal ageing 
    sdh2 = np.linspace(0.04,0.08,numyrsh)  # normal ageing 
    sdh3 = np.linspace(0.04,0.16,numyrsh)  # normal ageing
    
    sh = np.empty([numlps,numyrsh])
    sh[0] = sdh1
    sh[1] = sdh2
    sh[2] = sdh3
    

    #sdh = 0.04 + 0.0004*yearsh**2
    #sdh = 0.04 + (0.04/10**0.5)*yearsh**0.5
    #sdh = 0.04 + (0.36/10**0.5)*yearsh**0.5
    #sdh = 0.04 + 8e-4*yearsh**2
    alphahst = 0.2 + 0.00164*yearshst
    hetdata = np.empty([numyrsh,numedgesA])
    trxagingh = ((1 + 0.05*yearshst)).reshape(np.size(yearshst),1) 
    
    oxcagingh = ((0.03 + 0.007*yearshst)).reshape(np.size(yearshst),1)
    #linkPopt = np.empty([numyears,1])

    plt.plot(yearshst, sh[0], label = 'linear')
    #plt.plot(yearsh, sdh2, label = 'quadratic')
    #plt.plot(yearsh, sdh3, label = 'square root')
    plt.xlabel("time (years)")
    plt.ylabel("$\sigma$(dB)")
    plt.legend()
    #plt.savefig('sigmavstime.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.plot(yearshst, trxagingh, label = 'linear')
    plt.xlabel("time (years)")
    plt.ylabel("TRx ageing penalty (dB)")
    plt.legend()
    #plt.savefig('TRxageing.pdf', dpi=200,bbox_inches='tight')
    plt.show()
 
    
    hetdata = np.empty([numlps,numyrsh])
    for i in range(numlps):
        for j in range(numyrsh):
            hetdata[i][j] = datatshetdv(800, 80, numlamhst[j], NFhst[j], sh[i][j], alphahst[j], j, False)
    #np.savetxt('hetdata.csv', hetdata, delimiter=',')
    np.savetxt('hetdataextdegdv.csv', hetdata, delimiter=',')

# %%
