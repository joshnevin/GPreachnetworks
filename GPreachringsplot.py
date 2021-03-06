# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# U margin GP, random loading, DTAG
""" totgpiUmD1 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD2 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD3 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD4 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD5 = np.genfromtxt(open("totgpiUmrdD1.csv", "r"), delimiter=",", dtype =float)
totgpiUmD = (totgpiUmD1 + totgpiUmD2 + totgpiUmD3 + totgpiUmD4 + totgpiUmD5)/5 """

# total throughput FM DTAG, increasing loading 
totthrptfmD1 = np.genfromtxt(open("totthrptfmrdD1.csv", "r"), delimiter=",", dtype =float)
totthrptfmD2 = np.genfromtxt(open("totthrptfmrdD2.csv", "r"), delimiter=",", dtype =float)
totthrptfmD3 = np.genfromtxt(open("totthrptfmrdD3.csv", "r"), delimiter=",", dtype =float)
totthrptfmD4 = np.genfromtxt(open("totthrptfmrdD4.csv", "r"), delimiter=",", dtype =float)
totthrptfmD5 = np.genfromtxt(open("totthrptfmrdD5.csv", "r"), delimiter=",", dtype =float)
totthrptfmD = (totthrptfmD1 + totthrptfmD2 + totthrptfmD3 + totthrptfmD4 + totthrptfmD5)/5

# total FM U margin, random loading DTAG
""" totfmUmD1 = np.genfromtxt(open("totfmUmrdD1.csv", "r"), delimiter=",", dtype =float)
totfmUmD2 = np.genfromtxt(open("totfmUmrdD2.csv", "r"), delimiter=",", dtype =float)
totfmUmD3 = np.genfromtxt(open("totfmUmrdD3.csv", "r"), delimiter=",", dtype =float)
totfmUmD4 = np.genfromtxt(open("totfmUmrdD4.csv", "r"), delimiter=",", dtype =float)
totfmUmD5 = np.genfromtxt(open("totfmUmrdD5.csv", "r"), delimiter=",", dtype =float)
totfmUmD = (totfmUmD1 + totfmUmD2 + totfmUmD3 + totfmUmD4 + totfmUmD5)/5 """

# total throughput GP, DTAG, random loading 
totthrptgpD1 = np.genfromtxt(open("totthrptgprdD1.csv", "r"), delimiter=",", dtype =float)
totthrptgpD2 = np.genfromtxt(open("totthrptgprdD2.csv", "r"), delimiter=",", dtype =float)
totthrptgpD3 = np.genfromtxt(open("totthrptgprdD3.csv", "r"), delimiter=",", dtype =float)
totthrptgpD4 = np.genfromtxt(open("totthrptgprdD4.csv", "r"), delimiter=",", dtype =float)
totthrptgpD5 = np.genfromtxt(open("totthrptgprdD5.csv", "r"), delimiter=",", dtype =float)
totthrptgpD = (totthrptgpD1 + totthrptgpD2 + totthrptgpD3 + totthrptgpD4 + totthrptgpD5)/5

# total U margin GP, DTAG, random loading 
""" totUmgpD1 = np.genfromtxt(open("totUmgprdD1.csv", "r"), delimiter=",", dtype =float)
totUmgpD2 = np.genfromtxt(open("totUmgprdD2.csv", "r"), delimiter=",", dtype =float)
totUmgpD3 = np.genfromtxt(open("totUmgprdD3.csv", "r"), delimiter=",", dtype =float)
totUmgpD4 = np.genfromtxt(open("totUmgprdD4.csv", "r"), delimiter=",", dtype =float)
totUmgpD5 = np.genfromtxt(open("totUmgprdD5.csv", "r"), delimiter=",", dtype =float)
totUmgpD = (totUmgpD1 + totUmgpD2 + totUmgpD3 + totUmgpD4 + totUmgpD5)/5 """

# total throughput for Shannon limit, DTAG, GP, random loading 
totgpshcpD1 = np.genfromtxt(open("totgpshcprdD1.csv", "r"), delimiter=",", dtype =float)
totgpshcpD2 = np.genfromtxt(open("totgpshcprdD2.csv", "r"), delimiter=",", dtype =float)
totgpshcpD3 = np.genfromtxt(open("totgpshcprdD3.csv", "r"), delimiter=",", dtype =float)
totgpshcpD4 = np.genfromtxt(open("totgpshcprdD4.csv", "r"), delimiter=",", dtype =float)
totgpshcpD5 = np.genfromtxt(open("totgpshcprdD5.csv", "r"), delimiter=",", dtype =float)
totgpshcpD = (totgpshcpD1 + totgpshcpD2 + totgpshcpD3 + totgpshcpD4 + totgpshcpD5)/5

# total throughput RTM, DTAG, random loading 
totthrptrtmD1 = np.genfromtxt(open("totthrptrtmrdD1.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD2 = np.genfromtxt(open("totthrptrtmrdD2.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD3 = np.genfromtxt(open("totthrptrtmrdD3.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD4 = np.genfromtxt(open("totthrptrtmrdD4.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD5 = np.genfromtxt(open("totthrptrtmrdD5.csv", "r"), delimiter=",", dtype =float)
totthrptrtmD = (totthrptrtmD1 + totthrptrtmD2 + totthrptrtmD3 + totthrptrtmD4 + totthrptrtmD5)/5

# total throughput U margin, DTAG, RTM, random loading 
""" totUmrtmD1 = np.genfromtxt(open("totUmrtmrdD1.csv", "r"), delimiter=",", dtype =float)
totUmrtmD2 = np.genfromtxt(open("totUmrtmrdD2.csv", "r"), delimiter=",", dtype =float)
totUmrtmD3 = np.genfromtxt(open("totUmrtmrdD3.csv", "r"), delimiter=",", dtype =float)
totUmrtmD4 = np.genfromtxt(open("totUmrtmrdD4.csv", "r"), delimiter=",", dtype =float)
totUmrtmD5 = np.genfromtxt(open("totUmrtmrdD5.csv", "r"), delimiter=",", dtype =float)
totUmrtmD = (totUmrtmD1 + totUmrtmD2 + totUmrtmD3 + totUmrtmD4 + totUmrtmD5)/5 """

# total throughput for Shannon limit, DTAG, RTM, random loading 
totrtmshcprdD1 = np.genfromtxt(open("totrtmshcprdD1.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdD2 = np.genfromtxt(open("totrtmshcprdD2.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdD3 = np.genfromtxt(open("totrtmshcprdD3.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdD4 = np.genfromtxt(open("totrtmshcprdD4.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdD5 = np.genfromtxt(open("totrtmshcprdD5.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdD = (totrtmshcprdD1 + totrtmshcprdD2 + totrtmshcprdD3 + totrtmshcprdD4 + totrtmshcprdD5)/5

# U margin GP, DTAG, random loading 
""" gpUmD1 = np.genfromtxt(open("gpUmrdD1.csv", "r"), delimiter=",", dtype =float)
gpUmD2 = np.genfromtxt(open("gpUmrdD2.csv", "r"), delimiter=",", dtype =float)
gpUmD3 = np.genfromtxt(open("gpUmrdD3.csv", "r"), delimiter=",", dtype =float)
gpUmD4 = np.genfromtxt(open("gpUmrdD4.csv", "r"), delimiter=",", dtype =float)
gpUmD5 = np.genfromtxt(open("gpUmrdD5.csv", "r"), delimiter=",", dtype =float)
gpUmD = (gpUmD1 + gpUmD2 + gpUmD3 + gpUmD4 + gpUmD5)/5 """

# MF GP, DTAG, random loading 
""" gpmfD1 = np.genfromtxt(open("gpmfrdD1.csv", "r"), delimiter=",", dtype =float)
gpmfD2 = np.genfromtxt(open("gpmfrdD2.csv", "r"), delimiter=",", dtype =float)
gpmfD3 = np.genfromtxt(open("gpmfrdD3.csv", "r"), delimiter=",", dtype =float)
gpmfD4 = np.genfromtxt(open("gpmfrdD4.csv", "r"), delimiter=",", dtype =float)
gpmfD5 = np.genfromtxt(open("gpmfrdD5.csv", "r"), delimiter=",", dtype =float)
gpmfD = (gpmfD1 + gpmfD2 + gpmfD3 + gpmfD4 + gpmfD5)/5 """

# U margin RTM, DTAG, random loading 
""" rtmUmD1 = np.genfromtxt(open("rtmUmrdD1.csv", "r"), delimiter=",", dtype =float)
rtmUmD2 = np.genfromtxt(open("rtmUmrdD2.csv", "r"), delimiter=",", dtype =float)
rtmUmD3 = np.genfromtxt(open("rtmUmrdD3.csv", "r"), delimiter=",", dtype =float)
rtmUmD4 = np.genfromtxt(open("rtmUmrdD4.csv", "r"), delimiter=",", dtype =float)
rtmUmD5 = np.genfromtxt(open("rtmUmrdD5.csv", "r"), delimiter=",", dtype =float)
rtmUmD = (rtmUmD1 + rtmUmD2 + rtmUmD3 + rtmUmD4 + rtmUmD5)/5 """
# MF RTM, DTAG, random loading 
""" rtmmfD1 = np.genfromtxt(open("rtmmfrdD1.csv", "r"), delimiter=",", dtype =float)
rtmmfD2 = np.genfromtxt(open("rtmmfrdD2.csv", "r"), delimiter=",", dtype =float)
rtmmfD3 = np.genfromtxt(open("rtmmfrdD3.csv", "r"), delimiter=",", dtype =float)
rtmmfD4 = np.genfromtxt(open("rtmmfrdD4.csv", "r"), delimiter=",", dtype =float)
rtmmfD5 = np.genfromtxt(open("rtmmfrdD5.csv", "r"), delimiter=",", dtype =float)
rtmmfD = (rtmmfD1 + rtmmfD2 + rtmmfD3 + rtmmfD4 + rtmmfD5)/5 """
# U margin, GP initial, BT
""" totgpiUmB1 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB2 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB3 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB4 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB5 = np.genfromtxt(open("totgpiUmrdB1.csv", "r"), delimiter=",", dtype =float)
totgpiUmB = (totgpiUmB1 + totgpiUmB2 + totgpiUmB3 + totgpiUmB4 + totgpiUmB5)/5 """

# total throughput, FM, BT, random loading 
totthrptfmB1 = np.genfromtxt(open("totthrptfmrdB1.csv", "r"), delimiter=",", dtype =float)
totthrptfmB2 = np.genfromtxt(open("totthrptfmrdB2.csv", "r"), delimiter=",", dtype =float)
totthrptfmB3 = np.genfromtxt(open("totthrptfmrdB3.csv", "r"), delimiter=",", dtype =float)
totthrptfmB4 = np.genfromtxt(open("totthrptfmrdB4.csv", "r"), delimiter=",", dtype =float)
totthrptfmB5 = np.genfromtxt(open("totthrptfmrdB5.csv", "r"), delimiter=",", dtype =float)
totthrptfmB = (totthrptfmB1 + totthrptfmB2 + totthrptfmB3 + totthrptfmB4 + totthrptfmB5)/5

""" totfmUmB1 = np.genfromtxt(open("totfmUmrdB1.csv", "r"), delimiter=",", dtype =float)
totfmUmB2 = np.genfromtxt(open("totfmUmrdB2.csv", "r"), delimiter=",", dtype =float)
totfmUmB3 = np.genfromtxt(open("totfmUmrdB3.csv", "r"), delimiter=",", dtype =float)
totfmUmB4 = np.genfromtxt(open("totfmUmrdB4.csv", "r"), delimiter=",", dtype =float)
totfmUmB5 = np.genfromtxt(open("totfmUmrdB5.csv", "r"), delimiter=",", dtype =float)
totfmUmB = (totfmUmB1 + totfmUmB2 + totfmUmB3 + totfmUmB4 + totfmUmB5)/5 """
# total throughput, GP, BT, random loading 
totthrptgpB1 = np.genfromtxt(open("totthrptgprdB1.csv", "r"), delimiter=",", dtype =float)
totthrptgpB2 = np.genfromtxt(open("totthrptgprdB2.csv", "r"), delimiter=",", dtype =float)
totthrptgpB3 = np.genfromtxt(open("totthrptgprdB3.csv", "r"), delimiter=",", dtype =float)
totthrptgpB4 = np.genfromtxt(open("totthrptgprdB4.csv", "r"), delimiter=",", dtype =float)
totthrptgpB5 = np.genfromtxt(open("totthrptgprdB5.csv", "r"), delimiter=",", dtype =float)
totthrptgpB = (totthrptgpB1 + totthrptgpB2 + totthrptgpB3 + totthrptgpB4 + totthrptgpB5)/5

""" totUmgpB1 = np.genfromtxt(open("totUmgprdB1.csv", "r"), delimiter=",", dtype =float)
totUmgpB2 = np.genfromtxt(open("totUmgprdB2.csv", "r"), delimiter=",", dtype =float)
totUmgpB3 = np.genfromtxt(open("totUmgprdB3.csv", "r"), delimiter=",", dtype =float)
totUmgpB4 = np.genfromtxt(open("totUmgprdB4.csv", "r"), delimiter=",", dtype =float)
totUmgpB5 = np.genfromtxt(open("totUmgprdB5.csv", "r"), delimiter=",", dtype =float)
totUmgpB = (totUmgpB1 + totUmgpB2 + totUmgpB3 + totUmgpB4 + totUmgpB5)/5 """

# total Shannon-limited throughput, GP, BT, random loading  
totgpshcpB1 = np.genfromtxt(open("totgpshcprdB1.csv", "r"), delimiter=",", dtype =float)
totgpshcpB2 = np.genfromtxt(open("totgpshcprdB2.csv", "r"), delimiter=",", dtype =float)
totgpshcpB3 = np.genfromtxt(open("totgpshcprdB3.csv", "r"), delimiter=",", dtype =float)
totgpshcpB4 = np.genfromtxt(open("totgpshcprdB4.csv", "r"), delimiter=",", dtype =float)
totgpshcpB5 = np.genfromtxt(open("totgpshcprdB5.csv", "r"), delimiter=",", dtype =float)
totgpshcpB = (totgpshcpB1 + totgpshcpB2 + totgpshcpB3 + totgpshcpB4 + totgpshcpB5)/5

# total throughput, RTM, BT, random loading  
totthrptrtmB1 = np.genfromtxt(open("totthrptrtmrdB1.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB2 = np.genfromtxt(open("totthrptrtmrdB2.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB3 = np.genfromtxt(open("totthrptrtmrdB3.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB4 = np.genfromtxt(open("totthrptrtmrdB4.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB5 = np.genfromtxt(open("totthrptrtmrdB5.csv", "r"), delimiter=",", dtype =float)
totthrptrtmB = (totthrptrtmB1 + totthrptrtmB2 + totthrptrtmB3 + totthrptrtmB4 + totthrptrtmB5)/5
""" 
totUmrtmB1 = np.genfromtxt(open("totUmrtmrdB1.csv", "r"), delimiter=",", dtype =float)
totUmrtmB2 = np.genfromtxt(open("totUmrtmrdB2.csv", "r"), delimiter=",", dtype =float)
totUmrtmB3 = np.genfromtxt(open("totUmrtmrdB3.csv", "r"), delimiter=",", dtype =float)
totUmrtmB4 = np.genfromtxt(open("totUmrtmrdB4.csv", "r"), delimiter=",", dtype =float)
totUmrtmB5 = np.genfromtxt(open("totUmrtmrdB5.csv", "r"), delimiter=",", dtype =float)
totUmrtmB = (totUmrtmB1 + totUmrtmB2 + totUmrtmB3 + totUmrtmB4 + totUmrtmB5)/5 """
# total Shannon-limited throughput, RTM, BT, random loading  
totrtmshcprdB1 = np.genfromtxt(open("totrtmshcprdB1.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdB2 = np.genfromtxt(open("totrtmshcprdB2.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdB3 = np.genfromtxt(open("totrtmshcprdB3.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdB4 = np.genfromtxt(open("totrtmshcprdB4.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdB5 = np.genfromtxt(open("totrtmshcprdB5.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdB = (totrtmshcprdB1 + totrtmshcprdB2 + totrtmshcprdB3 + totrtmshcprdB4 + totrtmshcprdB5)/5

""" gpUmB1 = np.genfromtxt(open("gpUmrdB1.csv", "r"), delimiter=",", dtype =float)
gpUmB2 = np.genfromtxt(open("gpUmrdB2.csv", "r"), delimiter=",", dtype =float)
gpUmB3 = np.genfromtxt(open("gpUmrdB3.csv", "r"), delimiter=",", dtype =float)
gpUmB4 = np.genfromtxt(open("gpUmrdB4.csv", "r"), delimiter=",", dtype =float)
gpUmB5 = np.genfromtxt(open("gpUmrdB5.csv", "r"), delimiter=",", dtype =float)
gpUmB = (gpUmB1 + gpUmB2 + gpUmB3 + gpUmB4 + gpUmB5)/5 """
""" gpmfB1 = np.genfromtxt(open("gpmfrdB1.csv", "r"), delimiter=",", dtype =float)
gpmfB2 = np.genfromtxt(open("gpmfrdB2.csv", "r"), delimiter=",", dtype =float)
gpmfB3 = np.genfromtxt(open("gpmfrdB3.csv", "r"), delimiter=",", dtype =float)
gpmfB4 = np.genfromtxt(open("gpmfrdB4.csv", "r"), delimiter=",", dtype =float)
gpmfB5 = np.genfromtxt(open("gpmfrdB5.csv", "r"), delimiter=",", dtype =float)
gpmfB = (gpmfB1 + gpmfB2 + gpmfB3 + gpmfB4 + gpmfB5)/5 """
""" rtmUmB1 = np.genfromtxt(open("rtmUmrdB1.csv", "r"), delimiter=",", dtype =float)
rtmUmB2 = np.genfromtxt(open("rtmUmrdB2.csv", "r"), delimiter=",", dtype =float)
rtmUmB3 = np.genfromtxt(open("rtmUmrdB3.csv", "r"), delimiter=",", dtype =float)
rtmUmB4 = np.genfromtxt(open("rtmUmrdB4.csv", "r"), delimiter=",", dtype =float)
rtmUmB5 = np.genfromtxt(open("rtmUmrdB5.csv", "r"), delimiter=",", dtype =float)
rtmUmB = (rtmUmB1 + rtmUmB2 + rtmUmB3 + rtmUmB4 + rtmUmB5)/5 """
""" rtmmfB1 = np.genfromtxt(open("rtmmfrdB1.csv", "r"), delimiter=",", dtype =float)
rtmmfB2 = np.genfromtxt(open("rtmmfrdB2.csv", "r"), delimiter=",", dtype =float)
rtmmfB3 = np.genfromtxt(open("rtmmfrdB3.csv", "r"), delimiter=",", dtype =float)
rtmmfB4 = np.genfromtxt(open("rtmmfrdB4.csv", "r"), delimiter=",", dtype =float)
rtmmfB5 = np.genfromtxt(open("rtmmfrdB5.csv", "r"), delimiter=",", dtype =float)
rtmmfB = (rtmmfB1 + rtmmfB2 + rtmmfB3 + rtmmfB4 + rtmmfB5)/5 """
# total throughput FM, ES, random loading 
totthrptfmT1 = np.genfromtxt(open("totthrptfmrdT1.csv", "r"), delimiter=",", dtype =float)
totthrptfmT2 = np.genfromtxt(open("totthrptfmrdT2.csv", "r"), delimiter=",", dtype =float)
totthrptfmT3 = np.genfromtxt(open("totthrptfmrdT3.csv", "r"), delimiter=",", dtype =float)
totthrptfmT4 = np.genfromtxt(open("totthrptfmrdT4.csv", "r"), delimiter=",", dtype =float)
totthrptfmT5 = np.genfromtxt(open("totthrptfmrdT5.csv", "r"), delimiter=",", dtype =float)
totthrptfmT = (totthrptfmT1 + totthrptfmT2 + totthrptfmT3 + totthrptfmT4 + totthrptfmT5)/5
# total throughput GP, ES, random loading 
totthrptgpT1 = np.genfromtxt(open("totthrptgprdT1.csv", "r"), delimiter=",", dtype =float)
totthrptgpT2 = np.genfromtxt(open("totthrptgprdT2.csv", "r"), delimiter=",", dtype =float)
totthrptgpT3 = np.genfromtxt(open("totthrptgprdT3.csv", "r"), delimiter=",", dtype =float)
totthrptgpT4 = np.genfromtxt(open("totthrptgprdT4.csv", "r"), delimiter=",", dtype =float)
totthrptgpT5 = np.genfromtxt(open("totthrptgprdT5.csv", "r"), delimiter=",", dtype =float)
totthrptgpT = (totthrptgpT1 + totthrptgpT2 + totthrptgpT3 + totthrptgpT4 + totthrptgpT5)/5
# total throughput  GP shannon, ES, random loading 
totgpshcpT1 = np.genfromtxt(open("totgpshcprdT1.csv", "r"), delimiter=",", dtype =float)
totgpshcpT2 = np.genfromtxt(open("totgpshcprdT2.csv", "r"), delimiter=",", dtype =float)
totgpshcpT3 = np.genfromtxt(open("totgpshcprdT3.csv", "r"), delimiter=",", dtype =float)
totgpshcpT4 = np.genfromtxt(open("totgpshcprdT4.csv", "r"), delimiter=",", dtype =float)
totgpshcpT5 = np.genfromtxt(open("totgpshcprdT5.csv", "r"), delimiter=",", dtype =float)
totgpshcpT = (totgpshcpT1 + totgpshcpT2 + totgpshcpT3 + totgpshcpT4 + totgpshcpT5)/5
# total throughput RTM, ES, random loading 
totthrptrtmT1 = np.genfromtxt(open("totthrptrtmrdT1.csv", "r"), delimiter=",", dtype =float)
totthrptrtmT2 = np.genfromtxt(open("totthrptrtmrdT2.csv", "r"), delimiter=",", dtype =float)
totthrptrtmT3 = np.genfromtxt(open("totthrptrtmrdT3.csv", "r"), delimiter=",", dtype =float)
totthrptrtmT4 = np.genfromtxt(open("totthrptrtmrdT4.csv", "r"), delimiter=",", dtype =float)
totthrptrtmT5 = np.genfromtxt(open("totthrptrtmrdT5.csv", "r"), delimiter=",", dtype =float)
totthrptrtmT = (totthrptrtmT1 + totthrptrtmT2 + totthrptrtmT3 + totthrptrtmT4 + totthrptrtmT5)/5
# total throughput RTM Shannon, ES, random loading
totrtmshcprdT1 = np.genfromtxt(open("totrtmshcprdT1.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdT2 = np.genfromtxt(open("totrtmshcprdT2.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdT3 = np.genfromtxt(open("totrtmshcprdT3.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdT4 = np.genfromtxt(open("totrtmshcprdT4.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdT5 = np.genfromtxt(open("totrtmshcprdT5.csv", "r"), delimiter=",", dtype =float)
totrtmshcprdT = (totrtmshcprdT1 + totrtmshcprdT2 + totrtmshcprdT3 + totrtmshcprdT4 + totrtmshcprdT5)/5
# total throughput RTM, constant loading, all topologies
totrtmshcpnoloadingB = np.genfromtxt(open("totrtmshcpB.csv", "r"), delimiter=",", dtype =float)
totrtmshcpnoloadingD = np.genfromtxt(open("totrtmshcpD.csv", "r"), delimiter=",", dtype =float)
totrtmshcpnoloadingT = np.genfromtxt(open("totrtmshcpT.csv", "r"), delimiter=",", dtype =float)
# total throughput RTM Shannon, constant loading, all topologies
totrtmshcpnoloadingdiffB = np.genfromtxt(open("totrtmshcpdiffB.csv", "r"), delimiter=",", dtype =float)
totrtmshcpnoloadingdiffD = np.genfromtxt(open("totrtmshcpdiffD.csv", "r"), delimiter=",", dtype =float)
totrtmshcpnoloadingdiffT = np.genfromtxt(open("totrtmshcpdiffT.csv", "r"), delimiter=",", dtype =float)
# total throughput diff RTM Shannon, increasing loading, all topologies
totrtmshcpdiffrdD = 100*(totrtmshcprdD - totthrptfmD)/totthrptfmD
totrtmshcpdiffrdB = 100*(totrtmshcprdB - totthrptfmB)/totthrptfmB
totrtmshcpdiffrdT = 100*(totrtmshcprdT - totthrptfmT)/totthrptfmT
# total throughput and diff constant loading, all algs, DTAG
totthrptgpnlD = np.genfromtxt(open("totthrptgpD.csv", "r"), delimiter=",", dtype =float)
totthrptfmnlD = np.genfromtxt(open("totthrptfmD.csv", "r"), delimiter=",", dtype =float)
totthrptrtmnlD = np.genfromtxt(open("totthrptrtmD.csv", "r"), delimiter=",", dtype =float)
totthrptgpnldiffD = np.genfromtxt(open("totthrptdiffgpD.csv", "r"), delimiter=",", dtype =float)
totthrptrtmnldiffD = np.genfromtxt(open("totthrptdiffrtmD.csv", "r"), delimiter=",", dtype =float)
# total throughput and diff constant loading, all algs, BT
totthrptgpnlB = np.genfromtxt(open("totthrptgpB.csv", "r"), delimiter=",", dtype =float)
totthrptfmnlB = np.genfromtxt(open("totthrptfmB.csv", "r"), delimiter=",", dtype =float)
totthrptrtmnlB = np.genfromtxt(open("totthrptrtmB.csv", "r"), delimiter=",", dtype =float)
totthrptgpnldiffB = np.genfromtxt(open("totthrptdiffgpB.csv", "r"), delimiter=",", dtype =float)
totthrptrtmnldiffB = np.genfromtxt(open("totthrptdiffrtmB.csv", "r"), delimiter=",", dtype =float)
# total throughput and diff constant loading, all algs, ES
totthrptgpnlT = np.genfromtxt(open("totthrptgpT.csv", "r"), delimiter=",", dtype =float)
totthrptfmnlT = np.genfromtxt(open("totthrptfmT.csv", "r"), delimiter=",", dtype =float)
totthrptrtmnlT = np.genfromtxt(open("totthrptrtmT.csv", "r"), delimiter=",", dtype =float)
totthrptgpnldiffT = np.genfromtxt(open("totthrptdiffgpT.csv", "r"), delimiter=",", dtype =float)
totthrptrtmnldiffT = np.genfromtxt(open("totthrptdiffrtmT.csv", "r"), delimiter=",", dtype =float)
# total throughput diff, random loading, all topologies
totthrptdiffgpD = ((totthrptgpD - totthrptfmD)/totthrptfmD)*100
totthrptdiffrtmD = ((totthrptrtmD - totthrptfmD)/totthrptfmD)*100
totthrptdiffshD = ((totgpshcpD - totthrptfmD)/totthrptfmD)*100

totthrptdiffgpB = ((totthrptgpB - totthrptfmB)/totthrptfmB)*100
totthrptdiffrtmB = ((totthrptrtmB - totthrptfmB)/totthrptfmB)*100
totthrptdiffshB = ((totgpshcpB - totthrptfmB)/totthrptfmB)*100

totthrptdiffgpT = ((totthrptgpT - totthrptfmT)/totthrptfmT)*100
totthrptdiffrtmT = ((totthrptrtmT - totthrptfmT)/totthrptfmT)*100
totthrptdiffshT = ((totgpshcpT - totthrptfmT)/totthrptfmT)*100

years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)

totthrptfmplB = totthrptfmnlB*np.ones([numyears,1])
totthrptfmplD = totthrptfmnlD*np.ones([numyears,1])
totthrptfmplT = totthrptfmnlT*np.ones([numyears,1])

# %% Shannon results for constant network loading 
font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ln1 = ax1.plot(years, totrtmshcpnoloadingB,'-', color = 'b',label = 'BT')
ln2 = ax1.plot(years, totrtmshcpnoloadingD,'-.', color = 'g',label = 'DTAG' )
ln3 = ax2.plot(years, totrtmshcpnoloadingdiffB,'--', color = 'b',label = 'BT gain')
ln4 = ax2.plot(years, totrtmshcpnoloadingdiffD,':', color = 'g',label = 'DTAG gain' )


ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
ax2.set_ylabel("total throughput gain (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptnoloadingSH.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% total throughput results for constant loading - separate plots 

# BT
fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()


ln1 = ax1.plot(years, totthrptgpnlB,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptrtmnlB,':', color = 'r',label = 'RTM' )
ln3 = ax1.plot(years, totthrptfmplB,'-', color = 'm',label = 'FM' )
 
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptnoloadingB.pdf', dpi=200,bbox_inches='tight')
plt.show()

# DTAG
fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
totthrptfmplB = totthrptfmnlB*np.ones([numyears,1])
totthrptfmplD = totthrptfmnlD*np.ones([numyears,1])

ln1 = ax1.plot(years, totthrptgpnlD,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptrtmnlD,':', color = 'r',label = 'RTM' )
ln3 = ax1.plot(years, totthrptfmplD,'-', color = 'm',label = 'FM' )
 
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptnoloadingD.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% total throughput diff results for constant loading - separate plots 

# BT
fig, ax1 = plt.subplots()
ln1 = ax1.plot(years, totthrptgpnldiffB,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptrtmnldiffB,':', color = 'r',label = 'RTM' )
 
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffnoloadingB.pdf', dpi=200,bbox_inches='tight')
plt.show()

# DTAG
fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totthrptgpnldiffD,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptrtmnldiffD,':', color = 'r',label = 'RTM' )
 
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffnoloadingD.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% combined results for constant network loading 

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

ln1 = ax1.plot(years, totthrptgpnldiffB,'--', color = 'b',label = 'GP BT')
ln2 = ax1.plot(years, totthrptrtmnldiffB,':', color = 'r',label = 'RTM BT' )
ln3 = ax1.plot(years, totthrptgpnldiffD,'--', color = 'g',label = 'GP DTAG')
ln4 = ax1.plot(years, totthrptrtmnldiffD,':', color = 'k',label = 'RTM DTAG' )
#ln3 = ax2.plot(years, totthrptdiffsh,'-', color = 'g',label = 'Sh.')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffnoloadingBandD.pdf', dpi=200,bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
totthrptfmplB = totthrptfmnlB*np.ones([numyears,1])
totthrptfmplD = totthrptfmnlD*np.ones([numyears,1])


ln1 = ax1.plot(years, totthrptgpnlB,'--', color = 'b',label = 'GP BT')
ln2 = ax1.plot(years, totthrptrtmnlB,':', color = 'r',label = 'RTM BT' )
ln3 = ax1.plot(years, totthrptfmplB,':', color = 'r',label = 'FM BT' )
ln4 = ax1.plot(years, totthrptgpnlD,'--', color = 'g',label = 'GP DTAG')
ln5 = ax1.plot(years, totthrptrtmnlD,':', color = 'k',label = 'RTM DTAG' )
ln6 = ax1.plot(years, totthrptfmplD,':', color = 'c',label = 'FM DTAG' )
 
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4+ln5+ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptnoloadingBandD.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% DTAG plotting - random network loading 

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#totthrptfmpl = totthrptfm*np.ones([numyears,1])
ln1 = ax1.plot(years, totthrptgpD,'--', color = 'b',label = 'GP')
#ln2 = ax1.plot(years, totthrptfmpl,'-', color = 'r',label = 'FM' )
ln4 = ax1.plot(years, totthrptrtmD,':', color = 'r',label = 'RTM')
ln2 = ax1.plot(years, totthrptfmD,'-', color = 'm',label = 'FM' )

    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
#ax2.set_ylabel("Shannon limit (Tb/s)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptrdaveD.pdf', dpi=200,bbox_inches='tight')
plt.show()



fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

ln1 = ax1.plot(years, totthrptdiffgpD,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptdiffrtmD,':', color = 'r',label = 'RTM' )
#ln3 = ax2.plot(years, totthrptdiffshD,'-', color = 'g',label = 'Sh.')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdaveD.pdf', dpi=200,bbox_inches='tight')
plt.show()
    

# %% BT plotting - random network loading  

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

#totthrptfmpl = totthrptfm*np.ones([numyears,1])

ln1 = ax1.plot(years, totthrptgpB,'--', color = 'b',label = 'GP')

#ln3 = ax2.plot(years, totgpshcpB,'-.', color = 'g',label = 'Sh.')

ln4 = ax1.plot(years, totthrptrtmB,':', color = 'r',label = 'RTM')
ln2 = ax1.plot(years, totthrptfmB,'-', color = 'm',label = 'FM' )
#ln5 = ax2.plot(years, totrtmshcp,'-.', color = 'g',label = 'Sh. RTM')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
#ax2.set_ylabel("Shannon limit (Tb/s)")
    
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([5,50])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptrdaveB.pdf', dpi=200,bbox_inches='tight')
plt.show()



fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

ln1 = ax1.plot(years, totthrptdiffgpB,'--', color = 'b',label = 'GP')
ln2 = ax1.plot(years, totthrptdiffrtmB,':', color = 'r',label = 'RTM' )
#ln3 = ax2.plot(years, totthrptdiffshB,'-', color = 'g',label = 'Sh.')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([0,14])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdaveB.pdf', dpi=200,bbox_inches='tight')
plt.show()
    
# %% Shannon results for increasing network loading 
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ln1 = ax1.plot(years, totrtmshcprdB,'-', color = 'b',label = 'BT total')
ln2 = ax1.plot(years, totrtmshcprdD,'-.', color = 'g',label = 'DTAG total' )
ln3 = ax2.plot(years, totrtmshcpdiffrdB,'--', color = 'b',label = 'BT gain')
ln4 = ax2.plot(years, totrtmshcpdiffrdD,':', color = 'g',label = 'DTAG gain' )


ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
ax2.set_ylabel("total throughput gain (%)")
    
ax1.set_xlim([years[0], years[-1]])
ax2.set_ylim([160,215])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2, loc=4, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptrdSH.pdf', dpi=200,bbox_inches='tight')
plt.show()



# %% total throughput random loading combined 

fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totthrptgpB,'--', color = 'b',label = 'GP BT')
ln2 = ax1.plot(years, totthrptrtmB,':', color = 'r',label = 'RTM BT' )
ln3 = ax1.plot(years, totthrptfmB,'-', color = 'm',label = 'FM BT' )

ln4 = ax1.plot(years, totthrptgpD,'--', color = 'g',label = 'GP DTAG')
ln5 = ax1.plot(years, totthrptrtmD,':', color = 'k',label = 'RTM DTAG')
ln6 = ax1.plot(years, totthrptfmD,'-', color = 'c',label = 'FM DTAG' )


ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")

    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([5,50])
    
lns = ln1+ln2+ln3+ln4+ln5+ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptrdBandD.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% total throughput diff random loading combined 

fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totthrptdiffgpB,'--', color = 'b',label = 'GP BT')
ln2 = ax1.plot(years, totthrptdiffrtmB,':', color = 'r',label = 'RTM BT' )

ln3 = ax1.plot(years, totthrptdiffgpD,'--', color = 'g',label = 'GP DTAG')
ln4 = ax1.plot(years, totthrptdiffrtmD,':', color = 'k',label = 'RTM DTAG')


ax1.set_xlabel("time (years)")

ax1.set_ylabel("total throughput gain (%)")
    
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([0,25])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdBandD.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% Shannon results for constant and increasing network loading 
fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totrtmshcpnoloadingB,'-', color = 'b',label = 'BT constant load')
ln2 = ax1.plot(years, totrtmshcpnoloadingD,'-.', color = 'g',label = 'DTAG constant load' )
ln3 = ax1.plot(years, totrtmshcprdB,'-', color = 'r',label = 'BT increasing load')
ln4 = ax1.plot(years, totrtmshcprdD,'-.', color = 'k',label = 'DTAG increasing load' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
 
ax1.set_xlim([years[0], years[-1]])
#ax2.set_ylim([160,215])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=1, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptconstandrandSH.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% Shannon diff results for constant and increasing network loading 
fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totrtmshcpnoloadingdiffB,'-', color = 'b',label = 'BT constant load')
ln2 = ax1.plot(years, totrtmshcpnoloadingdiffD,'-.', color = 'g',label = 'DTAG constant load' )
ln3 = ax1.plot(years, totthrptdiffshB,'-', color = 'r',label = 'BT increasing load')
ln4 = ax1.plot(years, totthrptdiffshD,'-.', color = 'k',label = 'DTAG increasing load' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
 
ax1.set_xlim([years[0], years[-1]])
#ax2.set_ylim([160,215])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=1, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffconstandrandSH.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% calculate the area under the total throughput curve

# constant loading 
Bareagp = sum([ (totthrptgpnlB[i] + totthrptgpnlB[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9
Bareartm = sum([ (totthrptrtmnlB[i] + totthrptrtmnlB[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9
Bareafm = totthrptfmplB[0]*315576000/1e9
Bareash = sum([ (totrtmshcpnoloadingB[i] + totrtmshcpnoloadingB[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9

Dareagp = sum([ (totthrptgpnlD[i] + totthrptgpnlD[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9
Dareartm = sum([ (totthrptrtmnlD[i] + totthrptrtmnlD[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9
Dareafm = totthrptfmplD[0]*315576000/1e9
Dareash = sum([ (totrtmshcpnoloadingD[i] + totrtmshcpnoloadingD[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9

Tareagp = sum([ (totthrptgpnlT[i] + totthrptgpnlT[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9
Tareartm = sum([ (totthrptrtmnlT[i] + totthrptrtmnlT[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9
Tareafm = totthrptfmplT[0]*315576000/1e9
Tareash = sum([ (totrtmshcpnoloadingT[i] + totrtmshcpnoloadingT[i+1])*15778800*0.5 for i in range(numyears-1)  ])/1e9

#totrtmshcprdB
#totrtmshcpnoloadingB

# increasing loading 
rdBareagp = sum([ (1.5*totthrptgpB[i+1] - 0.5*totthrptgpB[i])*15778800 for i in range(numyears-1)  ])/1e9
rdBareartm = sum([ (1.5*totthrptrtmB[i+1] - 0.5*totthrptrtmB[i])*15778800 for i in range(numyears-1)  ])/1e9
rdBareafm = sum([ (1.5*totthrptfmB[i+1] - 0.5*totthrptfmB[i])*15778800 for i in range(numyears-1)  ])/1e9
rdBareash = sum([ (1.5*totrtmshcprdB[i+1] - 0.5*totrtmshcprdB[i])*15778800 for i in range(numyears-1)  ])/1e9

rdDareagp = sum([ (1.5*totthrptgpD[i+1] - 0.5*totthrptgpD[i])*15778800 for i in range(numyears-1)  ])/1e9
rdDareartm = sum([ (1.5*totthrptrtmD[i+1] - 0.5*totthrptrtmD[i])*15778800 for i in range(numyears-1)  ])/1e9
rdDareafm = sum([ (1.5*totthrptfmD[i+1] - 0.5*totthrptfmD[i])*15778800 for i in range(numyears-1)  ])/1e9
rdDareash = sum([ (1.5*totrtmshcprdD[i+1] - 0.5*totrtmshcprdD[i])*15778800 for i in range(numyears-1)  ])/1e9

rdTareagp = sum([ (1.5*totthrptgpT[i+1] - 0.5*totthrptgpT[i])*15778800 for i in range(numyears-1)  ])/1e9
rdTareartm = sum([ (1.5*totthrptrtmT[i+1] - 0.5*totthrptrtmT[i])*15778800 for i in range(numyears-1)  ])/1e9
rdTareafm = sum([ (1.5*totthrptfmT[i+1] - 0.5*totthrptfmT[i])*15778800 for i in range(numyears-1)  ])/1e9
rdTareash = sum([ (1.5*totrtmshcprdT[i+1] - 0.5*totrtmshcprdT[i])*15778800 for i in range(numyears-1)  ])/1e9

Bgaingp  = 100*(Bareagp - Bareafm)/Bareafm
Bgainrtm  = 100*(Bareartm - Bareafm)/Bareafm
Bgainsh  = 100*(Bareash - Bareafm)/Bareafm

Dgaingp  = 100*(Dareagp - Dareafm)/Dareafm
Dgainrtm  = 100*(Dareartm - Dareafm)/Dareafm
Dgainsh  = 100*(Dareash - Dareafm)/Dareafm

Tgaingp  = 100*(Tareagp - Tareafm)/Tareafm
Tgainrtm  = 100*(Tareartm - Tareafm)/Tareafm
Tgainsh  = 100*(Tareash - Tareafm)/Tareafm

rdBgaingp = 100*(rdBareagp - rdBareafm)/rdBareafm
rdBgainrtm = 100*(rdBareartm - rdBareafm)/rdBareafm
rdBgainsh = 100*(rdBareash - rdBareafm)/rdBareafm
rdDgaingp = 100*(rdDareagp - rdDareafm)/rdDareafm
rdDgainrtm = 100*(rdDareartm - rdDareafm)/rdDareafm
rdDgainsh = 100*(rdDareash - rdDareafm)/rdDareafm
rdTgaingp = 100*(rdTareagp - rdTareafm)/rdTareafm
rdTgainrtm = 100*(rdTareartm - rdTareafm)/rdTareafm
rdTgainsh = 100*(rdTareash - rdTareafm)/rdTareafm



# %% Shannon results for constant network loading with graph T
font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

fig, ax1 = plt.subplots()


ln1 = ax1.plot(years, totrtmshcpnoloadingB,'-', color = 'b',label = 'BT')
ln2 = ax1.plot(years, totrtmshcpnoloadingD,'-.', color = 'g',label = 'DTAG' )
ln3 = ax1.plot(years, totrtmshcpnoloadingT,':', color = 'r',label = 'ES' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")

    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=3, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptnoloadingSHwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totrtmshcpnoloadingdiffB,'-', color = 'b',label = 'BT')
ln2 = ax1.plot(years, totrtmshcpnoloadingdiffD,'-.', color = 'g',label = 'DTAG' )
ln3 = ax1.plot(years, totrtmshcpnoloadingdiffT,':', color = 'r',label = 'ES' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=3, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffnoloadingSHwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()



# %% constant loading results with graph T 

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

ln1 = ax1.plot(years, totthrptgpnldiffB,':', color = 'b',label = 'GP BT')
ln4 = ax1.plot(years, totthrptrtmnldiffB,'--', color = 'b',label = 'RTM BT' )
ln2 = ax1.plot(years, totthrptgpnldiffD,':', color = 'r',label = 'GP DTAG')
ln5 = ax1.plot(years, totthrptrtmnldiffD,'--', color = 'r',label = 'RTM DTAG' )
ln3 = ax1.plot(years, totthrptgpnldiffT,':', color = 'g',label = 'GP ES')
ln6 = ax1.plot(years, totthrptrtmnldiffT,'--', color = 'g',label = 'RTM ES' )
#ln3 = ax2.plot(years, totthrptdiffsh,'-', color = 'g',label = 'Sh.')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4+ln5+ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffnoloadingwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()


ln1 = ax1.plot(years, totthrptgpnlB,':', color = 'b',label = 'GP BT')
ln2 = ax1.plot(years, totthrptrtmnlB,'--', color = 'r',label = 'RTM BT' )
ln3 = ax1.plot(years, totthrptfmplB,'-', color = 'm',label = 'FM BT' )
ln4 = ax1.plot(years, totthrptgpnlD,':', color = 'g',label = 'GP DTAG')
ln5 = ax1.plot(years, totthrptrtmnlD,'--', color = 'k',label = 'RTM DTAG' )
ln6 = ax1.plot(years, totthrptfmplD,'-', color = 'r',label = 'FM DTAG' )
ln7 = ax1.plot(years, totthrptgpnlT,':', color = 'r',label = 'GP ES')
ln8 = ax1.plot(years, totthrptrtmnlT,'--', color = 'g',label = 'RTM ES' )
ln9 = ax1.plot(years, totthrptfmplT,'-', color = 'c',label = 'FM ES' )


ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([35,65])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4+ln5+ln6+ln7+ln8+ln9
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=3, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptnoloadingwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% Shannon results for increasing network loading with graph T
font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

fig, ax1 = plt.subplots()


ln1 = ax1.plot(years, totrtmshcprdB,'-', color = 'b',label = 'BT')
ln2 = ax1.plot(years, totrtmshcprdD,'-.', color = 'g',label = 'DTAG' )
ln3 = ax1.plot(years, totrtmshcprdT,':', color = 'r',label = 'ES' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")

    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=3, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptrdSHwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totrtmshcpdiffrdB,'-', color = 'b',label = 'BT')
ln2 = ax1.plot(years, totrtmshcpdiffrdD,'-.', color = 'g',label = 'DTAG' )
ln3 = ax1.plot(years, totrtmshcpdiffrdT,':', color = 'r',label = 'ES' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([48,56.5])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=3, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdSHwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% Shannon results combined as in paper with graph T 

fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totrtmshcprdB,'--', color = 'b',label = 'BT increasing')
ln2 = ax1.plot(years, totrtmshcprdD,'--', color = 'g',label = 'DTAG increasing' )
ln3 = ax1.plot(years, totrtmshcprdT,'--', color = 'r',label = 'ES increasing' )
ln4 = ax1.plot(years, totrtmshcpnoloadingB,'-', color = 'b',label = 'BT constant')
ln5 = ax1.plot(years, totrtmshcpnoloadingD,'-', color = 'g',label = 'DTAG constant' )
ln6 = ax1.plot(years, totrtmshcpnoloadingT,'-', color = 'r',label = 'ES constant' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")

    
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([0,160])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4+ln5+ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2,loc=4, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptcombSHwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()


fig, ax1 = plt.subplots()

ln1 = ax1.plot(years, totrtmshcpdiffrdB,'--', color = 'b',label = 'BT increasing')
ln2 = ax1.plot(years, totrtmshcpdiffrdD,'--', color = 'g',label = 'DTAG increasing' )
ln3 = ax1.plot(years, totrtmshcpdiffrdT,'--', color = 'r',label = 'ES increasing' )
ln4 = ax1.plot(years, totrtmshcpnoloadingdiffB,'-', color = 'b',label = 'BT constant')
ln5 = ax1.plot(years, totrtmshcpnoloadingdiffD,'-', color = 'g',label = 'DTAG constant' )
ln6 = ax1.plot(years, totrtmshcpnoloadingdiffT,'-', color = 'r',label = 'ES constant' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")

    
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([174,210])
#ax2.set_ylim([128,140.5])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3+ln4+ln5+ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2,loc=1, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffcombSHwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% total throughput random loading combined with graph T 

fig, ax1 = plt.subplots()


ln1 = ax1.plot(years, totthrptgpB,'--', color = 'b',label = 'GP BT')
ln2 = ax1.plot(years, totthrptrtmB,':', color = 'r',label = 'RTM BT' )
ln3 = ax1.plot(years, totthrptfmB,'-', color = 'm',label = 'FM BT' )

ln4 = ax1.plot(years, totthrptgpD,'--', color = 'g',label = 'GP DTAG')
ln5 = ax1.plot(years, totthrptrtmD,':', color = 'k',label = 'RTM DTAG')
ln6 = ax1.plot(years, totthrptfmD,'-', color = 'c',label = 'FM DTAG' )

ln7 = ax1.plot(years, totthrptgpT,'--', color = 'g',label = 'GP ES')
ln8 = ax1.plot(years, totthrptrtmT,':', color = 'b',label = 'RTM ES')
ln9 = ax1.plot(years, totthrptfmT,'-', color = 'k',label = 'FM ES' )

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput (Tb/s)")

ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([5,50])
    
lns = ln1+ln2+ln3+ln4+ln5+ln6+ln7+ln8+ln9
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=3, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptrdBandDwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% total throughput diff random loading combined 

fig, ax1 = plt.subplots()
ln1 = ax1.plot(years, totthrptdiffgpB,':', color = 'b',label = 'GP BT')
ln4 = ax1.plot(years, totthrptdiffrtmB,'--', color = 'b',label = 'RTM BT' )
ln2 = ax1.plot(years, totthrptdiffgpD,':', color = 'r',label = 'GP DTAG')
ln5 = ax1.plot(years, totthrptdiffrtmD,'--', color = 'r',label = 'RTM DTAG')
ln3 = ax1.plot(years, totthrptdiffgpT,':', color = 'g',label = 'GP ES')
ln6 = ax1.plot(years, totthrptdiffrtmT,'--', color = 'g',label = 'RTM ES')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([0,25])
lns = ln1+ln2+ln3+ln4+ln5+ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdBandDwithT.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% total throughput diff, constant loading, GP and RTM split with graph T
fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

ln1 = ax1.plot(years, totthrptgpnldiffB,'--', color = 'b',label = 'GP BT')
ln2 = ax1.plot(years, totthrptgpnldiffD,'-.', color = 'r',label = 'GP DTAG')
ln3 = ax1.plot(years, totthrptgpnldiffT,':', color = 'g',label = 'GP ES')
#ln3 = ax2.plot(years, totthrptdiffsh,'-', color = 'g',label = 'Sh.')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([0,21])
#ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffnoloadingwithTGP.pdf', dpi=200,bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

ln1 = ax1.plot(years, totthrptrtmnldiffB,'--', color = 'b',label = 'RTM BT' )
ln2 = ax1.plot(years, totthrptrtmnldiffD,'-.', color = 'r',label = 'RTM DTAG' )
ln3 = ax1.plot(years, totthrptrtmnldiffT,':', color = 'g',label = 'RTM ES' )
#ln3 = ax2.plot(years, totthrptdiffsh,'-', color = 'g',label = 'Sh.')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
#ax2.set_ylabel("Shannon limit (%)")
    
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([0,21])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffnoloadingwithTRTM.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% total throughput diff random loading combined, split GP and RTM with graph T

fig, ax1 = plt.subplots()
ln1 = ax1.plot(years, totthrptdiffgpB,'--', color = 'b',label = 'GP BT')
ln2 = ax1.plot(years, totthrptdiffgpD,'-.', color = 'r',label = 'GP DTAG')
ln3 = ax1.plot(years, totthrptdiffgpT,':', color = 'g',label = 'GP ES')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([0,25])
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdBandDwithTGP.pdf', dpi=200,bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots()
ln1 = ax1.plot(years, totthrptdiffrtmB,'--', color = 'b',label = 'RTM BT' )
ln2 = ax1.plot(years, totthrptdiffrtmD,'-.', color = 'r',label = 'RTM DTAG')
ln3 = ax1.plot(years, totthrptdiffrtmT,':', color = 'g',label = 'RTM ES')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("total throughput gain (%)")
ax1.set_xlim([years[0], years[-1]])
ax1.set_ylim([0,25])
lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.savefig('totalthrptdiffrdBandDwithTRTM.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %%
