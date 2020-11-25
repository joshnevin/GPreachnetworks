# %% imports 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import time
import os
from numpy.linalg import cholesky
from numpy import transpose as T
from numpy.linalg import inv, det, solve
from numpy import matmul as mul, exp
from numpy.random import normal
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import norm
from scipy.special import erfcinv
import pandas as pd

def get_list_simulations(root_dir):
    """
    Get a list with all the name of the simulation csv files.
    """
    file_list_simulations = []
    for _, file in enumerate(os.listdir(root_dir)):
        file_list_simulations.append(file)
    return file_list_simulations

def get_dict_scenario_txt(file_list_simulations):
    """
    Build dictionary of scenario number and the corresponding name of the csv file
    with simulation data.
    """
    simulation_files_dict = {}
    for file_name in file_list_simulations:
        if file_name.endswith('.docx'):
            continue
        else:
            #simulation_files_dict[str(file_name.split("_")[1]) + '_' + str(file_name.split("_")[3].rstrip('.txt'))] = file_name
            simulation_files_dict[str(file_name.split("_")[1])] = file_name
    return simulation_files_dict

def down_sample(df, sam):
    """
    downsample to every hour and return Q and CD arrays 
    """
    sam = int(sam)
    Qarr = df["Q-factor"].to_numpy()
    CDarr = df["CD"].to_numpy()
    PMDarr = df["PMD"].to_numpy()
    Qarr = np.asarray([Qarr[i] for i in range(0, round(len(Qarr)), sam)])
    CDarr = np.asarray([CDarr[i] for i in range(0, round(len(CDarr)), sam)])
    PMDarr = np.asarray([PMDarr[i] for i in range(0, round(len(PMDarr)), sam)])
    return Qarr, CDarr, PMDarr


def drop_bad_values(df):
    """
    remove Q factor values for which CD is > 2 sigma from the mean 
    """
    CDarr = df["CD"]
    Qarr = df["Q-factor"]
    CDsd = np.std(CDarr)
    CDmean = np.mean(CDarr)
    Qsd = np.std(Qarr)
    Qmean = np.mean(Qarr)
    badinds = []
    for i in range(len(CDarr)):
        if abs(CDarr[i] - CDmean) > 6*CDsd or abs(Qarr[i] - Qmean) > 6*Qsd:
            badinds.append(i)
    try:
        df = df.drop(badinds, axis = 0)  # drop bad rows
    except:
        print("no bad values to remove!")
    return df

def read_txt_to_df(file, root_dir):
    """
    read individual channel txt file to a df
    """
    file_path = os.path.join(root_dir, file)
    df = pd.read_csv(file_path, sep='\t', header=[0,1])
    columns = ["date", "Q-factor", "Power", "CD", "PMD"] 
    df.columns = columns
    df = convert_objects_to_float(df)
    return df

def convert_objects_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns with object dtype to floats in order to use them in models.
    """
    indicator = df.dtypes == 'object'
    categorical_columns = df.columns[indicator].tolist()
    for col in categorical_columns:
        try:
            df[col] = df[col].astype('float')
        except:
            continue
    return df

def get_times_from_dates(df):
    """
    convert the dates column to time elapsed since the first measurement in minutes
                        """
    dates = df['date'].astype('str')
    numtime = len(dates[0].split('.'))
    timeelapsed = np.empty([len(dates),5])
    totaltime = []
    timeelapsed[0] = 0.0
    for i in range(len(dates) - 1):
        timeelapsed[i+1] = [float(dates[i+1].split('.')[j]) - float(dates[0].split('.')[j]) for j in range(numtime-1)  ]
        totaltime.append(sum([timeelapsed[i+1][0]*365.2425*24*60, timeelapsed[i+1][1]*30.44*24*60,
        timeelapsed[i+1][2]*24*60, timeelapsed[i+1][3]*60, timeelapsed[i+1][4]]))
    totaltime.insert(0, 0.0)
    return totaltime

def get_discontinuities_in_time(time_column):
    """
    get the indices for which the next 
                        """
    discons = []
    for i in range(len(timecol) - 1):
        if timecol[i+1] - timecol[i] > 15.0:
            discons.append(i+1)
    return discons

def get_contiguous_bactches(discons, Q):
    """
    split up Q into contiguous segments 
    """  
    discons = [0] + discons
    segs = []
    for i in range(len(discons) -1 ):
        segs.append(Qarr[discons[i]:discons[i+1]])
    return segs


def convert_to_lin_Q(Q):
    """
    convert from Q(dB) to Q
    """
    return 10**(Qarr/20)

def convert_to_lin(SNR):
    """
    convert from SNR(dB) to SNR
    """
    return 10**(SNR/10)

def ber_to_snr_QPSK(x):
    """
    convert from BER to SNR for QPSK signal
    """
    return ((erfcinv(2*x))**2)*2

def convert_to_db(x):
    """
    convert from lin to db
    """
    return 10*np.log10(x)

def sqexp(X,Y,k1,k2):
        X = np.atleast_2d(X)
        if Y is None:
            dists = pdist(X / k2, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
            # return gradient 
            K_gradient = (K * squareform(dists))[:, :, np.newaxis]
            #K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \  # anisotropic case, see https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/gaussian_process/kernels.py
            #            / (k2 ** 2)
            #K_gradient *= K[..., np.newaxis]
            return k1*K, K_gradient
        else:
            dists = cdist(X / k2, Y / k2,metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            return k1*K
font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 14}
matplotlib.rc('font', **font)

# %%  Javier data 
root_dir = '/Users/joshnevin/Desktop/JavierBERdata'
""" snr_file = "javiersnr.csv"
time_file = "javiersnrtime.csv" """
""" snr_file = "javierprefecBER.csv"
time_file = "javierprefecBERtime.csv" """
""" snr_file = "10acctestber.csv"
time_file = "10acctesttime.csv" """

snr_file = "javierprefecBERversion2.csv"
time_file = "timefx.csv"
snr_path = os.path.join(root_dir, snr_file)
time_path = os.path.join(root_dir, time_file)
snr = np.genfromtxt(open(snr_path, "r"), delimiter=",", dtype =float)

snr = ber_to_snr_QPSK(snr)

snr = convert_to_db(snr)

x = np.genfromtxt(open(time_path, "r"), delimiter=",", dtype =float)
#snr = convert_to_lin(snr)
xscale = 100
snr = snr[0:100]
#x = x[1100:1200]/xscale
x = x[0:100]/60
x = x.reshape(-1,1)
#x = x/60
#x = x/(60*xscale)

#x = x[0:100]/60
#downsampleval = 100
#snr = np.asarray([snr[i] for i in range(0, round(len(snr)), downsampleval)])  # downsample by factor of 10
#x = np.asarray([x[i] for i in range(0, round(len(x)), downsampleval)])/(60*xscale)   # downsample by factor of 10 """

plt.plot(x, snr, '*')
plt.show()

#%% Use GN model to generate modelled SNR values 
Ls = 100 
#TRxb2b = 14.6 # from Cambridge lab Wavelogic 3
alpha = 0.2
NLco = 1.27
Disp = 16.7
lam = 1550

OSNRmeasBW = 12.478 # OSNR measurement BW [GHz]
Rs = 32 # Symbol rate [Gbd]
Bchrs = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
PchdBm = np.linspace(-6,6,500)  # 500 datapoints for higher resolution of Pch

def SNR_GN(Ls, D, gam, lam, numlam, alpha, NF, numspans ):
           # rather than the worst-case number 
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
    Rs = 32 # symbol rate [GBaud]
    h = 6.63*1e-34  # Planck's constant [Js]
    allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
    beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
    Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
    Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
    NchRS = numlam
    Df = 50 # 50 GHz grid 
    BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
    # ===================== find Popt for one span ==========================
    numpch = len(PchdBm)
    Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
    Gwdmsw = Pchsw/(BchRS*1e9)
    Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
    G = alpha*Ls
    NFl = 10**(NF/10) 
    Gl = 10**(G/10) 
    Pasesw = NFl*h*f*(Gl - 1)*BchRS*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
    snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*BchRS*1e9)
    Popt = PchdBm[np.argmax(snrsw)]  
    # ===================== find SNR ==========================
    Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
    Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans                                                                             
    Pase = NF*h*f*(convert_to_lin(alpha*Ls) - 1)*BchRS*1e9*numspans       
    Pch = 1e-3*10**(Popt/10) 
    snr = (Pch/(Pase + Gnli*BchRS*1e9))
    #snr = ( snr**(-1) + (convert_to_lin(TRxb2b))**(-1) )**(-1)
    return convert_to_db(snr)

snr_gn = SNR_GN(Ls, Disp, NLco, lam, 1, 0.2, 4.5, 10)

# %%

def ystar_diff(params, xst, x, y, y_GN, sig):
    [k1, k2] = params
    num_x = len(x)
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    K = sqexp(x,None,k1,k2**0.5)[0]
    alpha =  np.matmul(np.linalg.inv(K + (sig**2)*np.identity(num_x)), y).reshape(num_x,1)
    #testalphacalc = alphacalc(x, x, snr, 1.0, 1.0, 1.0)
    Kst = sqexp(xst, x, k1, k2)
    ystar = np.matmul(Kst, alpha)
    return sum((ystar - y_GN)**2)

def ystar(k1, k2, sig, x, y):
    num_x = len(x)
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    K = sqexp(x,None,k1,k2**0.5)[0]
    alpha =  np.matmul(np.linalg.inv(K + (sig**2)*np.identity(num_x)), y).reshape(num_x,1)
    #testalphacalc = alphacalc(x, x, snr, 1.0, 1.0, 1.0)
    Kst = sqexp(x, x, k1, k2)
    return np.matmul(Kst, alpha)

k1_i = 1.0 # initial guesses for hyperparameters 
k2_i = 1.0
sig_i = np.std(snr)
h1low = 1e-3
h1high = 1e3
h2low = 1e-3
h2high = 1e3

k_i = np.ndarray((2,), buffer=np.array([k1_i,k2_i]), dtype = float) # initial guesses
phys_model_hyps = minimize(ystar_diff,k_i,args=(x, x, snr, snr_gn, sig_i),method = 'Nelder-Mead',bounds = ((h1low,h1high),(h2low,h2high)),options={'maxiter':1e3})
k1_o, k2_o = phys_model_hyps.x  # GN model-defined hyperpriors

total_iters = phys_model_hyps.nit
print("K1 opt initial: " +  str(k1_o))
print("K2 opt initial: " +  str(k2_o))

# %% standard GP for hyperprior comparison
def GP_train_kernel(x,y,kernel):
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 20, normalize_y=False, alpha=np.var(y))
    gpr.fit(x, y)
    print("Optimised kernel: %s" % gpr.kernel_)
    ystar, sigma = gpr.predict(x, return_std=True )
    theta = gpr.kernel_.theta
    #lml = gpr.log_marginal_likelihood(theta=theta)
    lml = gpr.log_marginal_likelihood()
    sigma = np.reshape(sigma,(np.size(sigma), 1)) 
    sigma = (sigma**2 + 1)**0.5  
    ystarp = ystar + sigma
    ystari = scaler.inverse_transform(ystar)
    ystarpi = scaler.inverse_transform(ystarp)
    sigmai = (ystarpi - ystari)
    return ystari, sigmai, gpr, theta, lml

k1_iwu = np.random.uniform(1e-1, 1e1)
k2_iwu = np.random.uniform(1e-1, 1e1)
kernel_standard_gp = C(k1_iwu, (1e-1, 1e1)) * RBF(k2_iwu, (1e-1,1e1))
start = time.time()
ypred_sk_weak, sigma_sk_weak, _, theta, lml = GP_train_kernel(x, snr, kernel_standard_gp)
end = time.time()
#np.savetxt('ypred_sk_weak.csv', ypred_sk, delimiter=',')
#np.savetxt('sigma_sk_weak.csv', sigma_sk, delimiter=',')
print("weak prior")
print("GP fitting took " + str((end-start)/60) + " minutes")
print("LML = " + str(lml))
thetatrans = np.exp(theta)
print("theta (linear) = " + str(thetatrans))
print("theta (log) = " + str(theta))
print("sigma mean = " + str(np.mean(sigma_sk_weak)))

kernel_standard_gp = C(k1_o, (k1_o/1000, k1_o*1000)) * RBF(k2_o, (k2_o/1000, k2_o*1000))
#kernel_standard_gp = C(k1_o, (1e-5, 1e5)) * RBF(k2_o, (1e-5, 1e5))
start = time.time()
ypred_sk_gn, sigma_sk_gn, _, theta, lml = GP_train_kernel(x, snr, kernel_standard_gp)
end = time.time()
""" np.savetxt('ypred_sk_gn.csv', ypred_sk, delimiter=',')
np.savetxt('sigma_sk_gn.csv', sigma_sk, delimiter=',') """
print("GN prior")
print("GP fitting took " + str((end-start)/60) + " minutes")
print("LML = " + str(lml))
thetatrans = np.exp(theta)
print("theta (linear) = " + str(thetatrans))
print("theta (log) = " + str(theta))
print("sigma mean = " + str(np.mean(sigma_sk_gn)))

# %% plot the initial model from physical priors 
""" ypred_sk_gn = np.genfromtxt(open("ypred_sk_gn.csv", "r"), delimiter=",", dtype =float)
sigma_sk_gn = np.genfromtxt(open("sigma_sk_gn.csv", "r"), delimiter=",", dtype =float)
ypred_sk_weak = np.genfromtxt(open("ypred_sk_weak.csv", "r"), delimiter=",", dtype =float)
sigma_sk_weak = np.genfromtxt(open("sigma_sk_weak.csv", "r"), delimiter=",", dtype =float) """

ypred_gn = ystar(k1_o, k2_o, sig_i, x, snr) # prediction based on physical model priors alone
prmnp_gn = ypred_gn + 3*sig_i
prmnm_gn = ypred_gn - 3*sig_i
prmnp_sk_gn = ypred_sk_gn + 3*sigma_sk_gn
prmnm_sk_gn = ypred_sk_gn - 3*sigma_sk_gn
prmnp_sk_weak = ypred_sk_weak + 3*sigma_sk_weak
prmnm_sk_weak = ypred_sk_weak - 3*sigma_sk_weak

""" fig, ax = plt.subplots()
#ax.plot(x, snr, '+', color = 'k')
ax.plot(x, ypred_gn, '--', color = 'k')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp_gn,
                        (prmnm_gn)[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
#plt.legend(ncol = 3)
#plt.title("Channel " + str(channel))
ax.set_ylabel("SNR")
ax.set_xlabel("Time ")
ax.set_xlim([x[0], x[-1]])
#ax.plot(xgp, Qgp, label = "GP pred. mean", color = 'r')
#plt.savefig('javierGPfitsnrnew100.pdf', dpi=200,bbox_inches='tight')
plt.show() """

fig, ax = plt.subplots()
ax.plot(x, snr, '+', color = 'k')
ax.plot(x, ypred_sk_gn, '-', color = 'r', label = 'GN prior')
""" ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp_sk_gn,
                        (prmnm_sk_gn)[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$') """
ax.plot(x, ypred_sk_weak, '-' ,color = 'b', label = 'weak prior')
""" ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp_sk_weak,
                        (prmnm_sk_weak)[::-1]]),
         alpha=0.3, fc='g', ec='None', label='$3 \sigma$') """
plt.legend()
#plt.title("Channel " + str(channel))
ax.set_ylabel("SNR")
ax.set_xlabel("Time ")
ax.set_xlim([x[0], x[-1]])
#ax.plot(xgp, Qgp, label = "GP pred. mean", color = 'r')
plt.savefig('GPphyspriorcompprmn.pdf', dpi=200,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.plot(x, sigma_sk_gn, '-', color = 'r', label = 'GN prior')
""" ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp_sk_gn,
                        (prmnm_sk_gn)[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$') """
ax.plot(x, sigma_sk_weak, '-' ,color = 'b', label = 'weak prior')
""" ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp_sk_weak,
                        (prmnm_sk_weak)[::-1]]),
         alpha=0.3, fc='g', ec='None', label='$3 \sigma$') """
plt.legend()
#plt.title("Channel " + str(channel))
ax.set_ylabel("SNR $\sigma$")
ax.set_xlabel("Time ")
ax.set_xlim([x[0], x[-1]])
#ax.plot(xgp, Qgp, label = "GP pred. mean", color = 'r')
#plt.savefig('javierGPfitsnrnew100.pdf', dpi=200,bbox_inches='tight')
plt.savefig('GPphyspriorcompsig.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% this is what David has done...

y = snr.reshape(-1, 1)
h1low = 1e-1
h1high = 1e1
h2low = 1e-1
h2high = 1e1
h1_ini = 1.0
h2_ini = 1.0
n = len(x)
# David is optimising w.r.t. the LML of the physical model predictions as well as the data..
# This will also change the gradient of the LML function and the hyperparameter optimisation function
def lml_phys(params,y,sig,y_gn):
    #print(params)  # show progress of fit
    [k1, k2] = params
    global Kfh
    Kfh = sqexp(x,None,k1,k2**0.5)[0]
    #print(np.size(Kfh))
    Ky = Kfh + (sig**2)*np.identity(len(x)) # calculate initial kernel with noise
    global Kyinvh
    Kyinvh = inv(Ky)
    return -(-0.5*mul(mul(T(y),Kyinvh), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) + -(-0.5*mul(mul(T(y_gn),Kyinvh), y_gn) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)
    #return -(-0.5*mul(mul(T(y),Kyinvh), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi))

def lmlg_phys(params,y,sig,y_gn):
        k1, k2 = params
        al = mul(Kyinvh,y)
        al_gn = mul(Kyinvh,y_gn)
        dKdk1 = Kfh*(1/k1)
        dKdk2 = sqexp(x,None,k1,k2**0.5)[1].reshape(n,n)
        lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk1))) + -(0.5*np.trace(mul(mul(al_gn,T(al_gn)) - Kyinvh, dKdk1)))
        lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk2))) + -(0.5*np.trace(mul(mul(al_gn,T(al_gn)) - Kyinvh, dKdk2)))
        #lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk1)))
        #lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk2)))
        return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)

def hypopt_phys(y, numrestarts, sig, y_gn):  # this function is used to obtain optimised hyperparameters by minimising the LML
        numh = 2 # number of hyperparameters in kernel function 
        k1s4 = np.empty([numrestarts,1])
        k2s4 = np.empty([numrestarts,1])
        for i in range(numrestarts):    
            k1is4 = np.random.uniform(h1low,h1high)
            k2is4 = np.random.uniform(h2low,h2high)
            #k1is4 = h1_ini
            #k2is4 = h2_ini
            kis4 = np.ndarray((numh,), buffer=np.array([k1is4,k2is4]), dtype = float)
            s4res = minimize(lml_phys,kis4,args=(y,sig,y_gn),method = 'L-BFGS-B',jac=lmlg_phys,bounds = ((h1low,h1high),(h2low,h2high)),options={'maxiter':1e3})
            step4res = []
            if s4res.success:
                step4res.append(s4res.x)
                print("successful k1:" + str(s4res.x[0]))
                print("successful k2: " + str(s4res.x[1]))
            else:
                print("error " + str(k1is4))
                print("error " + str(k2is4))
                #raise ValueError(s4res.message)
                #k1is4 = np.random.uniform(1e-2,1e3)
                #k2is4 = np.random.uniform(2e-1,1e3)
                k1is4 = np.random.uniform(h1low,h1high)
                k2is4 = np.random.uniform(h2low,h2high)
                print("error in hypopth() - reinitialising hyperparameters")
                continue 
            k1s4[i] = step4res[0][0]
            k2s4[i] = step4res[0][1]
            #print("successful k1 test:" + str(k1s4[i]))
            #print("successful k2 test:" + str(k2s4[i]))
        lmltest = [lml_phys([k1s4[i],k2s4[i]],y,sig,y_gn) for i in range(numrestarts)]
        k1f = k1s4[np.argmin(lmltest)]
        k2f = k2s4[np.argmin(lmltest)]
        print(lmltest)
            #lml(params,y,sig)
        return k1f, k2f

def GPRfit_phys(xs,k1,k2,sig):  # algorithm 2.1 from R+W, fitted hyperparams go in here
        Ky = sqexp(x,None,k1,k2**0.5)[0] + (sig**2)*np.identity(n)
        Ks = sqexp(xs, x, k1, k2**0.5)
        Kss = sqexp(xs, None, k1, k2)[0]
        L = cholesky(Ky)
        al = solve(T(L), solve(L,y)) 
        fmst = mul(Ks,al) 
        varfmst = np.empty([n,1])
        for i in range(np.size(xs)):
            v = solve(L,T(Ks[:,i]))
            varfmst[i] = Kss[i,i] - mul(T(v),v)  
        lmlopt = -0.5*mul(T(y),al) - np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
        #return fmst, varfmst[::-1], lmlopt
        return fmst, varfmst, lmlopt

#snr_gn_ones = 14.15*np.ones(len(y)).reshape(-1,1)
snr_gn_ones = 14.15*np.random.normal(14.15, np.std(snr), len(x)).reshape(-1,1)
scaler_gn = StandardScaler().fit(snr_gn_ones)
snr_gn_ones = scaler_gn.transform(snr_gn_ones)
#snr_gn_gauss = np.random.normal(14.15, np.std())
scaler = StandardScaler().fit(y)
y = scaler.transform(y)

k1_opt, k2_opt = hypopt_phys(y, 20, np.std(y), snr_gn_ones)
fmst, varfmst, lmlopt = GPRfit_phys(x,k1_opt,k2_opt,np.std(y))

print("K1 = " + str(k1_opt))
print("K2 = " + str(k2_opt))

sigma = varfmst**0.5
sigma = (sigma**2 + 1)**0.5 
fmstp = fmst + sigma
fmst = scaler.inverse_transform(fmst)
fmstp = scaler.inverse_transform(fmstp)
sig_learned = fmstp - fmst

# %%

prmnp = fmst + 3*sig_learned
prmnm = fmst - 3*sig_learned

fig, ax = plt.subplots()
ax.plot(x, snr, '+', color = 'k')
ax.plot(x, fmst, '-', color = 'r', label = 'fmst')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp,
                        (prmnm)[::-1]]),
         alpha=0.3, fc='g', ec='None', label='$3 \sigma$')
plt.legend()
#plt.title("Channel " + str(channel))
ax.set_ylabel("SNR")
ax.set_xlabel("Time ")
ax.set_xlim([x[0], x[-1]])
#ax.plot(xgp, Qgp, label = "GP pred. mean", color = 'r')
plt.savefig('GPtestphysmodel.pdf', dpi=200,bbox_inches='tight')
plt.show()



# %%
fig, ax = plt.subplots()
ax.plot(x, snr, '+', color = 'k')
""" ax.plot(x, ypred_sk_gn, '-', color = 'r', label = 'GN prior')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp_sk_gn,
                        (prmnm_sk_gn)[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$') """
ax.plot(x, ypred_sk_weak, '-' ,color = 'b', label = 'weak prior')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp_sk_weak,
                        (prmnm_sk_weak)[::-1]]),
         alpha=0.3, fc='g', ec='None', label='$3 \sigma$')
plt.legend()
#plt.title("Channel " + str(channel))
ax.set_ylabel("SNR")
ax.set_xlabel("Time ")
ax.set_xlim([x[0], x[-1]])
#ax.plot(xgp, Qgp, label = "GP pred. mean", color = 'r')
plt.savefig('GPtestsk.pdf', dpi=200,bbox_inches='tight')
plt.show()
# %%
