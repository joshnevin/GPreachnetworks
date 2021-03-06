# physical model-enhanced priors version of HGP algorithm
# Implementation of most likely heteroscedastic GP regression algroithm:
# K. Kersting, C. Plagemann, P. Pfaff, and W. Burgard,
# “Most likely heteroscedastic gaussian process regression,” 
# inProc. 24th InternationalConference on Machine Learning,(Oregon, USA, 2007), pp. 393–400.

# %%  imports
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
#import multiprocessing
#from GHquad import GHquad
#import matplotlib
import math
#matplotlib.rc_file_defaults()   # use to return to Matplotlib defaults 
import pandas as pd
import cProfile
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


# %% MS data

channel = '345'
root_dir = '/Users/joshnevin/Desktop/MicrosoftDataset'
file_list_simulations= get_list_simulations(root_dir)
simulation_files_dict = get_dict_scenario_txt(file_list_simulations)

df = read_txt_to_df(simulation_files_dict[channel], root_dir)
timecol = get_times_from_dates(df)
df = df.drop('date', axis =1)
df['time'] = timecol
df = drop_bad_values(df)
timecol = df['time'].to_numpy()
discons = get_discontinuities_in_time(timecol)

Qarr, _, _ = down_sample(df, 1) # set sam = 1 to just return arrays with same sampling
#Qarr = convert_to_lin_Q(Qarr)

batches = get_contiguous_bactches(discons, Qarr)
discons = [0] + discons
batchlens = ([len(i) for i in batches ] )
batch_ind = 1 # note: first index is now 0 in batches list 
Qarr = batches[batch_ind]
print("batch size " + str(len(Qarr)))
downsampleval = 20

#snr = np.asarray([Qarr[i] for i in range(0, round(len(Qarr)), downsampleval)])  # downsample by factor of 10
snr = Qarr[300:400]
snr = snr.reshape(len(snr),1)
timecol = timecol[discons[batch_ind]:discons[batch_ind+1]]
#x = np.asarray([timecol[i]/(24*60) for i in range(0, round(len(timecol)), downsampleval)])
xscale = 1
#x = np.asarray([timecol[i]/(24*60) for i in range(0, round(len(timecol)), downsampleval)])/xscale
x = timecol[300:400]/xscale
x = x.reshape(len(x), 1)
numpoints = len(snr)
numedges = 1

plt.plot(x, snr, '*')
plt.show()
plot_title = "MSdata345bt1hgp100"
#plot_title = "MSchannel80batch13"
np.savetxt('snr' + str(plot_title) + '.csv', snr, delimiter=',')

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
snr = snr[:100]
#snr = snr[1100:1200]
#x = x[1100:1200]/xscale
#x = x[0:400]/xscale
#x = x/60
#x = x/(60*xscale)

x = x[0:100]/60
downsampleval = 100
#snr = np.asarray([snr[i] for i in range(0, round(len(snr)), downsampleval)])  # downsample by factor of 10
#x = np.asarray([x[i] for i in range(0, round(len(x)), downsampleval)])/(60*xscale)   # downsample by factor of 10 """

plt.plot(x, snr, '*')
plt.show()

#y_gn = 14.15*np.ones(len(snr))
y_gn = np.random.normal(14.15, np.std(snr), len(snr))
#y_gn = np.random.normal(0, 1.0, len(snr))
# %% standard GP for hyperparameter estimation
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
kernel_standard_gp = C(1, (1e-5, 1e5)) * RBF(1, (1e-5, 1e5))

start = time.time()
_, sigma_i, _, theta, lml = GP_train_kernel(x, snr, kernel_standard_gp)
end = time.time()
print("GP fitting took " + str((end-start)/60) + " minutes")
print("LML = " + str(lml))
thetatrans = np.exp(theta)
print("theta (linear) = " + str(thetatrans))
print("theta (log) = " + str(theta))
print("sigma mean = " + str(np.mean(sigma_i)))

# %%

#snr = np.genfromtxt(open("hetdata20.csv", "r"), delimiter=",", dtype =float) # run heteroscedastic datagen section from GPreachringsrand.py 
#snr = np.genfromtxt(open("hetdataextdegdv.csv", "r"), delimiter=",", dtype =float) # run heteroscedastic datagen section from GPreachringsrand.py 
#snr = np.genfromtxt(open("hetdataextdegst.csv", "r"), delimiter=",", dtype =float) # run heteroscedastic datagen section from GPreachringsrand.py 
#numpoints = np.size(snr,1)
#numedges = np.size(snr,0)
#x = np.linspace(0,numpoints-1,numpoints)
#x = np.linspace(0,1,numpoints)


#SNR = SNR[0]
#snr = snr[0:1]
# # data used by Goldberg - use for testing 
# =============================================================================
""" numpoints = 100
numedges = 1
x = np.linspace(0,1,numpoints).reshape(numpoints,numedges)
sd = np.linspace(0.5,1.5,numpoints)
y = np.zeros(numpoints)
n = np.size(x)
for i in range(numpoints):
    y[i] = 2*np.sin(2*np.pi*x[i]) + np.random.normal(0, sd[i])
snr = y.reshape(numpoints,numedges) """
    # =============================================================================
    
    # data used by Yuan and Wahba - use for testing 
    # =============================================================================
""" numpoints = 200
x = np.linspace(0,1,numpoints)  
ymean = 2*(exp(-30*(x-0.25)**2) + np.sin(np.pi*x**2)) - 2
sd = exp(np.sin(2*np.pi*x))
y = np.random.normal(ymean, sd)
snr = y """
    # =============================================================================
    
    # data used by Williams - use for testing 
""" numpoints = 200
x = np.linspace(0,np.pi,numpoints)
wmean = np.sin(2.5*x)*np.sin(1.5*x)
sd = 0.01 + 0.25*(1 - np.sin(2.5*x))**2
y = np.random.normal(wmean, sd)
snr = y """

def HGPfunc(x,y,y_gn,plot, h1low, h1high, h2low, h2high, h1low_z, h1high_z, h2low_z, h2high_z):
    y = y.reshape(-1,1)
    y_gn = y_gn.reshape(-1,1)
    x = x.reshape(-1,1)
    if plot:
        plt.plot(x,y,'+')
        plt.xlabel("Pch (dBm)")
        plt.ylabel("SNR (dB)")
        plt.savefig('Adataset.png', dpi=200)
        plt.show()
    n = np.size(x)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    #scaler_gn = StandardScaler().fit(y_gn)
    #y_gn = scaler_gn.transform(y_gn)
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
    # heteroscedastic versions of functions 
    global Kyinvh
    Kyinvh = 0.0
    global Kfh
    Kfh =  0.0 
    def lmlh(params,y,R,y_gn):
        #print(params)  # show progress of fit
        [k1, k2] = params
        global Kfh
        Kfh = sqexp(x,None,k1,k2**0.5)[0]
        #print(np.size(Kfh))
        Ky = Kfh + R # calculate initial kernel with noise
        global Kyinvh
        Kyinvh = inv(Ky)
        return -(-0.5*mul(mul(T(y),Kyinvh), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) + -(-0.5*mul(mul(T(y_gn),Kyinvh), y_gn) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)
    def lmlgh(params,y,R,y_gn):
        k1, k2 = params
        al = mul(Kyinvh,y)
        al_gn = mul(Kyinvh,y_gn)
        dKdk1 = Kfh*(1/k1)
        dKdk2 = sqexp(x,None,k1,k2**0.5)[1].reshape(n,n)
        lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk1))) + -(0.5*np.trace(mul(mul(al_gn,T(al_gn)) - Kyinvh, dKdk1)))
        lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk2))) + -(0.5*np.trace(mul(mul(al_gn,T(al_gn)) - Kyinvh, dKdk2)))
        return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)
    def GPRfith(xs,k1,k2,R,Rs):
        Ky = sqexp(x,None,k1,k2**0.5)[0] + R
        Ks = sqexp(xs, x, k1, k2**0.5)
        Kss = sqexp(xs, None, k1, k2)[0]
        L = cholesky(Ky)
        al = solve(T(L), solve(L,y))
        fmst = mul(Ks,al)
        varfmst = np.empty([n,1])
        for i in range(np.size(xs)):
            v = solve(L,T(Ks[:,i]))
            varfmst[i] = Kss[i,i] - mul(T(v),v)  + Rs[i,i]
        lmlopt = -0.5*mul(T(y),al) - np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
        #return fmst, varfmst[::-1], lmlopt
        return fmst, varfmst, lmlopt
    def hypopth(y, numrestarts, R, y_gn):
        numh = 2 # number of hyperparameters in kernel function 
        k1s4 = np.empty([numrestarts,1])
        k2s4 = np.empty([numrestarts,1])
        for i in range(numrestarts):    
            #k1is4 = np.random.uniform(1e-2,1e3)
            #k2is4 = np.random.uniform(1e-1,1e3)
            k1is4 = np.random.uniform(h1low,h1high)
            k2is4 = np.random.uniform(h2low,h2high)
            kis4 = np.ndarray((numh,), buffer=np.array([k1is4,k2is4]), dtype = float)
            s4res = minimize(lmlh,kis4,args=(y,R,y_gn),method = 'L-BFGS-B',jac=lmlgh,bounds = ((h1low,h1high),(h2low,h2high)),options={'maxiter':1e2})
            step4res = []
            if s4res.success:
                step4res.append(s4res.x)
                print("successful k1:" + str(k1is4))
                print("successful k2: " + str(k2is4))
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
        lmltest = [lmlh([k1s4[i],k2s4[i]],y,R,y_gn) for i in range(numrestarts)]
        #k1f = k1s4[np.argmin(lmltest)]
        #k2f = k2s4[np.argmin(lmltest)]
        k1f = k1s4[np.argmax(lmltest)]
        k2f = k2s4[np.argmax(lmltest)]
            #lml(params,y,sig)
        return k1f, k2f
    def hetloopSK(fmst,varfmst,numiters,numrestarts):
        s = 200
        #k1is3, k2is3, k1is4,k2is4  =  np.random.uniform(1e-2,1e2,4)
        MSE = np.empty([numiters,1])
        NLPD = np.empty([numiters,1])
        fmstf = np.empty([numiters,n])
        varfmstf = np.empty([numiters,n])
        lmloptf = np.empty([numiters,1])
        rf = np.empty([numiters,n])
        i = 0
        while i < numiters:        
            
            breakwhile = False
            # Step 2: estimate empirical noise levels z 
            #k1is4,k2is4  = np.random.uniform(1e-2,1e2,2)
            #k1is3, k1is4  =  np.random.uniform(1e-2,1e2,2)
            #k2is3, k2is4  =  np.random.uniform(1e-1,1e2,2)
            k1is3  =  np.random.uniform(h1low_z,h1high_z,1)
            k2is3  =  np.random.uniform(h2low_z,h2high_z,1)
            z = np.empty([n,1])
            for j in range(n):
                #np.random.seed()
                normdraw = normal(fmst[j], varfmst[j]**0.5, s).reshape(s,1)
                z[j] = np.log((1/s)*0.5*sum((y[j] - normdraw)**2))
                if math.isnan(z[j]): # True for NaN values
                    breakwhile = True
                    break
            if breakwhile:
                print("Nan value in z -- skipping iter "+ str(i))
                i = i + 1
                continue
            #  Step 3: estimate GP2 on D' - (x,z)
            kernel2 = C(k1is3, (h1low_z,h1high_z)) * RBF(k2is3, (h2low_z,h2high_z)) 
            gpr2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer = numrestarts, normalize_y=False, alpha=np.var(z))
            
            gpr2.fit(x, z)
            ystar2, sigma2 = gpr2.predict(x, return_std=True )
            sigma2 = (sigma2**2 + 1)**0.5
        # Step 4: train heteroscedastic GP3 using predictive mean of G2 to predict log noise levels r
            r = exp(ystar2)
            R = r*np.identity(n)
            k1s4, k2s4 = hypopth(y,numrestarts,R, y_gn)  # needs to be modified 
            fmst4, varfmst4, lmlopt4 = GPRfith(x,k1s4,k2s4,R,R)  # needs to be modified 
            # test for convergence 
            MSE[i] = (1/n)*sum(((y-fmst4)**2)/np.var(y))
            #NLPD[i] = sum([(1/n)*(-np.log(norm.pdf(x[j], fmst4[j], varfmst4[j]**0.5))) for j in range(n) ])
            nlpdarg = np.zeros([n,1])
            #nlpdtest = np.zeros([n,1])
            for k in range(n):
                nlpdarg[k] = -np.log10(norm.pdf(x[k], fmst4[k], varfmst4[k]**0.5))
                #nlpdtest[k] = norm.pdf(x[k], fmst4[k], varfmst4[k]**0.5)
            #print("mean NLPD log arg " + str(nlpdtest) )
                #test3[k] = -np.log(norm.pdf(x[k], fmst[k], varfmst[k]**0.5))
            NLPD[i] = sum(nlpdarg)*(1/n)
            print("MSE = " + str(MSE[i]))
            print("NLPD = " + str(NLPD[i]))
            print("finished iteration " + str(i+1))
            fmstf[i,:] = fmst4.reshape(n)
            varfmstf[i,:] = varfmst4.reshape(n)
            lmloptf[i] = lmlopt4
            fmst = fmst4
            varfmst = varfmst4
            rf[i,:] = r.reshape(n)
            #k1is3 = k1s4
            #k2is3 = k2s4
            i = i + 1
        return fmstf,varfmstf, lmloptf, MSE, rf, NLPD #  , NLPD 
    
    numiters = 10
    numrestarts = 20
    
    #kernel1 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
    #gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer = 0, normalize_y=True)
    kernel1 = C(1.0, (h1low,h1high)) * RBF(1.0, (h2low,h2high)) 
    gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer = numrestarts, normalize_y=False, alpha=np.var(y))
    gpr1.fit(x, y)
    ystar1, sigma1 = gpr1.predict(x, return_std=True )
    var1 = (sigma1**2 + np.var(y))
    #sigma1 = np.reshape(sigma1,(np.size(sigma1), 1))

    
    start_time = time.time()
    fmstf,varfmstf, lmlopt, mse, _,NLPD = hetloopSK(ystar1,var1,numiters,numrestarts)
    duration = time.time() - start_time

    ind = numiters - 1
    #ind = 
    fmst4 = fmstf[ind]
    varfmst4 = varfmstf[ind]
    
    sigs4 = varfmst4**0.5
    fmstps4 = fmst4 + sigs4
    fmst4i = scaler.inverse_transform(fmst4)
    fmstps4i = scaler.inverse_transform(fmstps4)
    

      
    print("HGP fitting duration: " + str(duration)) 
    
    return fmst4i, fmstps4i, lmlopt, mse, NLPD



""" prmn = np.empty([numedges,numpoints])
prmnp = np.empty([numedges,numpoints])
lml = np.empty([numedges,numiters])
MSE = np.empty([numedges,numiters])
NLPD = np.empty([numedges,numiters]) """

""" for i in range(np.size(snr,0)):
    prmn[i], prmnp[i], lmls, MSEs, NLPDs = HGPfunc(x,snr[i],False)
    lml[i] = lmls.reshape(numiters)
    MSE[i] = MSEs.reshape(numiters)
    NLPD[i] = NLPDs.reshape(numiters) """


h1low = 1e-1
h1high = 1e1
#h2low = 1e-1
#h2high = 20
h2low = 1e-2
h2high = 1e1

h1low_z = 1e-2
h1high_z = 1e2
h2low_z = 1e-2
h2high_z = 1e2

prmn, prmnp, lml, MSE, NLPD = HGPfunc(x,snr,y_gn,False, h1low, h1high, h2low, h2high, h1low_z, h1high_z, h2low_z, h2high_z)
sig = (prmnp - prmn)

prmnp1 = prmn + sig    
prmnn1 = prmn - sig
prmnp2 = prmn + 2*sig
prmnn2 = prmn - 2*sig
prmnp3 = prmn + 3*sig    
prmnn3 = prmn - 3*sig
prmnp4 = prmn + 4*sig    
prmnn4 = prmn - 4*sig
prmnp5 = prmn + 5*sig    
prmnn5 = prmn - 5*sig


# %%
algtest = True
if algtest:
    #xplt = x*(xscale /60)
    xplt = x
    #xplt = x*xscale
    frame1 = plt.gca()
    #xplt = x
    font = { 'family' : 'sans-serif',
                    'weight' : 'normal',
                    'size'   : 15}
    matplotlib.rc('font', **font)

    f, ax = plt.subplots()
    #ax2 = ax.twinx()
    ax.plot(xplt,snr,'+')
    ax.plot(xplt,prmn,color='k')
    ax.fill(np.concatenate([xplt, xplt[::-1]]),
            np.concatenate([prmnp4,
                            (prmnn4)[::-1]]),
            alpha=0.3, fc='r', ec='None', label='$4 \sigma$')
    # =============================================================================
    # ax.fill(np.concatenate([x, x[::-1]]),
    #          np.concatenate([prmnp3[ind],
    #                         (prmnp1[ind])[::-1]]),
    #          alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
    # ax.fill(np.concatenate([x, x[::-1]]),
    #          np.concatenate([prmnn1[ind],
    #                         (prmnn3[ind])[::-1]]),
    #          alpha=0.3, fc='r', ec='None')
    # =============================================================================
    ax.fill(np.concatenate([xplt, xplt[::-1]]),
            np.concatenate([prmnp5,
                            (prmnp4)[::-1]]),
            alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
    ax.fill(np.concatenate([xplt, xplt[::-1]]),
            np.concatenate([prmnn4,
                            (prmnn5)[::-1]]),
            alpha=0.3, fc='g', ec='None')
    #frame1.axes.get_xaxis().set_visible(False)

    #ax2.plot(x, sig, '--', label="learned $\sigma$")
    #ax2.plot(x, sd, '-', label="true $\sigma$")
    #ax2.set_ylabel("$\sigma$")
    ax.set_xlabel("Time (mins)")
    #ax.set_xlabel("Time (days)")
    ax.set_ylabel("SNR (dB)")
    #ax.set_ylabel("Q-factor (dB)")
    ax.set_xlim([xplt[0], xplt[-1]])
    #ax.set_ylim([6.4, 9.5])
    #ax.set_xticks([])
    #ax.set_xticklabels(xlab)
    #ax.set_yticklabels(ylab)
    ax.legend(ncol=2)
    #plt.savefig('HGPtestgoldberg.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestwilliams.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestyuanwhaba.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('hgpalgexample.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('hgpfitMSQcont' + str(batch_ind) + 'ch'  + str(channel) + '.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('javierSNR10HGPdB.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPfitbersnrfull.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPfirst400.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPnewdata.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPnewdata100.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPnewdata400lin.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGP10acctest.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPnewdata100gpex.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPphyspriors100.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('MSdata345bt1hgp100.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    f, ax = plt.subplots()
    ax.plot(xplt, sig, '--', label="learned $\sigma$")
    #ax.plot(x, sd, '-', label="true $\sigma$")
    ax.set_xlabel("Time (mins)")
    ax.set_ylabel("SNR $\sigma$ (dB)")
    #ax.set_ylabel("Q-factor $\sigma$ (dB)")
    ax.set_xlim([xplt[0], xplt[-1]])
    #plt.savefig('HGPtestgoldbergsig.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('hgpfitMSQsigcont' + str(batch_ind) + 'ch'  + str(channel) + '.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPfitbersnrsigmafull.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPfirst400sig.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('javierSNR10HGPdBsig.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPnewdatasig.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPnewdata400siglin.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPnewdata100gpexsig.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGP10acctestsig.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('MSdata345bt1hgp100sig.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('JavierHGPphyspriors100sig.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.plot(lml)
    plt.ylabel("LML")
    plt.xlabel("Number of iterations")
    #plt.savefig('HGPtestgoldberglml.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestwilliamslml.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestyuanwhabalml.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('lmlconvergence.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.plot(MSE)
    plt.ylabel("MSE")
    plt.xlabel("Number of iterations")
    #plt.savefig('HGPtestgoldbergmse.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestwilliamsmse.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestyuanwhabamse.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('mseconvergence.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    """ plt.plot(NLPD)
    plt.ylabel("NLPD")
    #plt.savefig('HGPtestgoldbergnlpd.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestgoldbergnlpd.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestyuanwhabanlpd.pdf', dpi=200,bbox_inches='tight')
    plt.show() """

# %%
# prmn, prmnp, lml, MSE
#plot_title = "javier_data_full"
#plot_title = "javier_data_full_400"
#plot_title = "javiernewdata400lin"
#plot_title = "javiernewdata100gpex"
#plot_title = "javier10acctest"
#plot_title = "MSchannel345batch10"
#plot_title = "MSchannel80batch6"
#plot_title = "MSchannel80batch13"
#plot_title = "MSchannel80batch8"
#plot_title = "MSdata345bt1hgp100"
plot_title = "JavierHGPphyspriors100"

def save_hgp():
    np.savetxt('prmn' + str(plot_title) + '.csv', prmn, delimiter=',') 
    np.savetxt('prmnp' + str(plot_title) + '.csv', prmnp, delimiter=',') 
    np.savetxt('lml' + str(plot_title) + '.csv', lml, delimiter=',') 
    np.savetxt('mse' + str(plot_title) + '.csv', MSE, delimiter=',') 
    np.savetxt('xplt' + str(plot_title) + '.csv', xplt, delimiter=',') 
    np.savetxt('snr' + str(plot_title) + '.csv', snr, delimiter=',')
save_hgp()

# %%

""" np.savetxt('hetprmnextdegst.csv', prmn, delimiter=',')
np.savetxt('hetsigextdegst.csv', sig, delimiter=',') """

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

ind = 2
f, ax = plt.subplots()
ax.plot(x,snr[ind],'+')
ax.plot(x,prmn[ind],color='k')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp3[ind],
                        (prmnn3[ind])[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
# =============================================================================
# ax.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([prmnp3[ind],
#                         (prmnp1[ind])[::-1]]),
#          alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
# ax.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([prmnn1[ind],
#                         (prmnn3[ind])[::-1]]),
#          alpha=0.3, fc='r', ec='None')
# =============================================================================
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp5[ind],
                        (prmnp3[ind])[::-1]]),
         alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnn3[ind],
                        (prmnn5[ind])[::-1]]),
         alpha=0.3, fc='g', ec='None')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim([x[0], x[-1]])
#ax.set_ylim([6.4, 9.5])
#ax.set_xticklabels(xlab)
#ax.set_yticklabels(ylab)
#plt.axis([-1,100,1.0,8.0])
plt.legend()
plt.savefig('hgpfitMSQ.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPextdegst.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPextdegstmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPextdegdvmoredata.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %%
plt.plot(lml[ind])
plt.ylabel("LML")
#plt.savefig('HGPtestgoldberglml.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestwilliamslml.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestyuanwhabalml.pdf', dpi=200,bbox_inches='tight')
plt.show()

plt.plot(MSE[ind])
plt.ylabel("MSE")
#plt.savefig('HGPtestgoldbergmse.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestwilliamsmse.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestyuanwhabamse.pdf', dpi=200,bbox_inches='tight')

plt.show()

plt.plot(NLPD[ind])
plt.ylabel("NLPD")
#plt.savefig('HGPtestgoldbergnlpd.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestgoldbergnlpd.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestyuanwhabanlpd.pdf', dpi=200,bbox_inches='tight')
plt.show()



# 
# %%

# [160,480,960,1200, 1600, 2400]

f, ax = plt.subplots()
#xl = ['0','2','4', '6', '8', '10']
ax.plot(x,sig[0],color='m',LineStyle='-.',label='400 km')
ax.plot(x,sig[1],color='g',LineStyle='-.',label= '800 km')
ax.plot(x,sig[2],color='b',LineStyle='-.',label= '1200 km')
ax.plot(x,sig[3],color='r',LineStyle='-.',label= '1600 km')

""" ax.plot(x,sig[0],color='r',LineStyle='-.',label='A')
ax.plot(x,sig[1],color='b',LineStyle='-.',label='B')
ax.plot(x,sig[2],color='g',LineStyle='-.',label= 'C') """


# for standard heteroscedastic experiment 
#sdhet = np.linspace(0.04,0.08,numpoints)
numyrsh = 200
# for simulation of TRx fault 
sdh1 = np.linspace(0.04,0.042,int(numyrsh/2)+1)
sdh2 = np.linspace(0.08,0.082,int(numyrsh/2)-1)
sdhet = np.append(sdh1,sdh2)

# for comparison of different variances 
""" sdhet1 = np.linspace(0.04,0.06,numpoints)
sdhet2 = np.linspace(0.04,0.08,numpoints)
sdhet3 = np.linspace(0.04,0.16,numpoints)  """

ax.plot(x,sdhet,color='k',label='Eq. (1) $\sigma(t)$')
""" ax.plot(x,sdhet1,color='r',label='Eq. (1) $\sigma(t)$ A')
ax.plot(x,sdhet2,color='b',label='Eq. (1) $\sigma(t)$ B')
ax.plot(x,sdhet3,color='g',label='Eq. (1) $\sigma(t)$ C') """
 
#plt.plot(x,sigrf[ind],color='k',LineStyle='-',label='$\sqrt{r(x)}$ 1')

ax.set_xlabel("time (years)")
ax.set_ylabel("SNR $\sigma$ (dB)")
#ax2.set_ylabel("Shannon throughput gain (%)")
ax.set_xlim([x[0], x[-1]])
#ax.set_ylim([0.03, 0.12])
#ax.set_xticklabels(xl)
ax.legend(loc=2,ncol=2, prop={'size': 11})

#plt.savefig('JOCNhetgpsig.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetgpsig20moredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetgpsigdvmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetgpsigst.pdf', dpi=200,bbox_inches='tight')
plt.savefig('hetgpsigstmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.axis([x[0],x[-1],0.03,0.09])
#plt.xticklabels()
plt.show()

# %%

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

ln2 = ax1.plot(x,snr[ind],'+')
ln3 = ax1.plot(x,prmn[ind],color='k')

ln4 = ax1.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp3[ind],
                        (prmnn3[ind])[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
# =============================================================================
# ax.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([prmnp3[ind],
#                         (prmnp1[ind])[::-1]]),
#          alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
# ax.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([prmnn1[ind],
#                         (prmnn3[ind])[::-1]]),
#          alpha=0.3, fc='r', ec='None')
# =============================================================================
ln5 = ax1.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp5[ind],
                        (prmnp3[ind])[::-1]]),
         alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
ln6 = ax1.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnn3[ind],
                        (prmnn5[ind])[::-1]]),
         alpha=0.3, fc='g', ec='None')
#ln7 = ax2.plot(x,sig[ind],color='r',LineStyle='-.',label='$R(time)$')
#ln8 = ax2.plot(x,sdhet,'--',color='k',label='Eq. (1) $\sigma(t)$')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("SNR (dB)")
#ax2.set_ylabel("SNR $\sigma$ (dB)")
    
ax1.set_xlim([x[0], x[-1]])
#ax1.set_ylim([13.2,14.5])
#ax2.set_ylim([0.03,0.09])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln4+ln5
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,loc=1, ncol=2, prop={'size': 13})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('hetGPexampledvmoredata.pdf', dpi=200,bbox_inches='tight')
plt.savefig('hetGPexamplestmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPexamplest.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPexample20moredata.pdf', dpi=200,bbox_inches='tight')
plt.show()



# %%
