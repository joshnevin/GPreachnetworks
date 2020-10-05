# %% analysis of MS Q-factor dataset 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W, ExpSineSquared, RationalQuadratic 
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, skew, skewnorm


# functions

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

def load_clean_save_raw_data_by_batch(
    root_dir,
    output_data_dirr,
    sep: str,
    header,
    time_window,
    means,
    stds
) -> pd.DataFrame:
    """
    Load, clean and save raw data batch by batch.

    Parameters
    ----------
    root_dir : str
        Directory to probe for loading.
    output_data_dir : str
        Directory where to save cleaned data.
    sep : str
        Separator used when reading CSVs.
    cols_to_keep : List[str]
        List of columns to keep. All other columns are dropped.
    """
    for i, file in enumerate(os.listdir(root_dir)):
        
        print(i, file)
        file_path = os.path.join(root_dir, file)
        df = pd.read_csv(file_path, sep=',', header=[0,1])
        try:
            df = clean_batch_of_raw_data_time_windowed(df, time_window, means, stds)
            save(df, output_data_dir, file.replace('.csv', '.pkl'))
        except:
            print("----- Error in processing ", file, " --- Mostly due to cascading failure ")


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

def down_sample(df, sam):
    """
    downsample by a factor 'sam' and return Q and CD arrays 
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


def get_segment(channel):
    """
    get segment from channel number 
    """
    channel = str(channel)
    return int(simulation_files_dict[channel].split("_")[3].rstrip('.txt'))

def find_bad_segments(simulation_files_dict, root_dir):
    """
    find bad segments based on PMD SD > 2% of mean 
    """
    badinds = []
    for i in range(len(simulation_files_dict) - 1):
        df = read_txt_to_df(simulation_files_dict[str(i+1)], root_dir)
        Qarr, _, _ = down_sample(df, 1)
        Qsd = np.std(Qarr)
        Qmean = np.mean(Qarr)
        if Qsd > 0.02*Qmean:
            badinds.append(i)
        else:
            continue
    badsegs = [get_segment(i+1) for i in badinds]
    return list(dict.fromkeys(badsegs)) 

def GPtrain(x,y):
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
    kernel = C(1, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3)) 
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 20, normalize_y=False, alpha=np.var(y))
    gpr.fit(x, y)
    #print("Optimised kernel: %s" % gpr.kernel_)
    ystar, sigma = gpr.predict(x, return_std=True )
    sigma = np.reshape(sigma,(np.size(sigma), 1)) 
    sigma = (sigma**2 + 1)**0.5  
    ystarp = ystar + sigma
    ystari = scaler.inverse_transform(ystar)
    ystarpi = scaler.inverse_transform(ystarp)
    sigmai = (ystarpi - ystari)
    return ystari, sigmai

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

def sample_std(qfac,n): # find the variance for samples of size n
    """
    sample std of Q in batches of size n
    """
    varsams = []
    for i in range(0,len(qfac),n):
        varsams.append(np.var(qfac[i:i+n])**0.5)
    return varsams

def convert_to_lin_Q(Q):
    """
    convert from Q(dB) to Q
    """
    return 10**(Qarr/20)

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
# get files and make a dictionary 
root_dir = '/Users/joshnevin/Desktop/MicrosoftDataset'
file_list_simulations= get_list_simulations(root_dir)
simulation_files_dict = get_dict_scenario_txt(file_list_simulations)
#file_path = os.path.join(root_dir, test_file) 
#badsegs = find_bad_segments(simulation_files_dict, root_dir)
# %% select channel and segment 
channel = '345'
print("segment = " + str(get_segment(channel)))
df = read_txt_to_df(simulation_files_dict[channel], root_dir)
timecol = get_times_from_dates(df)
df = df.drop('date', axis =1)
df['time'] = timecol
df = drop_bad_values(df)
timecol = df['time'].to_numpy()
discons = get_discontinuities_in_time(timecol)
Qarr, CDarr, _ = down_sample(df, 1) # set sam = 1 to just return arrays with same sampling
#Qarr = drop_bad_values(CDarr, Qarr)
Qarr = convert_to_lin_Q(Qarr)

Qmean = np.mean(Qarr) # ignore Nans
Qsd = np.std(Qarr) # ignore Nans
Qskew = skew(Qarr) # ignore Nans
# fit Gaussian to histogram

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 14}
matplotlib.rc('font', **font)


plot_discon = False
if plot_discon:
    fig, ax = plt.subplots()
    ax.plot(timecol)
    ax.set_ylabel("Time (mins)")
    ax.set_xlabel("Samples")
    plt.title("Channel " + str(channel))
    plt.savefig('timediscons' + str(channel) + '.pdf', dpi=200,bbox_inches='tight')
    plt.show()


    Qvar = sample_std(Qarr, 100)
    fig, ax = plt.subplots()
    ax.plot(Qvar, '+')
    ax.set_ylabel("Q-factor $\sigma$ (dB)")
    ax.set_xlabel("Time (AU)")
    #plt.savefig('hyp2variation.pdf', dpi=200,bbox_inches='tight')
    plt.show()


    plt.hist(Qarr, normed=True, color='c')
    plt.xlim((np.nanmin(Qarr), np.nanmax(Qarr)))
    x = np.linspace(np.nanmin(Qarr), np.nanmax(Qarr), len(Qarr))
    plt.plot(x, norm.pdf(x, Qmean, Qsd), label = 'Normal', color='r')
    plt.plot(x, skewnorm.pdf(x, -Qskew, Qmean, Qsd), label = 'Skewed', color='b')
    plt.xlabel("Q-factor (dB)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("channel " + str(channel))
    plt.savefig('Qlinvstimehistch' + str(channel) + '.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.hist(Qvar, normed=True, color='c')
    plt.xlim((np.nanmin(Qvar), np.nanmax(Qvar)))
    x = np.linspace(np.nanmin(Qvar), np.nanmax(Qvar), len(Qvar))
    Qvarmean = np.nanmean(Qvar)
    Qvarsd = np.nanstd(Qvar)
    Qvarskew = skew(Qvar, nan_policy='omit')
    plt.plot(x, norm.pdf(x, Qvarmean, Qvarsd), label = 'Normal', color='r')
    plt.plot(x, skewnorm.pdf(x, -Qskew, Qvarmean, Qvarsd), label = 'Skewed', color='b')
    plt.xlabel("Q-factor $\sigma$ (dB)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("channel " + str(channel))
    plt.savefig('Qlinvarvstimehistch' + str(channel) + '.pdf', dpi=200,bbox_inches='tight')
    plt.show()

# %% find batches of contiguous data

batches = get_contiguous_bactches(discons, Qarr)
batch_lens = ([len(i) for i in batches ] )
batch_max = batch_lens.index(max(batch_lens))
batch_ind = 10
Qvar = sample_std(batches[batch_ind], 100)

plot_sample_std_cont = False
if plot_sample_std_cont:
    fig, ax = plt.subplots()
    ax.plot(Qvar, '+')
    ax.set_ylabel("Q-factor $\sigma$ (dB)")
    ax.set_xlabel("Samples (increasing time)")
    #plt.savefig('hyp2variation.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.hist(Qvar, normed=True, color='c')
    plt.xlim((np.nanmin(Qvar), np.nanmax(Qvar)))
    x = np.linspace(np.nanmin(Qvar), np.nanmax(Qvar), len(Qvar))
    Qvarmean = np.nanmean(Qvar)
    Qvarsd = np.nanstd(Qvar)
    Qvarskew = skew(Qvar, nan_policy='omit')
    plt.plot(x, norm.pdf(x, Qvarmean, Qvarsd), label = 'Normal', color='r')
    plt.plot(x, skewnorm.pdf(x, -Qskew, Qvarmean, Qvarsd), label = 'Skewed', color='b')
    plt.xlabel("Q-factor $\sigma$ (linear)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("channel " + str(channel) + " contig " + str(batch_ind))
    plt.savefig('Qlinvarhistsegch' + str(channel) + '.pdf', dpi=200,bbox_inches='tight')
    plt.show()

# %% perform downsampling for GP fitting 
downsampleval = 1
#Qarrtrain = np.asarray([Qarr[i] for i in range(0, round(len(Qarr)), downsampleval)])  # downsample by factor of 10
#Qarrtrain = np.asarray(varsample(Qarr, 100))
#xgp = np.asarray([timecol[i] for i in range(0, round(len(timecol)), downsampleval)])
Qarrtrain = batches[batch_ind]
timecol_gp = timecol[discons[batch_ind-1]:discons[batch_ind]]
#x = np.asarray([timecol[i]/(24*60) for i in range(0, round(len(timecol)), downsampleval)])
xgp = np.asarray([timecol_gp[i]/(24*60) for i in range(0, round(len(timecol_gp)), downsampleval)])
#xgp = x.reshape(len(xgp), 1)

# %% try different kernels and record LML

kernel1 = C(1, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4))

kernel2 = C(1, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4))  + C(1, (1e-4, 1e4)) * RationalQuadratic(1, 1, (1e-4, 1e4), (1e-4, 1e4))

kernel3 = C(1, (1e-4, 1e4)) * RationalQuadratic(1, 1, (1e-4, 1e4), (1e-4, 1e4))

kernel4 = C(1, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4)) + \
    C(1, (1e-4, 1e4)) * ExpSineSquared(1, 1, (1e-4, 1e4), (1e-4, 1e4)) * RBF(1, (1e-4, 1e4))

kernel5 = C(1, ((1e-4, 1e4))) * ExpSineSquared(1, 1, (1e-4, 1e4), (1e-4, 1e4)) * RBF(1, (1e-4, 1e4))
   
kernel6 = C(1, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4)) + \
    C(1, (1e-4, 1e4)) * ExpSineSquared(1, 1, (1e-4, 1e4), (1e-4, 1e4)) * RBF(1, (1e-4, 1e4)) + \
    C(1, (1e-4, 1e4)) * RationalQuadratic(1, 1, (1e-4, 1e4), (1e-4, 1e4))

start = time.time()
prmn, sdgp, GPmodel, theta, lml = GP_train_kernel(xgp, Qarrtrain, kernel1)
end = time.time()
print("GP fitting took " + str((end-start)/60) + " minutes")
print("LML = " + str(lml))
thetatrans = np.exp(theta)
print("theta (linear) = " + str(thetatrans))
print("theta (log) = " + str(theta))

# %% plot fitted GP model 
sdgp = np.mean(sdgp)
prmnp3 = prmn + 3*sdgp
prmnm3 = prmn - 3*sdgp

fig, ax = plt.subplots()
ax.plot(xgp, Qarrtrain, '+', color = 'k')
ax.plot(xgp, prmn, color = 'r')
linfit = np.polyfit(xgp, Qarrtrain, 1)
p = np.poly1d(linfit)
#ax.plot(xgpplt, p(xgpplt), label = 'linear fit')
ax.fill(np.concatenate([xgp, xgp[::-1]]),
         np.concatenate([prmnp3,
                        (prmnm3)[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
plt.legend(ncol = 3)
plt.title("Channel " + str(channel))
ax.set_ylabel("Q-factor (dB)")
ax.set_xlabel("time (hours)")
ax.set_xlim([xgp[0], xgp[-1]])
#ax.plot(xgp, Qgp, label = "GP pred. mean", color = 'r')
plt.savefig('GPfitCh' + str(channel) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% plot LML vs hyperparameter curves 

#thetatests = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
thetatests = np.linspace(-10,10,20)

lmlnew = []
for i in range(len(thetatests)):
    thetanew = theta
    thetanew[1] =   thetatests[i]    
    lmlnew.append(GPmodel.log_marginal_likelihood(theta = thetanew))

fig, ax = plt.subplots()
ax.plot(thetatests, lmlnew)
ax.set_ylabel("Log marginal likelihood")
ax.set_xlabel("H2")
plt.savefig('hyp2variation.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% plotting 
#fig, ax = plt.subplots()
a = np.hstack(Qarr)
_ = plt.hist(a, color='c')
plt.xlabel("Q-factor")
plt.ylabel("Frequency")
plt.savefig('Qvstimech1seghist.pdf', dpi=200,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.plot(Qarr, '+')
ax.set_ylabel("Q-factor (dB)")
ax.set_xlabel("time (hours)")
plt.savefig('Qvstimech1seg.pdf', dpi=200,bbox_inches='tight')
plt.show()

""" fig, ax = plt.subplots()
ax.plot(PMDarr, '+')
ax.set_ylabel("PMD (ps)")
ax.set_xlabel("time (hours)")
plt.savefig('pmdplotex.pdf', dpi=200,bbox_inches='tight')
plt.show() """

""" fig, ax = plt.subplots()
ax.plot(CDarr,'+')
ax.set_ylabel("CD (ps/nm)")
ax.set_xlabel("time (hours)")
plt.savefig('CDvstimech1seg1.pdf', dpi=200,bbox_inches='tight')
plt.show() """

""" fig, ax = plt.subplots()
ax.plot(Parr, '+')
ax.set_ylabel("P (dBm)")
ax.set_xlabel("time (hours)")
#plt.savefig('powerplotex.pdf', dpi=200,bbox_inches='tight')
plt.show() """


""" Qgrad = np.gradient(np.asarray(Qarr))
plt.plot(Qgrad)
plt.show() """

# %%
