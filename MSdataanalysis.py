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
from scipy.stats import spearmanr, pearsonr
from scipy.special import erf
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
    return 10**(Q/20)

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
    for i in range(len(time_column) - 1):
        if time_column[i+1] - time_column[i] > 15.0:
            discons.append(i+1)
    return discons

def get_contiguous_bactches(discons, Q):
    """
    split up Q into contiguous segments 
    """  
    discons = [0] + discons
    segs = []
    for i in range(len(discons) -1 ):
        segs.append(Q[discons[i]:discons[i+1]])
    return segs

def find_corr_between_batches(channel1, channel2, batch_ind):
    """
   find spearman's rank correlation between two contigous data batches 
    """
    batches_1, timecol_1, discons_1 = read_and_process_data(channel1)
    batches_2, timecol_2, discons_2 = read_and_process_data(channel2)
    # 
    
    Q_1 = batches_1[batch_ind]
    time_1 = timecol_1[discons_1[batch_ind]:discons_1[batch_ind+1]]/60 # hours
    Q_2 = batches_2[batch_ind]
    #time_2 = timecol_1[[0]:discons_2[0]]
    corr, _ = spearmanr(Q_1, Q_2)
    return corr

def read_and_process_data(channel):
    """
    read in data for one channel, get the time column, convert Q to an array and 
    split data into contiguous batches 
    """
    df = read_txt_to_df(simulation_files_dict[channel], root_dir)
    timecol = get_times_from_dates(df)
    df = df.drop('date', axis =1)
    df['time'] = timecol
    df = drop_bad_values(df)
    timecol = df['time'].to_numpy()
    discons = get_discontinuities_in_time(timecol)
    Qarr, _, _ = down_sample(df, 1) # set sam = 1 to just return arrays with same sampling
    Qarr = convert_to_lin_Q(Qarr)
    batches = get_contiguous_bactches(discons, Qarr)
    discons = [0] + discons
    return batches, timecol, discons


def check_for_fully_contiguous_channels(channel):
    """
    read in data for one channel, get the time column, convert Q to an array and 
    split data into contiguous batches 
    """
    df = read_txt_to_df(simulation_files_dict[channel], root_dir)
    timecol = get_times_from_dates(df)
    print(df['Q-factor'][0])
    """ df = df.drop('date', axis =1)
    df['time'] = timecol
    df = drop_bad_values(df)
    timecol = df['time'].to_numpy() """    
    return  check_for_discon_in_time(timecol)

def check_for_discon_in_time(time_column):
    """
    get the indices for which the next 
                        """
    
    for i in range(len(time_column) - 1):
        if time_column[i+1] - time_column[i] > 15.0:
            discon = True
        else:
            discon = False
    return discon

def autocorr(x):
    x = x - np.mean(x)
    auto_corr = np.correlate(x, x, mode='full')
    index_max = np.argmax(auto_corr)
    return auto_corr[index_max:]/auto_corr[index_max]


# get files and make a dictionary 
root_dir = '/Users/joshnevin/Desktop/MicrosoftDataset'
file_list_simulations= get_list_simulations(root_dir)
simulation_files_dict = get_dict_scenario_txt(file_list_simulations)
#file_path = os.path.join(root_dir, test_file) 
#badsegs = find_bad_segments(simulation_files_dict, root_dir)

# %% plots for JOCN 2020
font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 14}
matplotlib.rc('font', **font)
x = np.linspace(0,1,100)
sig = np.random.normal(0.05,0.01,100)

y = []
for i in range(100):
    y.append( -2*x[i] + 1 + np.random.normal(0, sig[i]) ) 


y5p = y + 5*sig
y5n = y - 5*sig
y4p = y + 4*sig
y4n = y - 4*sig
fig, ax = plt.subplots()
ax.plot(x,y, color='k')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y4p,
                        (y4n)[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$4 \sigma$')
ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y5p,
                            (y4p)[::-1]]),
            alpha=0.5, fc='g', ec='None', label='$5 \sigma$')
ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y4n,
                            (y5n)[::-1]]),
            alpha=0.5, fc='g', ec='None')

ax.set_ylabel("QoT")
ax.set_xlabel("Time")
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
ax.set_xlim([x[0], x[-1]])
plt.legend(loc=1)
#plt.title("Channel " + str(channel))
plt.savefig('GPexamplefit.pdf', dpi=200,bbox_inches='tight')
plt.show()

sigplot = np.linspace(4,5,100)
errprob = erf(sigplot/(2**0.5))*100

fig, ax1 = plt.subplots()
ln1 = ax1.plot(sigplot, errprob)
ax1.set_ylabel("Availability (%)")
ax1.set_xlabel("GP confidence (number of $\sigma$)")
        
ax1.set_xlim([4,5])
ax1.set_ylim([errprob[0],100])
ax1.set_xticks([4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0])
ax1.set_yticks([99.994, 99.995, 99.996, 99.997, 99.998, 99.999, 99.9999])
ax1.set_yticklabels(['99.994', '99.995', '99.996', '99.997', '99.998', '99.999', '99.9999'])
plt.grid()
lns = ln1
labs = [l.get_label() for l in lns]
#ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
        #plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('availvssigma.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %%
x = np.linspace(0,10,1000)

y = 0.3*np.sin(x)*np.cos(0.5*x)  + 0.2*np.cos(0.6*x) 

plt.plot(x,y)

plt.show()


# %% get correlation between channels 

def find_linear_corr_between_batches(channel1, channel2, batch_ind):
    """
   find spearman's rank correlation between two contigous data batches 
    """
    batches_1, timecol_1, discons_1 = read_and_process_data(channel1)
    batches_2, timecol_2, discons_2 = read_and_process_data(channel2)
    # 
    
    Q_1 = batches_1[batch_ind]
    time_1 = timecol_1[discons_1[batch_ind]:discons_1[batch_ind+1]]/60 # hours
    Q_2 = batches_2[batch_ind]
    #time_2 = timecol_1[[0]:discons_2[0]]
    corr, _ = pearsonr(Q_1, Q_2)
    return corr


batches_1, timecol_1, discons_1 = read_and_process_data('80')
batch = 3
channel1 = '345'
#channel2 = ['76','77','78','79','81','82','83','84']
channel2 = ['342','343','344','346','347','348']
#channel2 = ['2','3','4','6','7','8']
corr = []
#corr = [find_linear_corr_between_batches(channel1, i, batch) for i in channel2] 
corr = [find_corr_between_batches(channel1, i, batch) for i in channel2] 
#test2 =  find_corr_between_batches(channel1, '2', 1)

#x = np.array([0,1,2,3])
x = np.linspace(0,len(channel2),len(channel2))
plt.plot(x,corr)
plt.xticks(x, channel2)
plt.xlabel("Comparison channel")
plt.ylabel("Correlation")
#plt.title("Channel " + str(channel1) + ' batch ' + str(batch) )
plt.savefig('corrch' + str(channel1) + 'batch' + str(batch) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %% check for fully contiguous channels 

#contigs = []
start = time.time()
contigs = [i+1  for i in range(5) if check_for_fully_contiguous_channels(str(i+1)) ]
end = time.time()
print(end-start)
np.savetxt('contigs.csv', contigs, delimiter=',') 

# %% find the 5% and 95% percentile of all the batches for all the channels
def find_bounds_for_GP_algorithm(channel):
    """
    find the 5% and 95% percentiles for each contiguous batch for each channel 
    """
    df = read_txt_to_df(simulation_files_dict[channel], root_dir)
    timecol = get_times_from_dates(df)
    df = df.drop('date', axis =1)
    df['time'] = timecol
    #df = drop_bad_values(df)
    timecol = df['time'].to_numpy()
    discons = get_discontinuities_in_time(timecol)
    Qarr, _, _ = down_sample(df, 1) # set sam = 1 to just return arrays with same sampling
    #Qarr = convert_to_lin_Q(Qarr)
    batches = get_contiguous_bactches(discons, Qarr)
    lower_bounds = []
    upper_bounds = []
    for i in range(len(batches)):
        lower_bound, upper_bound = np.percentile(sample_std(batches[i], 50), [5, 95])
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)  
    #discons = [0] + discons
    return lower_bounds, upper_bounds

all_lower_bounds = []
all_upper_bounds = []
start = time.time()
for i in range(4000):
    lower_bounds, upper_bounds = find_bounds_for_GP_algorithm(str(i+1))
    all_lower_bounds.extend(lower_bounds)
    all_upper_bounds.extend(upper_bounds)
    if i % 100 == 0:
        print("completed channel " + str(i))
end = time.time()
print(end-start)

np.savetxt('allupperbounds.csv', all_upper_bounds, delimiter=',')
np.savetxt('alllowerbounds.csv', all_lower_bounds, delimiter=',')



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

Qmean = np.mean(Qarr) 
Qsd = np.std(Qarr) 
Qskew = skew(Qarr) 
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

# %%
batch_ind = 4
Qplt = batches[batch_ind]

x = timecol[discons[batch_ind-1]:discons[batch_ind]]/(60*24)
plt.plot(x,Qplt)
plt.xlabel("Time (days)")
plt.ylabel("Q-factor")
plt.title("Channel "+ str(channel) + ' batch ' + str(batch_ind))
plt.savefig('Qtimeseries' + str(channel) + 'batch' + str(batch_ind) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()

plt.hist(Qplt, normed=True, color='c')
plt.xlim((np.nanmin(Qplt), np.nanmax(Qplt)))
x = np.linspace(np.nanmin(Qplt), np.nanmax(Qplt), len(Qplt))
Qvarmean = np.nanmean(Qplt)
Qvarsd = np.nanstd(Qplt)
Qvarskew = skew(Qplt, nan_policy='omit')
plt.plot(x, norm.pdf(x, Qvarmean, Qvarsd), label = 'Gaussian', color='r')
plt.plot(x, skewnorm.pdf(x, Qskew, Qvarmean, Qvarsd), label = 'Skewed Gaussian', color='b')
plt.xlabel("Q-factor")
plt.ylabel("Normalised frequency")
plt.title("Channel "+ str(channel) + ' batch ' + str(batch_ind))
#plt.legend(loc=2)
#plt.savefig('Qhist' + str(channel) + 'batch' + str(batch_ind) + 'nofit.pdf', dpi=200,bbox_inches='tight')
plt.savefig('Qhist' + str(channel) + 'batch' + str(batch_ind) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% autocorrelation

Q_gauss_test = np.random.normal(np.mean(Qplt), np.std(Qplt), (len(Qplt))) 

auto_corr = autocorr(Qplt)
auto_corr_gauss = autocorr(Q_gauss_test)

plt.plot(auto_corr, label='Experimental')
plt.plot(auto_corr_gauss, label='Gaussian')
plt.xlabel("Time difference")
plt.ylabel("Auto correlation")
plt.legend()
plt.title("channel " + str(channel) + ' batch ' + str(batch_ind))
plt.savefig('aurocorrcompMSch' + str(channel) + 'batch' + str(batch_ind) +  '.pdf', dpi=200,bbox_inches='tight')
plt.show()

""" plt.plot(auto_corr, label='Experimental')
plt.xlabel("Time difference")
plt.ylabel("Auto correlation")
plt.legend()
#plt.savefig('aurocorrexpMSch' + str(channel) + 'batch' + str(batch_ind) +  '.pdf', dpi=200,bbox_inches='tight')
plt.show()

plt.plot(auto_corr_gauss, label='Gaussian')
plt.xlabel("Time difference")
plt.ylabel("Auto correlation")
plt.legend()
#plt.savefig('aurocorrgaussMSch' + str(channel) + 'batch' + str(batch_ind) +  '.pdf', dpi=200,bbox_inches='tight')
plt.show() """


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
#sdgp = np.mean(sdgp)
prmnp4 = prmn + 4*sdgp
prmnm4 = prmn - 4*sdgp
prmnp5 = prmn + 5*sdgp
prmnm5 = prmn - 5*sdgp

fig, ax = plt.subplots()
ax.plot(xgp, Qarrtrain, '+', color = 'k')
ax.plot(xgp, prmn, color = 'r')
linfit = np.polyfit(xgp, Qarrtrain, 1)
p = np.poly1d(linfit)
#ax.plot(xgpplt, p(xgpplt), label = 'linear fit')
ax.fill(np.concatenate([xgp, xgp[::-1]]),
         np.concatenate([prmnp4,
                        (prmnm4)[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$4 \sigma$')
ax.fill(np.concatenate([xgp, xgp[::-1]]),
            np.concatenate([prmnp5,
                            (prmnp4)[::-1]]),
            alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
ax.fill(np.concatenate([xgp, xgp[::-1]]),
            np.concatenate([prmnm4,
                            (prmnm5)[::-1]]),
            alpha=0.3, fc='g', ec='None')
plt.legend(ncol = 3)
#plt.title("Channel " + str(channel))
ax.set_ylabel("Q-factor (dB)")
#ax.set_xlabel("Time (hours)")
ax.set_xlabel("Time")
ax.set_xlim([xgp[0], xgp[-1]])
ax.set_xticks([])
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
