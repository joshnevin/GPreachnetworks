# %% analysis of MS Q-factor dataset 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 16}
matplotlib.rc('font', **font)

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

def sample_every_hour(df):
    """
    downsample to every hour and return Q and CD arrays 
    """
    Qarr = df["Q-factor"].to_numpy()
    CDarr = df["CD"].to_numpy()
    PMDarr = df["PMD"].to_numpy()
    Qarr = [Qarr[i] for i in range(0, round(len(Qarr)), 4)]  # select every hour 
    CDarr = [CDarr[i] for i in range(0, round(len(CDarr)), 4)]  # select every hour 
    PMDarr = [PMDarr[i] for i in range(0, round(len(PMDarr)), 4)]  # select every hour 
    return Qarr, CDarr, PMDarr

def drop_bad_values(CDarr, Qarr):
    """
    set Q factor values for which CD is > 2 sigma from the mean to NaN so they are missed from plots
    """
    CDsd = np.std(CDarr)
    CDmean = np.mean(CDarr)
    for i in range(len(CDarr)):
        if abs(CDarr[i] - CDmean) > 2*CDsd:
            Qarr[i] = np.nan
    return Qarr

def get_segment(channel):
    """
    get segment from channel number 
    """
    channel = str(channel)
    return int(simulation_files_dict[channel].split("_")[3].rstrip('.txt'))

def find_bad_segments(simulation_files_dict, root_dir):
    """
    find good segments based on PMD SD > 10% of mean 
    """
    badinds = []
    for i in range(len(simulation_files_dict) - 1):
        df = read_txt_to_df(simulation_files_dict[str(i+1)], root_dir)
        Qarr, _, _ = sample_every_hour(df)
        Qsd = np.std(Qarr)
        Qmean = np.mean(Qarr)
        if Qsd > 0.02*Qmean:
            badinds.append(i)
        else:
            continue
    badsegs = [get_segment(i+1) for i in badinds]
    return list(dict.fromkeys(badsegs)) 

# %% get files and make a dictionary 
root_dir = '/Users/joshnevin/Desktop/MicrosoftDataset'
file_list_simulations= get_list_simulations(root_dir)
simulation_files_dict = get_dict_scenario_txt(file_list_simulations)
#file_path = os.path.join(root_dir, test_file)
# %%
badsegs = find_bad_segments(simulation_files_dict, root_dir)
# %%
get_segment('472')
# %% select channel and segment 

df = read_txt_to_df(simulation_files_dict['1'], root_dir)
Qarr, CDarr, PMDarr = sample_every_hour(df)
Qmean = np.mean(Qarr)
Qsd = np.std(Qarr)

#Qarr = drop_bad_values( CDarr, Qarr)

# %%
#fig, ax = plt.subplots()
a = np.hstack(Qarr)
_ = plt.hist(a, bins='auto')
plt.xlabel("Q-factor")
plt.ylabel("Frequency")
plt.savefig('Qvstimech1seghist.pdf', dpi=200,bbox_inches='tight')
plt.show()


# %%
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
# %%
fig, ax = plt.subplots()
ax.plot(CDarr,'+')
ax.set_ylabel("CD (ps/nm)")
ax.set_xlabel("time (hours)")
#plt.savefig('CDvstimech1seg1.pdf', dpi=200,bbox_inches='tight')
plt.show()

""" fig, ax = plt.subplots()
ax.plot(Parr, '+')
ax.set_ylabel("P (dBm)")
ax.set_xlabel("time (hours)")
#plt.savefig('powerplotex.pdf', dpi=200,bbox_inches='tight')
plt.show() """



# %%
