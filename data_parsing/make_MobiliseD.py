import os
import pickle
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.stats import mode


def loadmat(filename):
    '''
    This function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from mat objects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def Bouts2Labels(BoutsInfo, fs, labels):
    for bout in range(len(BoutsInfo)):
        CurrentBoutsInfo = BoutsInfo[bout]
        StartPoint = int(CurrentBoutsInfo.Start * fs)
        EndPoint = int(CurrentBoutsInfo.End * fs)
        labels[StartPoint:EndPoint] = 1
    return labels


def resample_data(data, original_fs, target_fs):
    '''
    :param data: Numpy array. Data to resample.
    :param original_fs: Float, the raw data sampling rate
    :param target_fs: Float, the sampling rate of the resampled signal
    :return: resampled data
    '''
    # calculate resampling factor
    resampling_factor = original_fs / target_fs
    # calculate number of samples in the resampled data and labels
    num_samples = int(len(data) / resampling_factor)
    # use scipy.signal.resample function to resample data, labels, and subjects
    resampled_data = signal.resample(data, num_samples)
    return resampled_data


RESAMPLED_HZ = 30
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(RESAMPLED_HZ * WINDOW_SEC)
WINDOW_OVERLAP_LEN = int(RESAMPLED_HZ * WINDOW_OVERLAP_SEC)
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN

DATA_PATH = r"N:\Gait-Neurodynamics by Names\Yonatan\SSL\ElderNet\data\MobiliseD\data\mat\Free-Living\data.mat"
OUTPUT_PATH = r"N:\Gait-Neurodynamics by Names\Yonatan\SSL\ElderNet\data\MobiliseD\data"

data = loadmat(DATA_PATH)  # Load the .mat file as a dictionary
data = data['data']  # Read the data from the dictionary
# Check the configuration of the wrist-worn sensor (right/left)
if 'LeftWrist' in data['TimeMeasure1']['Recording4']['SU_INDIP'].keys():
    RawData = data['TimeMeasure1']['Recording4']['SU_INDIP']['LeftWrist']
elif 'RightWrist' in data['TimeMeasure1']['Recording4']['SU_INDIP'].keys():
    RawData = data['TimeMeasure1']['Recording4']['SU_INDIP']['RightWrist']

# Load the information on the start and stop of each walking bout according to the INDIP system
if 'INDIP' in data['TimeMeasure1']['Recording4']['Standards'].keys():
    BoutsInfo = data['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod']


# Extract the acceleration & gyro data and the sampling rate from the loaded struct
acc = RawData['Acc']  # Acceleration of the wrist worn sensor
fs = RawData['Fs']['Acc']

# Extract the labels using the start and end point of each bout as indicated in the BoutsInfo
labels = np.zeros((acc.shape[0], 1))
if 'ndarray' in str(type(BoutsInfo)):
    labels = Bouts2Labels(BoutsInfo, fs, labels)
elif 'dict' in str(type(BoutsInfo)):
    BoutStart = int(BoutsInfo['Start'] * fs)
    BoutEnd = int(BoutsInfo['End'] * fs)
    labels[BoutStart:BoutEnd] = 1

data_res = resample_data(acc, fs, RESAMPLED_HZ)
labels_res = resample_data(labels, fs, RESAMPLED_HZ)
labels_res = np.round(labels_res).astype(int).squeeze()

# Divide data to windows
data_win = np.empty((len(labels_res) // WINDOW_STEP_LEN, WINDOW_LEN, 3))
labels_win = np.empty((len(labels_res) // WINDOW_STEP_LEN,))
for i, start_idx in enumerate(range(0, (len(labels_res) - WINDOW_STEP_LEN), WINDOW_STEP_LEN)):
    end_idx = start_idx + WINDOW_LEN
    data_win[i, :, :] = data_res[start_idx:end_idx]
    labels_win[i] = mode(labels_res[start_idx:end_idx])[0]

# Convert labels to one-hot encoded array
one_hot_labels = np.zeros((len(labels_win), 2), dtype=int)
one_hot_labels[np.arange(len(labels_win)), labels_win.astype(int)] = 1

with open(os.path.join(OUTPUT_PATH, 'WindowsData.p'), 'wb') as f:
    pickle.dump(data_win, f)

with open(os.path.join(OUTPUT_PATH, 'WindowsLabels.p'), 'wb') as f:
    pickle.dump(one_hot_labels, f)
