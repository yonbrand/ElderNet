import os
import random
from datetime import datetime, timedelta
import numpy as np
import mat73
from scipy.io import loadmat
from scipy import signal
import glob
from tqdm.auto import tqdm
import pandas as pd
import multiprocessing
from functools import partial
import logging


# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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



CHUNK_SIZE = 1 # Number of files to execute in parallel
RESAMPLED_HZ = 30# Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(RESAMPLED_HZ * WINDOW_SEC)
WINDOW_OVERLAP_LEN = int(RESAMPLED_HZ * WINDOW_OVERLAP_SEC)
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN
MIN_DATA = 0 #can be modified to set a threshold for minimal recording duration
# Path to the folder with demo example of one participant from RUSH Memory and Aging Project (MAP)
DATAFILES = 'N:\Gait-Neurodynamics by Names\Yonatan\SSL\ElderNet\data\RUSH\data\mat\*.mat'
OUTDIR = r'N:\Gait-Neurodynamics by Names\Yonatan\SSL\ElderNet\data\RUSH\data'
TEST_RATIO = 0.2

file_list = pd.Series(name='file_list', dtype=object)

def time_synch(StartTime, fs):
    """
    Each participant has different start time of the recording. When we analyze the data we want to have shared
    start point, begin in the midnight of the first recording day.

    :param StartTime: string. Starting time of the recording. In the format of Hour:Minute:Second:Millisecond
    :param fs: sampling rate

    :return: TimeDiffSamples - the sample number indicate midnight of the first recording day
    """

    # Parse input time string
    StartTimeDateTime = datetime.strptime(StartTime[:8], '%H:%M:%S')
    # Get the date of the input start time
    start_date = StartTimeDateTime.date()
    # Define the datetime representation of the midnight of the day after the input start day
    midnight = datetime.combine(start_date + timedelta(days=1), datetime.min.time())
    # Calculate time difference
    TimeDiff = midnight - StartTimeDateTime
    # Calculate time difference in seconds and convert to samples
    TimeDiffSamples = TimeDiff.total_seconds() * fs

    return TimeDiffSamples

# Define a worker function to process a single file
def process_file(min_data, window_step_len, window_len, out_dir, file):
    try:
        if os.path.isfile(file):
            values = loadmat(file)
            values = values['values'][0,0]
            startTime = values['startTime'][0]
            device_fs = values['sampFreq'][0]
            min_data = device_fs * min_data
            sync_start_point = time_synch(startTime, RESAMPLED_HZ)
            data = values['acc'].astype(float)
            if data.shape[0] < min_data:
                return None
            resampled_data = resample_data(data, device_fs, RESAMPLED_HZ)
            efficient_data = resampled_data.astype(np.float16)

            sub_win = np.empty((len(efficient_data) // window_step_len, window_len, 3))
            for i, start_idx in enumerate(range(0, (len(efficient_data) - window_step_len), window_step_len)):
                end_idx = start_idx + window_len
                sub_win[i, :, :] = efficient_data[start_idx:end_idx]
            std_win = np.std(sub_win, axis=1)
            sub_win = np.concatenate((sub_win, std_win[:, np.newaxis]), axis=1)
            #
            # # Extract subject id from the file's name
            file_name = file.split('\\')[-1]
            # file_name_components = file_name.split('-')
            sub_id = file_name.split('.')[0]
            # sub_id = file_name_components[0] + '_' + file_name_components[1]
            file2save = sub_id + '.npy'
            file_path = os.path.join(OUTDIR, file2save)
            np.save(os.path.join(out_dir, file2save), sub_win)
            # Inside the process_file function, replace the print statement with logging
            logging.info(f'subject {file2save} was processed')

            return sync_start_point, file_path


    except Exception as e:
        logging.error(f'Error processing file: {file}\nError message: {str(e)}')
        return None

# Define the main function to parallelize the loop
def process_files_parallel(chunk, min_data, window_step_len, window_len, out_dir):
    with multiprocessing.Pool() as pool:
        partial_process_file = partial(process_file, min_data, window_step_len, window_len, out_dir)
        results = pool.map(partial_process_file, chunk)
        start_time = [result[0] for result in results if result is not None]
        sub_id = [result[1] for result in results if result is not None]
    return (start_time, sub_id)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Split the file list into chunks to be processed by each worker
    chunk_size = CHUNK_SIZE  # Number of files to be processed by each worker
    files_names = [f for f in glob.glob(DATAFILES)]
    file_chunks = [files_names[i:i + chunk_size] for i in range(0, len(files_names), chunk_size)]

    # Process files in parallel
    processed_files = []
    start_point_list = []
    for chunk in tqdm(file_chunks):
        start_points, sub_ids = process_files_parallel(chunk, MIN_DATA, WINDOW_STEP_LEN, WINDOW_LEN, OUTDIR)
        start_point_list.extend(start_points)
        processed_files.extend(sub_ids)

    # Save the start point list
    point_list_df = pd.Series(start_point_list, name='start_point_list')
    point_list_df.to_csv(os.path.join(OUTDIR, 'start_point_list.csv'), index=False)

    # Split the files list to train and test lists
    sample_size = int(len(processed_files) * TEST_RATIO)
    test_files = random.sample(processed_files, sample_size)
    # Create new file lists for train and test
    train_files = [filename for filename in processed_files if filename not in test_files]
    train_file_list = pd.DataFrame({'file_list': train_files})
    test_file_list = pd.DataFrame({'file_list': test_files})
    # Save file lists
    train_file_list.to_csv(os.path.join(OUTDIR, 'train', 'file_list.csv'), index=False)
    test_file_list.to_csv(os.path.join(OUTDIR, 'test', 'file_list.csv'), index=False)
