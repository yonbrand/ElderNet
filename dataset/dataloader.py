import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from scipy import signal
from scipy.signal import filtfilt
from dataset import transformations
import constants

np.random.seed(42)


def bandpass_filter(data, cfg):
    """Apply a band-pass filter to the input data.
    :param data: NumPy array of shape (n_points, 3)
    :param low_cut: Lower frequency cutoff (Hz)
    :param high_cut: Upper frequency cutoff (Hz)
    :param sampling_rate: Sampling frequency (Hz)
    :param order: Order of the Butterworth filter

    :return: Filtered data
    """
    nyq = 0.5 * cfg.dataloader.sampling_rate
    low = cfg.dataloader.low_cut / nyq
    high = cfg.dataloader.high_cut / nyq
    b, a = signal.butter(cfg.dataloader.order, [low, high], btype='bandpass')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def reshape_data(data):
    """
    Flat the data to be in the shape of (n,3)
    :param data: data in the shape of (n, window_size+1, 3)
    :return: the reshaped_data + vector of the std for each window
    """
    data_new = data[:, :-1, :]  # remove the std
    data_std = data[:, -1, np.newaxis]  # add a new axis
    data_flat = data_new.reshape(-1, 3)
    return data_flat, data_std

def standardize(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    acc_standardized = (data - mean) / std
    return acc_standardized

def convert_y_label(batch, label_pos):
    row_y = [item[1 + label_pos] for item in batch]
    master_y = torch.cat(row_y)
    final_y = master_y.long()
    return final_y

def subject_collate(batch):
    # Filter out None values
    data = [item[0] for item in batch if item is not None]
    pid = [item[1] for item in batch if item is not None]

    # Check if data is not empty before using torch.cat
    if len(data) == 0:
        return None, None

    return torch.cat(data), pid

def subject_collate_simclr(batch):
    return torch.cat(batch)

def subject_collate_mtl(batch):
    data = [item[0] for item in batch]
    data = torch.cat(data)

    aot_y = convert_y_label(batch, constants.TIME_REVERSAL_POS)
    scale_y = convert_y_label(batch, constants.SCALE_POS)
    permutation_y = convert_y_label(batch, constants.PERMUTATION_POS)
    time_w_y = convert_y_label(batch, constants.TIME_WARPED_POS)
    return [data, aot_y, scale_y, permutation_y, time_w_y]

def worker_init_fn(worker_id):
    np.random.seed(int(time.time()))

def generate_labels(X, shuffle, cfg):
    labels = []
    new_X = []
    for i in range(len(X)):
        current_x = X[i, :, :]

        current_label = [0, 0, 0, 0]
        if cfg.task.time_reversal:
            choice = np.random.choice(
                2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
            )[0]
            current_x = transformations.flip(current_x, choice)
            current_label[constants.TIME_REVERSAL_POS] = choice

        if cfg.task.scale:
            choice = np.random.choice(
                2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
            )[0]
            current_x = transformations.scale(current_x, choice)
            current_label[constants.SCALE_POS] = choice

        if cfg.task.permutation:
            choice = np.random.choice(
                2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
            )[0]
            current_x = transformations.permute(current_x, choice)
            current_label[constants.PERMUTATION_POS] = choice

        if cfg.task.time_warped:
            choice = np.random.choice(
                2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
            )[0]
            current_x = transformations.time_warp(current_x, choice)
            current_label[constants.TIME_WARPED_POS] = choice

        new_X.append(current_x)
        labels.append(current_label)

    new_X = np.array(new_X)
    labels = np.array(labels)
    if shuffle:
        feature_size = new_X.shape[-1]
        new_X = np.concatenate([new_X, labels], axis=2)
        np.random.shuffle(new_X)

        labels = new_X[:, :, feature_size:]
        new_X = new_X[:, :, :feature_size]

    new_X = torch.Tensor(new_X)
    labels = torch.Tensor(labels)
    return new_X, labels

def weighted_epoch_sample(data_with_std, num_sample=400):
    """
    Weighted sample the windows that have most motion
    Args:
        data_with_std (np_array) of shape N x 3 x 301:
         last ele is the std per sec. We
        assume the sampling rate is 30hz.
        num_sample (int): windows to sample per subject
        epoch_len (int): how long should each epoch last in sec
        sample_rate (float): Sample frequency
        is_weighted_sample (boolean): random sampling if false
    Returns:
        sampled_data : high motion windows of size num_sample x 3 x 300
    """
    ori_data = data_with_std[:, :, :300]
    data_std = data_with_std[:, :, -1][:, 0]

    # To prevent cases when there are too many windows with std=0 (which raise error in the sampling process)
    zero_std_idx = np.where(data_std == 0)[0]
    data_std[zero_std_idx] = 1e-6
    assert np.sum(data_std > 0) > num_sample , "Fewer samples with std > 0"

    np.random.seed(42)
    sample_ides = np.random.choice(
        len(ori_data), num_sample, replace=False, p=data_std / np.sum(data_std)
    )

    sampled_data = np.zeros([num_sample, 3, 300])
    for ii in range(num_sample):
        idx = sample_ides[ii]
        sampled_data[ii, :] = ori_data[idx, :, :]
    return sampled_data

def remove_non_wear_time(acceleration_data, window_size=(60 * 30 * 30), threshold_std=0.01):
    # Calculate the number of samples and segments
    n_samples = acceleration_data.shape[0]
    n_segments = n_samples // window_size

    # Take only the portion that is divisible by the window size
    acceleration_data = acceleration_data[: n_segments * window_size]

    # Initialize a mask for each axis
    mask_all_axes = np.zeros((n_segments,))

    # Iterate over the 3 acceleration axes and find the std in windows of size window_size
    for axis in range(3):
        current_data = acceleration_data[:, axis]
        data_reshaped = current_data.reshape((n_segments, window_size))

        # Calculate the standard deviation in each window
        std_in_windows = np.std(data_reshaped, axis=1)

        # Create a mask for windows with standard deviation greater than threshold
        mask = std_in_windows > threshold_std
        mask_all_axes = mask_all_axes + mask

    # If any axis has data exceeding the threshold in a window, mark it
    mask_all_axes = mask_all_axes > 0

    # Reshape the mask from windows to number of samples
    final_mask = np.repeat(mask_all_axes, window_size)

    # Apply the final mask to the acceleration data
    return acceleration_data[final_mask]



class SSL_dataset:
    def __init__(
            self,
            data_root,
            file_list_path,
            cfg,
            transform=None,
            shuffle=False,
    ):
        """
        Args:
            data_root (string): directory containing all data files
            file_list_path (string): file list
            cfg (dict): config
            params: dictionary of hyperparameters
            shuffle (bool): whether permute subject data


        Returns:
            data : transformed sample
            labels (dict) : labels for avalaible transformations
        """

        file_list_df = pd.read_csv(file_list_path)
        self.file_list = file_list_df["file_list"].to_list()
        self.data_root = data_root
        self.cfg = cfg
        self.transform = transform
        self.shuffle = shuffle
        self.window_size = cfg.dataloader.epoch_len * cfg.dataloader.sampling_rate

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # idx starts from zero
        file_to_load = self.file_list[idx]
        X = np.load(file_to_load, allow_pickle=True)

        # Preprocess the data
        if self.cfg.dataloader.bandpass_filtering or self.cfg.dataloader.standardize:
            num_windows = X.shape[0]
            # Reshape the data to be (n,3)
            X, X_std = reshape_data(X)
            if self.cfg.dataloader.bandpass_filtering:
                X = bandpass_filter(X, self.cfg)
            if self.cfg.dataloader.standardize:
                X = standardize(X)
            X = X.reshape(num_windows, self.window_size, 3)
            X = np.concatenate((X, X_std), axis=1)

        # transpose axes if ordered different from the original paper
        if X.shape[1] != 3:
            X = np.transpose(X, (0, 2, 1))
        # sample windows according to the std of the windows
        X = weighted_epoch_sample(X, num_sample=self.cfg.dataloader.num_samples)

        if self.cfg.model.ssl_method == 'mtl':
            X, labels = generate_labels(X, self.shuffle, self.cfg)

            if self.transform:
                X = self.transform(X)

            return (
                X,
                labels[:, constants.TIME_REVERSAL_POS],
                labels[:, constants.SCALE_POS],
                labels[:, constants.PERMUTATION_POS],
                labels[:, constants.TIME_WARPED_POS],
            )

        elif self.cfg.model.ssl_method == 'SimCLR':
            X = np.array(X)
            X = torch.Tensor(X)
            # Apply the transformation twice to create two augmented views
            view1 = self.transform(X)
            view2 = self.transform(X)
            # Stack the two views to create a single sample with 2 views
            X = torch.stack([view1, view2])
            return X


class FT_Dataset(Dataset):
    def __init__(self,
                 X,
                 y=None,
                 name="",
                 cfg=None,
                 transform=None,
                 verbose=True):

        self.X = X
        self.y = y
        self.cfg = cfg
        self.transform = transform

        if verbose:
            print(f"{name} set sample count: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]
        y = self.y[idx]
        sample = torch.Tensor(sample)

        if self.transform is not None:
            sample = self.transform(sample)


        return sample, y


