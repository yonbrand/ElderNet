import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve



def majority_vote(preds, window_size, overlap, fs, num_win_sub):

    new_preds = np.empty(0)
    num_sec_win = int(window_size / fs) # Num of seconds in each window
    num_win_sub = np.asarray(num_win_sub)
    end_points_win = np.cumsum(num_win_sub.astype(int))
    for sub in range(len(num_win_sub)):
        if sub==0:
            current_preds = preds[0 : end_points_win[sub]]
        else:
            current_preds = preds[end_points_win[sub - 1]: end_points_win[sub]]

        # Create an array with the predictions for each second
        # Each row represent one second and each column represent a window in which this second was appeared
        sliding_preds = sliding_window_view(current_preds, num_sec_win, 0)
        # For each second - sum the predictions of this sec
        sum_votes = np.mean(sliding_preds, axis=1)
        # Moving from second level to sample level
        sample_preds= np.repeat(sum_votes, fs)
        # For each subject, the first seconds has not enough overlap data for majority vote, so we considered them as 0
        sample_preds = np.pad(sample_preds, [overlap, 0])
        # Concatenate the predictions of each subject together
        new_preds = np.append(new_preds, sample_preds)

    return new_preds

def get_classification_threshold(labels, preds):
    """Plot the ROC and precision-recall (PR) curves.

           Parameters:
           - labels: Numpy array of shape (n_points,)
           - preds: Numpy array of shape (n_points,)

           Returns:
           Classification threshold which maximize the F1score
           """

    # Calculate the PR curve
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    # Find intersection between precision, recall and f1
    intersection = np.where(precision > recall)[0][0]

    return thresholds[intersection]

def plot_curves_for_seeds(seeds, metric_arrays, model_path, curve_type):
    plt.figure(figsize=(10, 6))

    for seed in range(len(seeds)):
        curve_values = metric_arrays[curve_type][seed]
        if curve_type == 'pr' or curve_type == 'roc':
            plt.plot(curve_values[0], curve_values[1], label=f"Seed: {seed}, AUC: {curve_values[2]}")
        elif curve_type == 'performance':
            # Find intersection between precision, recall and f1
            thresholds = curve_values[0]
            precision = curve_values[1]
            recall =curve_values[2]
            f1 = curve_values[3]
            intersection = curve_values[4]
            plt.plot(thresholds, precision[:-1], label=f"Precision: Seed {seed}")
            plt.plot(thresholds, recall[:-1], label=f"Recall: Seed {seed}")
            plt.plot(thresholds, f1[:-1], label=f"F1: Seed {seed}")
            plt.axvline(x=thresholds[intersection], color="red", linestyle="--")
    if curve_type == 'pr':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curves for Different Seeds')
    elif curve_type == 'roc':
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Seeds')
    elif curve_type == 'performance':
        plt.xlabel('Thresholds')
        plt.ylabel('Performance')
        plt.title('Performance  Curves for Different Seeds')

    plt.legend(loc="lower left")
    plt.savefig(os.path.join(model_path, f'{curve_type.capitalize()}_Curves_All_Seeds.jpeg'), dpi=300)
    plt.close()

def check_performance_post(labels, predictions):
    """Check the performance of the model after the post-processing stage.

           Parameters:
           - labels: Numpy array of shape (n_points)
           - predictions: Numpy array of shape (n_points)

           Returns:
           - accuracy, specificity, recall, precision, F1Score
           """
    tp = np.sum(((predictions == labels) & (labels == 1)))
    tn = np.sum(((predictions == labels) & (labels == 0)))
    fp = np.sum(((predictions != labels) & (labels == 0)))
    fn = np.sum(((predictions != labels) & (labels == 1)))

    accuracy = 100 * ((tp + tn) / (tp + tn + fp + fn))
    specificity = 100 * (tn / (1 + tn + fp))
    recall = 100 * ((1+tp) / (1 + tp + fn))
    precision = 100 * ((1+tp) / (1 + tp + fp))
    F1Score = 2 * (precision * recall) / (precision + recall)

    return np.round(accuracy, 2), np.round(specificity, 2), np.round(recall, 2), np.round(precision, 2), np.round(
        F1Score, 2)

def post_processing(arr, merge_distance, fs, min_bout):
    '''
          Merge neighboring gait segments and remove bouts that are less than the minimal bout duration that defined.

           Parameters:
           - arr: Numpy array of shape (n_points), labels or prediction vector
           - merge_distance: Scalar, merge gait bouts with distance lower than merge_distance (in seconds)
           - fs: Scalar, the sampling rate.
           - min_bout: Scalar, minimal bout duration to be considered as gait (in seconds)

           Returns:
           - New smoothed labels/predictions vector of shape (n_points)
    '''

    # Buffer arr with '0' in the start and the end of the vector
    buffered_arr = np.insert(arr, 0, 0)
    buffered_arr = np.insert(buffered_arr, len(buffered_arr), 0)
    # Find the indices of the predicted gait bouts
    gait_bouts = np.where(np.diff(buffered_arr))[0]
    # Concatenate the start and end points of each bout
    GaitBoutsStartStop = np.vstack((gait_bouts[::2], gait_bouts[1::2]))

    new_arr = buffered_arr.copy()
    for bout in range(1, GaitBoutsStartStop.shape[1]):
        # Merge bouts with interval less than 1 sec
        if GaitBoutsStartStop[0, bout] - GaitBoutsStartStop[1, bout - 1] < (merge_distance * fs):
            new_arr[GaitBoutsStartStop[1, bout - 1]:GaitBoutsStartStop[0, bout]] = 1

    # Find the indices of the new gait bouts
    gait_bouts_new = np.where(np.diff(new_arr))[0]
    # Concatenate the start and end points of each bout
    GaitBoutsStartStopNew = np.vstack((gait_bouts_new[::2], gait_bouts_new[1::2]))
    # Durations of the different gait bouts
    ranges = GaitBoutsStartStopNew[1, :] - GaitBoutsStartStopNew[0, :]
    # Remove gait bouts with durations that are less than min_bout
    GaitBoutsStartStopNew = np.delete(GaitBoutsStartStopNew, ranges < min_bout * fs, 1)
    final_arr = np.zeros_like(arr)
    for bout in range(GaitBoutsStartStopNew.shape[1]):
        final_arr[GaitBoutsStartStopNew[0, bout]:GaitBoutsStartStopNew[1, bout]] = 1
    return final_arr



