import os
import hydra
import pickle
import numpy as np
import torch

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 1000000000
mpl.use('TkAgg')

from torchvision import transforms

from fine_tuning import GaitDetectorSSL
import postprocessing
from postprocessing import check_performance_post
import constants
from data_loader.transformations import RotationAxis, RandomSwitchAxis

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score

from datetime import datetime

now = datetime.now()

import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(my_seed=0, device='cuda'):
    random_seed = my_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_seed)


# def get_execution_arguments():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("-config", "--configuration",
#                         help="running configuration type - initialize \ preparation \ modeling \ postprocessing ")
#     parser.add_argument("-input_path", "--input_path", help="Path for loading input files")
#     parser.add_argument("-model_path", "--model_path", help="Path for loading the trained model")
#     parser.add_argument("-output_path", "--output_path", help="Path for storing processing and modeling related files")
#     parser.add_argument("-run_mode", "--run_mode", help="CV/train/test")
#     parser.add_argument("-net", "--net", help="Name of the network architecture: Resnet/Unet")
#     args = parser.parse_args()
#
#     print(
#         "configuration: {}\n input_path: {}\n  model_path: {}\n output_path: {}\n run_mode: {}\n network: {}\n".format(
#             args.configuration,
#             cfg.data.data_root,
#             args.model_path,
#             output_path,
#             args.run_mode,
#             args.net
#         ))
#
#     return args


def create_folders(path):
    '''
    Create new folders in which the new output files will be stored.
    :param path: input path of the current project (e.g., 'Mobilise-D')
    :return: folder to save the output files
    '''
    if not os.path.isdir(os.path.join(path)):
        os.mkdir(os.path.join(path))


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def objective(trial, input_path, output_path, device, model_path):
    params = {
        'batch_size': trial.suggest_int('batch_size', 100, 100),
        'num_epochs': trial.suggest_int('num_epochs', 100, 100),
        'patience': trial.suggest_int('patience', 5, 5),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-4),
        'n_layers': trial.suggest_int('n_layers', 1, 6)
    }

    # Load the data (resample to 30 hz and divided into windows of 10 sec)
    X_train = pickle.load(open(os.path.join(input_path, 'WindowsData.p'), 'rb'))
    y_train = pickle.load(open(os.path.join(input_path, 'WindowsLabels.p'), 'rb'))
    groups_train = pickle.load(open(os.path.join(input_path, 'WindowsSubjects.p'), 'rb'))

    # Convert labels to one-hot encoded array
    one_hot_labels = np.zeros((len(y_train), 2), dtype=int)
    one_hot_labels[np.arange(len(y_train)), y_train.squeeze().astype(int)] = 1

    weight_path = os.path.join(output_path, 'weights.pt')
    gd = GaitDetectorSSL(weights_path=weight_path, device=device, verbose=True, model_path=model_path)
    f1_score = gd.cross_val(X_train, one_hot_labels, params, groups_train, return_f1=True)
    return f1_score


def objective(trial, cfg):
    lr_choices = [1e-1, 1e-2, 1e-3, 1e-4]
    params = {
        'num_epochs': trial.suggest_int('num_epochs', 10, 100),
        'lr': trial.suggest_categorical('lr', lr_choices),
        'num_subjects': 4,  # trial.suggest_int('num_subjects', 3,8),
        'num_samples': 1500,  # 100 * trial.suggest_int('num_samples', 5,25),
        'patience': 5  # trial.suggest_int('num_samples', 5, 10)
    }
    best_f1 = main_worker(params, cfg)

    return best_f1


@hydra.main(config_path="conf", config_name="config_ft", version_base='1.1')
def main(cfg):
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial,cfg) , n_trials=10)

    params = {
        'num_epochs_ft': 30,  # trial.suggest_int('num_epochs', 10, 100),
        'lr_ft': 1e-4,  # trial.suggest_categorical('lr', lr_choices),
        'batch_size_ft': 100,
        'patience': 5,
        'num_layers': 3,
        'non_linearity': True
    }

    main_worker(params, cfg)


def main_worker(params, cfg):
    # args = get_execution_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = params['num_epochs_ft']
    lr = params['lr_ft']
    batch_size = params['batch_size_ft']
    window_size = cfg.dataloader.epoch_len * cfg.dataloader.sample_rate
    switch_aug = cfg.augmentation.axis_switch
    rotation_aug = cfg.augmentation.rotation
    if switch_aug and rotation_aug:
        my_transform = transforms.Compose(
            [RandomSwitchAxis(), RotationAxis()]
        )
    elif switch_aug:
        my_transform = RandomSwitchAxis()
    elif rotation_aug:
        my_transform = RotationAxis()
    else:
        my_transform = None

    log_dir = cfg.data.log_path
    dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
    # Directory for the SSL model used for FT
    log_ssl_dir = os.path.join(log_dir, cfg.model.output_folder)

    if not os.path.isdir(log_ssl_dir):
        os.mkdir(log_ssl_dir)

    output_path = os.path.join(
        log_ssl_dir,
        # "FT_"
        # + 'fclayers_'
        # + str(cfg.model.num_layers)
        # + '_nonlinearity_'
        # + str(cfg.model.non_linearity)+
        "_lr_"
        + str(lr)
        + 'n_epochs_'
        + str(num_epochs)
        + '_batch_size_'
        + str(batch_size)
        + '_'
        + dt_string
    )

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    seeds = [0, 42, 100]
    # List of performance metrics names
    performance_metrics_names = ['accuracy', 'specificity', 'recall', 'precision', 'F1score']
    # Initialize a dictionary to store the arrays
    performance_dict = {metric_name: np.zeros(len(seeds)) for metric_name in performance_metrics_names}
    #
    curves_arrays = {
        'roc': {},
        'pr': {},
        'performance': {}
    }
    for seed in range(len(seeds)):
        set_seed(my_seed=seeds[seed], device=device)
        weight_path = os.path.join(output_path, 'weights.pt')
        # Creating a gait detection (=gd) instance
        gd = GaitDetectorSSL(cfg=cfg, params=params, weights_path=weight_path,
                             device=device, verbose=True, n_jobs=1, transform=my_transform)
        # Load the data (resample to 30 hz and divided into windows of 10 sec)
        X_train = pickle.load(open(os.path.join(cfg.data.data_root, 'WindowsData.p'), 'rb'))
        y_train = pickle.load(open(os.path.join(cfg.data.data_root, 'WindowsLabels.p'), 'rb'))
        groups_train = pickle.load(open(os.path.join(cfg.data.data_root, 'WindowsSubjects.p'), 'rb'))

        # Convert labels to one-hot encoded array
        one_hot_labels = np.zeros((len(y_train), 2), dtype=int)
        one_hot_labels[np.arange(len(y_train)), y_train.squeeze().astype(int)] = 1
        model, labels, predictions = gd.cross_val(X_train, one_hot_labels, params, groups_train)

        # Save the trained model as the gd object
        with open(os.path.join(output_path, 'model' + str(seed) + '.p'), 'wb') as OutputFile:
            pickle.dump(model, OutputFile)
        # Save the labels and predictions
        with open(os.path.join(output_path, 'labels' + str(seed) + '.p'), 'wb') as OutputFile:
            pickle.dump(labels, OutputFile)
        with open(os.path.join(output_path, 'predictions' + str(seed) + '.p'), 'wb') as OutputFile:
            pickle.dump(predictions, OutputFile)

        # Load relevant data
        StdIndex = pickle.load(open(os.path.join(cfg.data.data_root, 'StdIndex.p'), 'rb'))
        inclusion_idx = pickle.load(open(os.path.join(cfg.data.data_root, 'InclusionIndex.p'), 'rb'))
        labels = pickle.load(open(os.path.join(cfg.data.data_root, 'WindowsLabels.p'), 'rb'))
        labels = labels.squeeze()  # Reshape to be in the same shape as the prediction vector
        subjects_win = pickle.load(open(os.path.join(cfg.data.data_root, 'WindowsSubjects.p'), 'rb'))
        all_subjects = pickle.load(open(os.path.join(cfg.data.data_root, 'ResampledSubjects.p'), 'rb'))

        # Sort the labels and the subjects according to the test indices
        test_indices = model.cv_results['test_indices']
        test_indices = np.concatenate(test_indices)
        labels = labels[test_indices]

        # Rearrange the subjects according to the test indices
        subjects = np.repeat(subjects_win[test_indices], window_size)  # reshape from windows to samples
        all_subjects[inclusion_idx.astype(bool)] = subjects

        predictions = predictions[:, 1]

        # Convert from labels/predictions per window to sample-level prediction
        final_labels = postprocessing.reconstruct_windows(labels, window_size, inclusion_idx, StdIndex)
        final_labels = final_labels.squeeze()
        final_preds = postprocessing.reconstruct_windows(predictions, window_size, inclusion_idx, StdIndex)

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(final_labels, final_preds)
        roc_auc = roc_auc_score(final_labels, final_preds)  # area under the curve
        curves_arrays['roc'][seed] = (fpr, tpr, roc_auc)
        # Calculate the PR curve
        precision, recall, thresholds = precision_recall_curve(final_labels, final_preds)
        auprc = average_precision_score(final_labels, final_preds)  # area under the curve
        curves_arrays['pr'][seed] = (recall, precision, auprc)
        # Calculate the recall-precision-f1score curve
        eps = 1e-15
        f1 = (2 * precision * recall) / (precision + recall + eps)
        intersection = np.where(precision > recall)[0][0]
        curves_arrays['performance'][seed] = (thresholds, precision, recall, f1, intersection)

        if len(np.unique(final_preds)) > 2:
            classification_threshold = 0.5

            # Round predictions to get binary classification
            final_preds = np.where(final_preds > classification_threshold, 1, 0)

        # Postprocessing of the predictions
        # post_preds = postprocessing.post_processing(round_predictions, __MERGE_DISTANCE__, __SAMPLING_RATE__, __MIN_BOUT__)

        # Compute the performance's metrics after post-processing
        final_labels = final_labels.astype(final_preds.dtype)  # for reliable comparison
        performance_dict['accuracy'][seed], performance_dict['specificity'][seed], performance_dict['recall'][seed], \
        performance_dict['precision'][seed], performance_dict['F1score'][seed] = \
            check_performance_post(final_labels, final_preds)

    with open(os.path.join(output_path, 'performance_matrix.p'), 'wb') as OutputFile:
        pickle.dump(performance_dict, OutputFile)

    # Plot PR and ROC curves for all seeds
    postprocessing.plot_curves_for_seeds(seeds, curves_arrays, output_path, 'pr')
    postprocessing.plot_curves_for_seeds(seeds, curves_arrays, output_path, 'roc')
    postprocessing.plot_curves_for_seeds(seeds, curves_arrays, output_path, 'performance')

    return np.mean(performance_dict['F1score'])


if __name__ == '__main__':
    main()
