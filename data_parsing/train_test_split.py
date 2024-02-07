import os
import random
import shutil
import pandas as pd

# Define paths
train_folder = r'N:\Gait-Neurodynamics by Names\Yonatan\SSL\ElderNet\data\RUSH\data\train'
test_folder = r'N:\Gait-Neurodynamics by Names\Yonatan\SSL\ElderNet\data\RUSH\data\test'
file_list_csv = r'N:\Gait-Neurodynamics by Names\Yonatan\SSL\ElderNet\data\RUSH\data\file_list.csv'
TEST_RATIO = 0.15

# Create test folder if it doesn't exist
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Read file list
file_list = pd.read_csv(file_list_csv)

# Get the number of files to sample
sample_size = int(len(file_list) * TEST_RATIO)

# Randomly select files for testing
test_files = random.sample(file_list['file_list'].tolist(), sample_size)

# Move test files to the test folder
for filename in test_files:
    src = os.path.join(train_folder, filename)
    dst = os.path.join(test_folder, filename)
    shutil.move(src, dst)

# Create new file lists for train and test
train_files = [filename for filename in file_list['file_list'] if filename not in test_files]
train_file_list = pd.DataFrame({'file_list': train_files})
test_file_list = pd.DataFrame({'file_list': test_files})

# Save file lists
train_file_list.to_csv(os.path.join(train_folder, 'file_list.csv'), index=False)
test_file_list.to_csv(os.path.join(test_folder, 'file_list.csv'), index=False)
