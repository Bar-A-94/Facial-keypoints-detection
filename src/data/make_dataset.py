from zipfile import ZipFile
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import SequentialSampler, SubsetRandomSampler
from src.data.Augmentation import create_full_augmentation
from src.data.DataObj import FaceKeypointDataSet
from src.utilities.utility import text_img_to_numpy

main_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_directory = os.path.join(main_directory, 'data')


def extract_file_from_raw():
    """
    extract the zipfile file from the raw data into interim folder
   """
    with ZipFile(os.path.join(data_directory, 'raw', 'training.zip'), 'r') as zipfile:
        zipfile.extractall(os.path.join(os.path.join(data_directory, 'interim')))
    with ZipFile(os.path.join(data_directory, 'raw', 'test.zip')) as zipfile:
        zipfile.extractall(os.path.join(data_directory, 'interim'))


def split_data_train_val(data, split_rate=0.65):
    """
    split the data into train data and validation data by split rate
    :param data: dataframe
    :param split_rate: float
    :return: train data and validation data
    """
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    split = int(np.floor((1 - split_rate) * len(full_only)))
    train_indices, val_indices = indices[split:], indices[:split]
    train_data = data.iloc[train_indices]
    val_data = data.iloc[val_indices]
    return train_data, val_data


extract_file_from_raw()
train_and_val_data = pd.read_csv(os.path.join(data_directory, 'interim', 'training.csv'))
text_img_to_numpy(train_and_val_data)
test_csv = pd.read_csv(os.path.join(data_directory, 'interim', 'test.csv'))
text_img_to_numpy(test_csv)
full_only = train_and_val_data.dropna()
missing_only = train_and_val_data[train_and_val_data.isna().sum(axis=1) != 0]

all_datasets = []
aug_set = create_full_augmentation(full_only)
all_datasets += aug_set

full_only_train_data, full_only_val_data = split_data_train_val(full_only)

missing_dataset = FaceKeypointDataSet(missing_only, transformer=None)
full_only_train_dataset = FaceKeypointDataSet(full_only_train_data, transformer=None)
val_dataset = FaceKeypointDataSet(full_only_val_data, transformer=None)

all_datasets += [missing_dataset]
all_datasets += [full_only_train_dataset]
train_dataset = torch.utils.data.ConcatDataset(all_datasets)

val_sampler = SequentialSampler(range(len(val_dataset)))
train_sampler = SubsetRandomSampler(range(len(train_dataset)))
if torch.cuda.is_available():
    batch_size = 128
else:
    batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

