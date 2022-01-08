from src.data.Augmentation import flip_aug, noise_aug, brightness_aug, rotate_aug
from src.data.make_dataset import split_data_train_val
from src.Models.Adjusted_models import resnet50, efficientb4
import torch
import numpy as np
import os
import pandas as pd


def test_aug_type_and_shape(func):
    img = torch.zeros(96, 96, 1)
    labels = np.zeros(30)
    img, labels = func(img, labels, 1)
    assert (img.dtype == np.uint8 and img.shape == (96, 96, 1)), "error in aug function - image shape/dtype " \
                                                                 + func.__name__
    assert (isinstance(labels, np.ndarray) and len(labels) == 30), "error in aug function - labels shape/type " \
                                                                   + func.__name__
    print("Test passed for " + func.__name__)


def test_split(func, data, split):
    train, val = func(data, split)
    assert train.shape[1] == val.shape[1] and train.shape[1] == data.shape[1], "spilt damaged the shape of the data"
    assert len(train) + len(val) == len(data), "split decrease/increased the length"
    assert np.round(len(train) / len(data), 2) == split, "split rate has done something weird"
    print("Test passed for " + func.__name__)


def test_model(model):
    img = torch.zeros(1, 3, 96, 96)
    new = model
    new.eval()
    assert new(img).shape == (1, 30), "wrong output/input shape"
    print("Test passed for " + model.__name__)


main_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_directory = os.path.join(main_directory, 'data')
train_and_val_data = pd.read_csv(os.path.join(data_directory, 'interim', 'training.csv'))
full_only = train_and_val_data.dropna()
for f in [flip_aug, noise_aug, brightness_aug, rotate_aug]:
    test_aug_type_and_shape(f)
test_split(split_data_train_val, full_only, 0.9)
test_model(resnet50)
test_model(efficientb4)
try:
    sub = pd.read_csv(os.path.join(main_directory, 'submission.csv'))
    assert len(sub) == 27124, "check the submission file length"
    assert (sub.values[:, 1] > 0).any() and (sub.values[:, 1] < 96).any(), "some predictions are out of range(0,96)"
    print("Test passed for submission file")
except FileNotFoundError:
    print("Finished without submission check")
