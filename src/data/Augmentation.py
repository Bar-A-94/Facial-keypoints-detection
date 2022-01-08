from src.data.DataObj import FaceKeypointDataSet
import matplotlib.pyplot as plt
import numpy as np
import cv2

original_index = list(range(30))
flip_index = [2, 3, 0, 1, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 21, 24, 25, 22, 23, 26, 27, 28,
              29]


def create_aug_sets(data, aug_transformer, params, all_datasets):
    """
    create augmented data sets from data using aug_transformer and each parameter from params, add it to all_dataset
    :param data: data containing img as numpy and features
    :param aug_transformer: function on img
    :param params: parameter for the function
    :param all_datasets: list of previous data sets
    """
    aug_sets = []
    for param in params:
        aug_data = FaceKeypointDataSet(data, transformer=aug_transformer,
                                       transformer_factor=param)
        aug_sets.append(aug_data)
    # base_data_set = FaceKeypointDataSet(data)
    # show_aug(aug_sets, base_data_set)
    all_datasets += aug_sets


def show_img_and_features(data, img_index):
    """
    show the image indexed 'img_index' with its features on a graph
    """
    dat = data[img_index]
    img = dat[0].reshape(3, 96, 96)[1]
    plt.imshow(img, cmap='gray')
    plt.scatter(dat[1][::2], dat[1][1::2], marker='o', s=100)


def show_aug(datasets, original_train_data):
    """
    Show the difference between the augmented dataset to the original
    """
    fig = plt.figure(figsize=(10, 20))
    num_of_datasets = len(datasets)
    rand_img = np.random.randint(0, len(original_train_data))
    for index, aug_data in enumerate(datasets):
        fig.add_subplot(num_of_datasets, 2, (index + 1) * 2 - 1)
        show_img_and_features(original_train_data, rand_img)
        fig.add_subplot(num_of_datasets, 2, (index + 1) * 2)
        show_img_and_features(aug_data, rand_img)
    plt.show()


def flip_aug(img, fea, factor=None):
    """
    flip the img and the features
    :param img: array of 9,216 ints representing an image
    :param fea: features of original image
    :param factor:
    :return: flipped image
    """
    img = np.array(img)
    img = img[:, ::-1, :]
    fea = fea[flip_index]
    fea[::2] = 96 - fea[::2]
    img = img.astype(np.uint8)
    return img, fea


def noise_aug(img, fea, factor=None):
    """
    add noise to the img
    :param img: array of 9,216 ints representing an image
    :param fea: features of original image
    :param factor:
    :return: noisy image
    """
    img = np.array(img)
    img = img + 0.008 * np.random.randn(96, 96, 1)
    img = img.astype(np.uint8)
    return img, fea


def brightness_aug(img, fea, factor):
    """
    change the brightness of the img
    :param img: array of 9,216 ints representing an image
    :param fea: features of original image
    :param factor: brightness factor
    :return: image with different factor of brightness
    """

    img = np.array(img).astype(np.uint16)
    img = img + factor
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8).reshape(96, 96, 1)
    return img, fea


def rotate_aug(img, fea, factor=-30):
    """
    rotate the img and the features
    :param img: array of 9,216 ints representing an image
    :param fea: features of original image
    :param factor: rotate factor
    :return: rotated image by the factor
    """
    rad = -factor / 180 * np.pi
    rot = cv2.getRotationMatrix2D((48, 48), factor, 1)
    img = cv2.warpAffine(np.array(img).reshape(96, 96), rot, (96, 96), flags=cv2.INTER_CUBIC).reshape(96, 96, 1)
    fea -= 48
    for index in range(0, len(fea), 2):
        x = fea[index]
        y = fea[index + 1]
        fea[index] = x * np.cos(rad) - y * np.sin(rad)
        fea[index + 1] = x * np.sin(rad) + y * np.cos(rad)
    fea += 48
    img = img.astype(np.uint8)
    return img, fea


def create_full_augmentation(basic_data):
    """
    make a list of different augmentation of the basic data
    :param basic_data: data containing img as numpy and features
    :return: list of all augmented data
    """
    all_datasets = []
    transformer_params = [None]
    create_aug_sets(basic_data, flip_aug, transformer_params, all_datasets)
    create_aug_sets(basic_data, noise_aug, transformer_params, all_datasets)
    transformer_params = [70, -70, 120, -120]
    create_aug_sets(basic_data, brightness_aug, transformer_params, all_datasets)
    transformer_params = [30, -30, 15, -15]
    create_aug_sets(basic_data, rotate_aug, transformer_params, all_datasets)
    return all_datasets
