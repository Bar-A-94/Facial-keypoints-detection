import matplotlib.pyplot as plt
import numpy as np
import os
import torch

main_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def visualize_examples(headline, images, features, pred_labels=None):
    """
    Make 16 images and labels examples from the input,
    if the input includes predicted labels - show them with x marks
    """
    fig = plt.figure(figsize=(20, 10))
    for i in range(16):
        fig.add_subplot(4, 4, i + 1)
        img = images[i].reshape(3, 224, 224)[0]
        plt.imshow(img.reshape(224, 224), cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.scatter(features[i][::2], features[i][1::2], marker='o', s=50)
        if pred_labels is not None:
            plt.scatter(pred_labels[i][::2], pred_labels[i][1::2], marker='x', color='red', s=50)
    fig.suptitle(headline, fontsize='xx-large')
    plt.savefig(os.path.join(main_directory, 'src', 'Models', 'log', headline + '.png'))
    plt.show()


def text_img_to_numpy(data):
    """
    change the Image column from string to numpy array
    :param data: dataframe with Image column
    """
    imgs = data.Image
    imgs = np.array(imgs)
    for j in range(len(imgs)):
        imgs[j] = np.fromstring(imgs[j], sep=' ')
    data.Image = imgs


def RMSELoss_custom_per_image(pred, y):
    """
    pred: the prediction of the model
    y: the true labels
    return: root-mean-square error without considering the NaN labels
                - each NaN's loss is equal to 0, so it won't affect the total loss
                - to scale the loss per not-NaN value, we divide it by the not_nan amount and multiply by the total
                  predictions
    """
    not_nan = 30 - y.isnan().sum()
    return torch.sqrt(torch.mean((pred - y).nan_to_num() ** 2)) * 30 / not_nan


def show_error(model, loader, num_of_examples):
    with torch.no_grad():
        model.eval()
        losses = [np.inf * -1]
        im = ['im']
        la = ['la']
        pr = ['pr']
        criterion = RMSELoss_custom_per_image
        for images, labels in loader:
            predicted_labels = model(images)
            for i in range(len(images)):
                loss = float(criterion(predicted_labels[i], labels[i]))
                for j in range(min(len(losses), num_of_examples)):
                    if loss > losses[j]:
                        losses = losses[0:j] + [loss] + losses[j:num_of_examples]
                        im = im[0:j] + [images[i]] + im[j:num_of_examples]
                        la = la[0:j] + [labels[i]] + la[j:num_of_examples]
                        pr = pr[0:j] + [predicted_labels[i]] + pr[j:num_of_examples]
                        break
    print("Losses = " + str(losses[0:16]))
    return im[0:len(im) - 1], la[0:len(im) - 1], pr[0:len(im) - 1]
