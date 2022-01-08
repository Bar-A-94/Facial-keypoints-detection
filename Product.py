import cv2
import numpy as np
import torch
import os
from src.Models.Adjusted_models import resnet50, efficientb4
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import time


def new_image():
    image = cv2.imread('input_image.jpg')
    # turn image from rgb to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # crop the center of the image
    H, W = gray_image.shape
    H_start = max(0, ((H - W) // 2))
    H_end = H_start + min(H, W)
    W_start = max(0, ((W - H) // 2))
    W_end = W_start + min(H, W)
    gray_image = gray_image[H_start:H_end, W_start:W_end]
    return cv2.resize(gray_image, (96, 96), interpolation=cv2.INTER_LINEAR)


def given_image(index):
    data = pd.read_csv(os.path.join(data_directory, 'interim', 'test.csv')).head(index + 1)
    imgs = data.Image
    imgs = np.array(imgs)
    for j in range(len(imgs)):
        imgs[j] = np.fromstring(imgs[j], sep=' ')
    return imgs[index]


main_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(main_directory, 'data')
start = time.perf_counter()
if os.path.exists(os.path.join(main_directory, 'input_image.jpg')):
    gray = new_image()
else:
    gray = given_image(25)

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])

gray = torch.tensor(gray, dtype=torch.float)
gray = gray.reshape(1, 96, 96)
gray_norm = transform(gray)
gray_norm = gray_norm.reshape(1, 1, 96, 96)
print('Redesigned the image for the model, Time: ' + str(round(time.perf_counter() - start, 3)))
model = resnet50
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
checkpoint = torch.load(os.path.join(main_directory, 'Models', model.__name__, model.__name__ + '_model.pt'),
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
print('Model loaded, Time: ' + str(round(time.perf_counter() - start, 3)))

with torch.no_grad():
    model.eval()
    predicted_labels = model(gray_norm)
print('Labeled the image, Time: ' + str(round(time.perf_counter() - start, 3)))

fig = plt.figure(figsize=(20, 10))
plt.imshow(gray.reshape(96, 96), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.scatter(predicted_labels[0][::2], predicted_labels[0][1::2], marker='x', color='red', s=200)
plt.show()
