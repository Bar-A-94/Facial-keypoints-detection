import numpy as np
import torch
from torchvision import transforms


class FaceKeypointDataSet(torch.utils.data.Dataset):
    def __init__(self, data, transformer=None, transformer_factor=None, is_test_set=False):
        self.image_data = data.Image
        self.feature_data = data.drop(['Image'], axis=1)
        self.transformer = transformer
        self.transformer_factor = transformer_factor
        self.is_test_set = is_test_set
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224),
                                             transforms.Lambda(lambda x: np.stack((x,) * 3, axis=-1)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5,), std=(0.5,))])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):

        img = self.image_data.iloc[index]
        img = img.astype(np.uint8).reshape(96, 96, 1)
        feature = np.array(self.feature_data.iloc[index])
        if self.transformer is not None:
            img, feature = self.transformer(img, feature, self.transformer_factor)
        img = self.transform(img)
        if self.is_test_set:
            return img
        feature = feature * 224 / 96
        return img, feature
