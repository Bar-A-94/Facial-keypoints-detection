import torch

import src.Models.predict as pred
from src.data import make_dataset
from src.utilities import utility


print('Biggest mistakes')
utility.visualize_examples('biggest mistakes', *utility.show_error(pred.model, make_dataset.val_loader, 16))

print('Model results on the train set')
images, labels = next(iter(make_dataset.train_loader))
with torch.no_grad():
    pred.model.eval()
    predicted_labels = pred.model(images)
utility.visualize_examples('Train set', images, labels, predicted_labels)

print('Model results on the validation set')
images, labels = next(iter(make_dataset.val_loader))
with torch.no_grad():
    pred.model.eval()
    predicted_labels = pred.model(images)
utility.visualize_examples('Validation set', images, labels, predicted_labels)

print('Model results on the test set')
images = next(iter(pred.test_loader))
with torch.no_grad():
    pred.model.eval()
    predicted_labels = pred.model(images)
utility.visualize_examples('Test set', images, predicted_labels, None)




