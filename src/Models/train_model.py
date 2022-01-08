import torch
import src.data.make_dataset as dataset_py
import Adjusted_models as models
import time
from pathlib import Path
from src.utilities.logger import set_logger


def RMSELoss(pred, y):
    return torch.sqrt(torch.mean((pred - y) ** 2))


def RMSELoss_custom(pred, y):
    """
    :param pred: the prediction of the model
    :param y: the true labels
    :return: root-mean-square error without considering the NaN labels
                - each NaN's loss is equal to 0 so it won't affect the total loss
                - to scale the loss per not-NaN value, we divide it by the not_nan amount and multiply by the total
                  predictions
    """
    not_nan = (dataset_py.batch_size * 30 - y.isnan().sum())
    return torch.sqrt(torch.mean((pred - y).nan_to_num() ** 2)) * dataset_py.batch_size * 30 / not_nan


epochs = 150
model = models.inception_v3
model_name = model.__name__
logger = set_logger("./log/" + model_name + "_training.log")
model_dir = Path("./pt/" + model_name + "_model.pt")
learning_rate = 0.01
criterion = RMSELoss_custom
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
gpu = torch.cuda.is_available()
if gpu:
    torch.cuda.empty_cache()
    model.cuda()
train_losses, val_losses = [], []
val_loss_min = torch.inf
logger.debug(f"-------------------------Train-------------------------")
print("Started training")
for e in range(1, epochs + 1):
    start = time.perf_counter()
    model.train()
    train_loss = 0
    for images, labels in dataset_py.train_loader:
        if gpu:
            images = images.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        prediction = model(images)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0
    with torch.no_grad():
        model.eval()
        for images, labels in dataset_py.val_loader:
            if gpu:
                images = images.cuda()
                labels = labels.cuda()
            prediction = model(images)
            loss = criterion(prediction, labels)
            val_loss += loss.item()
        scheduler.step(val_loss)
    train_losses.append(train_loss / len(dataset_py.train_loader))
    val_losses.append(val_loss / len(dataset_py.val_loader))
    logger.info("Epoch: {}/{} ".format(e, epochs) +
                "Training Loss: {:.4f} ".format(train_losses[-1]) +
                "Val Loss: {:.4f}".format(val_losses[-1]))
    if val_loss < val_loss_min:
        val_loss_min = val_loss
        torch.save(model.state_dict(), model_dir)
        logger.info('---Detected network improvement, saving current model--')
    end = time.perf_counter()
    total = (end - start) * (epochs - e)
    logger.info('----------------Estimated time: {:d}:{:d}:{:d}----------------'.format(int(total // 3600),
                                                                                        int(total % 3600 // 60),
                                                                                        int(total % 60)))
print('Done training!')
logger.info('Done training!')
