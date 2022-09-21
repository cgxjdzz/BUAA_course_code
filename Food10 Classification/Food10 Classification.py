
# Import necessary packages.
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
from PIL import Image
import sys
import warnings
import bisect
import math
from collections import Counter
from sklearn import metrics

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    transforms.RandomChoice(
        [transforms.AutoAugment(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)]
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
    transforms.ToTensor(),
])


test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert("RGB")


def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(" ")
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


class ImageSet(Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.samples = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, gt = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, gt

    def __len__(self):
        return len(self.samples)


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes



batch_size = 32


data_root = 'data/Food10'

test_root = os.path.join(data_root, 'public')
test_label_file = os.path.join(test_root, 'public.txt')
test_set = ImageSet(test_root, loader=lambda x: Image.open(x), label=test_label_file, transform=test_tfm)

eval_root = os.path.join(data_root, 'private')
eval_label_file = os.path.join(eval_root, 'student.txt')
eval_set = ImageSet(eval_root, loader=lambda x: Image.open(x), label=eval_label_file, transform=test_tfm)

unlabeled_root = os.path.join(data_root, 'semi')
unlabeled_set = DatasetFolder(unlabeled_root, loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)

train_root = os.path.join(data_root, 'train')
train_label_file = os.path.join(train_root, 'train.txt')
train_set = ImageSet(train_root, loader=lambda x: Image.open(x), label=train_label_file, transform=train_tfm)

# You could utilize validation set for training and public set for validation,
# but the selected model might not work well as expected in private set.
CROSS_VALID = True
valid_root = os.path.join(data_root, 'valid')
valid_label_file = os.path.join(valid_root, 'valid.txt')
if CROSS_VALID:
    valid_set = ImageSet(valid_root, loader=lambda x: Image.open(x), label=valid_label_file, transform=test_tfm)
else:
    train_valid_set = ImageSet(valid_root, loader=lambda x: Image.open(x), label=valid_label_file, transform=train_tfm)
    train_set = ConcatDataset([train_set, train_valid_set])
    valid_set = ImageSet(test_root, loader=lambda x: Image.open(x), label=test_label_file, transform=test_tfm)

# Construct data loaders. Do not change the shuffle option.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, drop_last=False)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):

        x = self.cnn_layers(x)

        x = x.flatten(1)

        x = self.fc_layers(x)
        return x


def get_pseudo_labels(dataset, model, threshold=0.8):

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(axis=-1)
    samples = []
    temp = None
    # Iterate over the dataset by batches.
    for batch_id, data in enumerate(data_loader):
        img, _ = data
        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with paddle.no_grad():
            logits = model(img)

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)


    # # Turn off the eval mode.
    model.train()
    return dataset




# The number of training epochs.
n_epochs = 80
learning_rate = 0.0003


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# the path where checkpoint saved
work_path = 'work/model'

# Initialize a model
model = Classifier().to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
# grad_norm = paddle.nn.ClipGradByGlobalNorm(clip_norm=10)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Whether to do semi-supervised learning.
do_semi = False
gate = 0.0
threshold = 0.0
best_acc = 0.0
val_acc = 0.0
loss_record = {'train': {'loss': [], 'iter': []}, 'val': {'loss': [], 'iter': []}}  # for recording loss
acc_record = {'train': {'acc': [], 'iter': []}, 'val': {'acc': [], 'iter': []}}  # for recording accuracy

loss_iter = 0
acc_iter = 0

for epoch in range(n_epochs):
    if do_semi:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels(unlabeled_set, model)
        print('epoch {}: Pseudo Set Size: {}'.format(epoch, len(pseudo_set)))
        # Construct a new dataset and a data loader for training.
        # This is used in semi-supervised learning only.
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        this_train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        this_train_loader = train_loader

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()
    train_num = 0.0
    train_loss = 0.0

    val_num = 0.0
    val_loss = 0.0

    train_preds = None
    train_labels = None
    val_preds = None
    val_labels = None

    # Iterate the training set by batches.
    for batch_id, data in enumerate(this_train_loader):
        # A batch consists of image data and corresponding labels.
        x_data, y_data = data

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(x_data.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, y_data.to(device))

        # Compute the accuracy for current batch.
        if train_preds is None:
            train_preds = torch.argmax(logits, dim=1)
            train_labels = y_data
        else:
            train_preds = torch.cat((train_preds, torch.argmax(logits, dim=1)), dim=0)
            train_labels = torch.cat((train_labels, y_data), dim=0)

        # Compute the gradients for parameters.
        loss.backward()

        # Update the parameters with computed gradients.
        optimizer.step()
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        train_loss += loss.item()
        train_num += len(y_data)

    # The average loss and accuracy of the training set is the average of the recorded values.
    total_train_loss = (train_loss / train_num) * batch_size
    loss_record['train']['loss'].append(total_train_loss)
    loss_record['train']['iter'].append(loss_iter)
    loss_iter += 1
    train_acc = metrics.accuracy_score(train_labels.to('cpu').numpy(), train_preds.to('cpu').numpy())
    acc_record['train']['acc'].append(train_acc)
    acc_record['train']['iter'].append(acc_iter)
    acc_iter += 1
    # Print the information.
    print(
        "#===epoch: {}, train loss is: {}, train acc is: {:2.2f}%===#".format(epoch, total_train_loss, train_acc * 100))

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # Iterate the validation set by batches.
    for batch_id, data in enumerate(valid_loader):

        # A batch consists of image data and corresponding labels.
        x_data, y_data = data

        # We don't need gradient in validation.
        # Using paddle.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(x_data.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, y_data.to(device))

        # Compute the accuracy for current batch.
        if val_preds is None:
            val_preds = torch.argmax(logits, dim=1)
            val_labels = y_data
        else:
            val_preds = torch.cat((val_preds, torch.argmax(logits, dim=1)), dim=0)
            val_labels = torch.cat((val_labels, y_data), dim=0)

        # Record the loss and accuracy.
        val_loss += loss.item()
        val_num += len(y_data)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    total_val_loss = (val_loss / val_num) * batch_size
    loss_record['val']['loss'].append(total_val_loss)
    loss_record['val']['iter'].append(loss_iter)
    val_acc = metrics.accuracy_score(val_labels.to('cpu').numpy(), val_preds.to('cpu').numpy())
    acc_record['val']['acc'].append(val_acc)
    acc_record['val']['iter'].append(acc_iter)

    print("#===epoch: {}, val loss is: {}, val acc is: {:2.2f}%===#".format(epoch, total_val_loss, val_acc * 100))

    # ===================save====================
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(work_path, 'best_model.pth'))

print('best accuracy on validation set: ', best_acc)
torch.save(model.state_dict(), os.path.join(work_path, 'final_model.pth'))



def plot_learning_curve(record, title='loss', ylabel='CE Loss'):
    ''' Plot learning curve of your CNN '''
    maxtrain = max(map(float, record['train'][title]))
    maxval = max(map(float, record['val'][title]))
    ymax = max(maxtrain, maxval) * 1.1
    mintrain = min(map(float, record['train'][title]))
    minval = min(map(float, record['val'][title]))
    ymin = min(mintrain, minval) * 0.9

    total_steps = len(record['train'][title])
    x_1 = list(map(int, record['train']['iter']))
    x_2 = list(map(int, record['val']['iter']))
    figure(figsize=(10, 6))
    plt.plot(x_1, record['train'][title], c='tab:red', label='train')
    plt.plot(x_2, record['val'][title], c='tab:cyan', label='val')
    plt.ylim(ymin, ymax)
    plt.xlabel('Training steps')
    plt.ylabel(ylabel)
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


plot_learning_curve(loss_record, title='loss', ylabel='CE Loss')
plot_learning_curve(acc_record, title='acc', ylabel='Accuracy')

# ## Testing

# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
# Initialize a list to store the predictions.
predictions = []
model_state_dict = torch.load(os.path.join(work_path, 'best_model.pth'))
model.load_state_dict(model_state_dict)

test_num = 0.0
test_loss = 0.0
test_preds = None
test_labels = None

model.eval()

# test on testing set
for batch_id, data in enumerate(test_loader):
    x_data, y_data = data

    # ===================forward=====================
    # We don't need gradient in testing.
    # Using paddle.no_grad() accelerates the forward process.
    with torch.no_grad():
        predicts = model(x_data.to(device))
        loss = criterion(predicts, y_data.to(device))

    # ==================calculate acc================
    if test_preds is None:
        test_preds = torch.argmax(predicts, dim=1)
        test_labels = y_data
    else:
        test_preds = torch.cat((test_preds, torch.argmax(predicts, dim=1)), dim=0)
        test_labels = torch.cat((test_labels, y_data), dim=0)

    test_loss += loss.item()
    test_num += len(y_data)

test_acc = metrics.accuracy_score(test_labels.to('cpu').numpy(), test_preds.to('cpu').numpy())
total_test_loss = (test_loss / test_num) * batch_size
print("test loss is: {}, test acc is: {}".format(total_test_loss, test_acc))




