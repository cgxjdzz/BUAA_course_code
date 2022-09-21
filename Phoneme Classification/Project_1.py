#!/usr/bin/env python
# coding: utf-8

# # **Project 1 - Phoneme Classification**


# ## Import Some Packages


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import gc
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# ## Preparing Data



print('Loading data ...')

data_root='work/data/'
train = np.load(data_root + 'train_x.npy')
train_label = np.load(data_root + 'train_y.npy')
test = np.load(data_root + 'test_x.npy')
test_label = np.load(data_root + 'test_y.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


# ## Create Dataset



class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.tensor(X,dtype=torch.float32)
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.tensor(y).long()
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx], -1

    def __len__(self):
        return len(self.data)


VAL_RATIO = 0.3

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))


# Create a data loader from the dataset.
BATCH_SIZE = 128

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


del train, train_label, train_x, train_y, val_x, val_y
gc.collect()


# ## Create Model
# Define model architecture, you are encouraged to change and experiment with the model architecture.



class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512,128)
        self.out = nn.Linear(128, 39) 

        self.act_fn = nn.Sigmoid()
        self.act_fn2 = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(429)
        self.drop = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.bn(x)
        

        
        x = self.layer1(x)
        x = self.act_fn(x)
        
        
        x = self.drop(x)
        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        
        return x


# ## Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training parameters
epochs = 40            # number of training epoch
     # learning rate
learning_rate = 0.001
# the path where checkpoint saved
work_path = 'work/model'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.0
loss_record = {'train': [], 'val': []}      # for recording training loss
        
for epoch in range(epochs):
    if (epoch >10):
      learning_rate=learning_rate/1.2
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    train_num = 0.0
    train_loss = 0.0
    train_hit = 0.0

    val_num = 0.0
    val_loss = 0.0
    
    train_preds = None
    train_labels = None
    val_preds = None
    val_labels = None
    
    for batch_id, data in enumerate(train_loader):
        x_data = data[0].to(device)
        y_data = data[1].to(device)

        # ===================forward=====================
        predicts = model(x_data)
        loss = criterion(predicts, y_data)
        
        # ==================calculate acc================
        if train_preds is None:
            train_preds = torch.argmax(predicts, dim=1)
            train_labels = y_data
        else:
            train_preds = torch.cat((train_preds, torch.argmax(predicts, dim=1)), dim=0)
            train_labels = torch.cat((train_labels, y_data), dim=0)
    
        # ===================backward====================
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_num += len(y_data)

    train_acc = metrics.accuracy_score(train_labels.to('cpu').numpy(), train_preds.to('cpu').numpy())
    total_train_loss = (train_loss / train_num) * BATCH_SIZE
    loss_record['train'].append(total_train_loss)
    print("epoch: {}, train loss is: {}, train acc is: {}".format(epoch, total_train_loss, train_acc))
    
    model.eval()
    for batch_id, data in enumerate(val_loader):
        x_data = data[0].to(device)
        y_data = data[1].to(device)
        
        # ===================forward=====================
        predicts = model(x_data)
        loss = criterion(predicts, y_data)

        # ==================calculate acc================
        if val_preds is None:
            val_preds = torch.argmax(predicts, dim=1)
            val_labels = y_data
        else:
            val_preds = torch.cat((val_preds, torch.argmax(predicts, dim=1)), dim=0)
            val_labels = torch.cat((val_labels, y_data), dim=0)

        val_loss += loss.item()
        val_num += len(y_data)

    val_acc = metrics.accuracy_score(val_labels.to('cpu').numpy(), val_preds.to('cpu').numpy())
    total_val_loss = (val_loss / val_num) * BATCH_SIZE
    loss_record['val'].append(total_val_loss)
    print("epoch: {}, val loss is: {}, val acc is: {}".format(epoch, total_val_loss, val_acc))
    # ===================save====================
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(work_path, 'best_model.pth'))

print('best accuracy on validation set: ', best_acc)
torch.save(model.state_dict(), os.path.join(work_path, 'best_model.pth'))


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Classifier().to(device)
# criterion = nn.CrossEntropyLoss() 
# work_path = 'work/model'

test_set = TIMITDataset(test, test_label)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

test_num = 0.0
test_loss = 0.0
test_preds = None
test_labels = None

model_state_dict = torch.load(os.path.join(work_path, 'best_model.pth'))
model.load_state_dict(model_state_dict)

model.eval()

for batch_id, data in enumerate(test_loader):
    x_data = data[0].to(device)
    y_data = data[1].to(device)
    
    # ===================forward=====================
    predicts = model(x_data)
    loss = criterion(predicts, y_data)

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
total_test_loss = (test_loss / test_num) * BATCH_SIZE
print("test loss is: {}, test acc is: {}".format(total_test_loss, test_acc))
# ## Plot loss curves

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & val loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[len(loss_record['train']) // len(loss_record['val'])-1::len(loss_record['train']) // len(loss_record['val'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['val'], c='tab:cyan', label='val')
    plt.ylim(0.0, 4.)
    plt.xlabel('Training steps')
    plt.ylabel('CE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

plot_learning_curve(loss_record, title='deep model')





