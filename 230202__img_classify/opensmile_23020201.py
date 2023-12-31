# -*- coding: utf-8 -*-
"""OpenSmile-LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XjhMSH1I0DiaWgZGLbozHDNhrz12i0cW
"""

import librosa  # https://librosa.org/doc/main/feature.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import pandas as pd
import math

import opensmile

# 忽略librosa读取的warning
import warnings

warnings.filterwarnings('ignore')


# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=5):
        super(LSTM, self).__init__()

        self.RNN_input_size = input_size
        self.h_RNN_layers = h_RNN_layers
        self.h_RNN = h_RNN
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        self.x_tsne = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(self.x_tsne)

        return x


# 读取数据
data_dir = "./original"  # 设置自己路径
img_x, img_y = 256, 256  # 图片大小
vedio_path = []
labels = []

i = 0
for root, dirs, files in os.walk(data_dir):
    for file in files:
        path = os.path.join(root, file)
        label_pt = torch.tensor(i - 1)
        labels.append(label_pt)
        vedio_path.append(path)
    i += 1

#print(labels)
# 0: 9-10
# 1: 5-6
# 2: 1-2
# 3: 7-8
# 4: 3-4
#print(vedio_path)

# train, test split
train_list, test_list, train_label, test_label = train_test_split(vedio_path, labels, test_size=0.1, random_state=2)

y, sr = librosa.load(vedio_path[0])

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)


def get_opensmile_features(path):
    y, sr = librosa.load(path)
    features = smile.process_signal(y, sr)
    features = np.array(features)
    features = torch.from_numpy(features)
    return features


# 读取数据
class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data_path, labels):
        # max_frames 设置需要读取的帧数
        # 初始化
        self.data_path = data_path
        self.labels = labels

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data_path)

    def read_audio(self, path):
        X = get_opensmile_features(path)

        return X

    def __getitem__(self, index):
        """Generates one sample of data"""
        # 选择视频
        path = self.data_path[index]

        # 加载视频
        X = self.read_audio(path)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


train_set = Dataset(train_list, train_label)
valid_set = Dataset(test_list, test_label)

save_model_path = "./openmodel"  # 设置保存模型路径

# RNN模型参数
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# 训练参数
k = 5  # 类别数
epochs = 50  # 训练迭代
batch_size = 30
learning_rate = 1e-4  # 学习率
log_interval = 1  # 每10次迭代显示一次信息
input_size = 25  # opensmile输出feauture维度为25
# 选用GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # 训练模型函数
    rnn_decoder = model
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0  # 已经训练过的视频
    for batch_idx, (X, y) in enumerate(train_loader):

        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(X)  # 输出

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        accu_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        balanced_score = balanced_accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        f1 = f1_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy(), average="macro")
        weighted_f1 = f1_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy(), average="weighted")
        step_score = [accu_score, balanced_score, f1, weighted_f1]
        scores.append(step_score)  # computed on CPU

        loss.backward()
        optimizer.step()

        # log信息
        if (batch_idx + 1) % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%, Weighted Accu: {:.2f}%, F1: {:.2f}%, Weighted F1: {:.2f}%'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    loss.item(), 100 * step_score[0], 100 * step_score[1], 100 * step_score[2], 100 * step_score[3]))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # 测试模型函数
    rnn_decoder = model
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    all_tsne = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(X)
            output_tsne = rnn_decoder.x_tsne
            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()  # 计算loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # 收集全部预测类别
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_tsne.extend(output_tsne)

    test_loss /= len(test_loader.dataset)

    # 计算准确度
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    all_tsne = torch.stack(all_tsne, dim=0)

    accu_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    balanced_score = balanced_accuracy_score(all_y.cpu().data.squeeze().numpy(),
                                             all_y_pred.cpu().data.squeeze().numpy())
    f1 = f1_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy(), average="macro")
    weighted_f1 = f1_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy(),
                           average="weighted")
    test_score = [accu_score, balanced_score, f1, weighted_f1]

    # log信息
    print(
        '\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Weighted Accuracy: {:.2f}%, F1: {:.2f}%, Weighted F1: {:.2f}%\n'.format(
            len(all_y), test_loss, 100 * test_score[0], 100 * test_score[1], 100 * test_score[2], 100 * test_score[3]))
    datas = 'Average loss: ' + str(test_loss) + ' Accuracy: ' + str(100 * test_score[0]) + ' Weighted Accuracy: ' + str(
        100 * test_score[1]) + ' F1: ' + str(100 * test_score[2]) + ' Weighted F1: ' + str(100 * test_score[3])

    with open('opensmiledatas.txt', 'a', encoding='utf-8') as f:
        f.write(datas)
        f.write('\n')
        f.close()
    # save Pytorch models of best record
    if (epoch + 1) % 5 == 0:
        torch.save(rnn_decoder.state_dict(),
                   os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(),
                   os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))  # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score, all_y, all_y_pred, all_tsne


rnn_decoder = LSTM(input_size=input_size, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                   h_FC_dim=RNN_FC_dim, drop_p=0, num_classes=k).to(device)

# 在多个GPU上平行运行
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    rnn_decoder = nn.DataParallel(rnn_decoder)

optimizer = torch.optim.Adam(rnn_decoder.parameters(), lr=learning_rate)

# 记录训练过程
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
def plt_tsne(gt_labels, pred_labels):
    tsne = TSNE(n_components=2, perplexity=5)
    X_tsne = tsne.fit_transform(pred_labels)
    X_tsne_data = np.vstack((X_tsne.T, gt_labels)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
    plt.savefig('opensmile_tsne.jpg')

def plot_confusion_matrix(gt_labels, pred_labels, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    cm = confusion_matrix(gt_labels, pred_labels)
    print("confusion_mat.shape : {}".format(cm.shape))
    print("confusion_mat : {}".format(cm))

    if normalize:
        matrix = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure()
    # 设置输出的图片大小
    figsize = 8, 8
    figure, ax = plt.subplots(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # 设置title的大小以及title的字体
    font_title = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15,
                  }
    plt.title(title, fontdict=font_title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45, )
    # plt.yticks(tick_marks, classes)
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    print(labels)
    [label.set_fontname('Times New Roman') for label in labels]
    if normalize:
        fm_int = 'd'
        fm_float = '.2%'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_float),
                     horizontalalignment="center", verticalalignment='bottom', family="Times New Roman",
                     weight="normal", size=15,
                     color="white" if cm[i, j] > thresh else "black")
            plt.text(j, i, format(matrix[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='top', family="Times New Roman", weight="normal",
                     size=15,
                     color="white" if cm[i, j] > thresh else "black")
    else:
        fm_int = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='bottom',
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.savefig('opensmile_cm.jpg')

# 开始训练
if __name__ == '__main__':
    for epoch in range(epochs):
        # 训练，测试
        train_losses, train_scores = train(log_interval, rnn_decoder, device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score,all_y, all_y_pred, all_tsne = validation(rnn_decoder, device, optimizer, valid_loader)

        # 保存结果
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # 保存全部结果
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
        np.save('./Opensmile_LSTM_training_losses.npy', A)
        np.save('./Opensmile_LSTM_training_scores.npy', B)
        np.save('./Opensmile_LSTM_test_loss.npy', C)
        np.save('./Opensmile_LSTM_test_score.npy', D)
    plot_confusion_matrix(all_y.cpu(), all_y_pred.cpu()[:, 0])
    plt_tsne(all_y.cpu(),all_tsne.cpu())