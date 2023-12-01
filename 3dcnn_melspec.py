from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
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

import librosa  # https://librosa.org/doc/main/feature.html
import torchaudio
from librosa.filters import mel

# 忽略librosa读取的warning
import warnings

warnings.filterwarnings('ignore')


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_class=5):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.num_class = num_class

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, num_class)

    def forward(self, classifier_x):
        x = self.fc1(classifier_x)
        x = self.fc2(x)

        return x


# 3D CNN
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # 3DCNN输出结果维度
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=5,
                 load_pretrain=False):
        super(CNN3D, self).__init__()
        self.load_pretrain = load_pretrain
        # 设置视频维度
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # 全连接层
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        if self.load_pretrain:
            self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # 输出层

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.load_pretrain:
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            return x
        else:
            return x


# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=5,
                 load_pretrain=False):
        super(LSTM, self).__init__()

        self.load_pretrain = load_pretrain
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
        if self.load_pretrain:
            self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        if self.load_pretrain:
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc2(x)

            return x
        else:
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

# print(labels)
# 0: 9-10
# 1: 5-6
# 2: 1-2
# 3: 7-8
# 4: 3-4
# print(vedio_path)

# train, test split
train_list, test_list, train_label, test_label = train_test_split(vedio_path, labels, test_size=0.1, random_state=2)

from librosa.filters import mel


# 读取视频，返回音频Mel-Spec
def get_mel_spec(path):
    y, sr = librosa.load(path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return mel_spec


# 读取视频并抽取帧数
def get_center_image(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)

    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


# 加载视频
def load_video(path, max_frames=0, use_transform=None, skip=20):
    # 跳帧读取，连续帧数效果一般
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    try:
        while True:
            if i % skip == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = get_center_image(frame)
                frame = Image.fromarray(frame).convert('L')

                if use_transform is not None:
                    frame = use_transform(frame)

                frames.append(frame.squeeze_(0))

                if len(frames) == max_frames:
                    break
                i += 1
            else:
                i += 1
                continue

    finally:
        cap.release()
    return frames


class Dataset_Vedio_MelSpec(data.Dataset):
    def __init__(self, data_path, labels, max_frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.transform = transform
        self.frames = max_frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data_path)

    def read_images(self, path, use_transform):
        X = load_video(path, self.frames, use_transform)
        X = torch.stack(X, dim=0)

        return X

    def read_audio(self, path):
        X = get_mel_spec(path)
        X = torch.from_numpy(X)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # 选择视频
        path = self.data_path[index]

        # 加载视频
        X_vedio = self.read_images(path, self.transform).unsqueeze_(0)  # (input) spatial images
        X_mel_spec = self.read_audio(path)
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X_vedio, X_mel_spec, y


transform = transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])

max_frames = 30

train_set = Dataset_Vedio_MelSpec(train_list, train_label, max_frames, transform)
valid_set = Dataset_Vedio_MelSpec(test_list, test_label, max_frames, transform)

save_model_path = './combinemodel'

# 3D CNN 参数
fc_hidden1, fc_hidden2 = 256, 256
dropout = 0.0  # dropout probability

# RNN模型参数
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256
input_size = 475  # Mel-Spec的维度为（128，475）

# classifier参数
input_dim = 512

# 训练参数
k = 5  # 类别数
epochs = 100
batch_size = 30
learning_rate = 1e-4
log_interval = 1
img_x, img_y = 256, 256

# 选用GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# 创建模型
cnn3d = CNN3D(t_dim=max_frames, img_x=img_x, img_y=img_y,
              drop_p=dropout, fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, num_classes=k, load_pretrain=False).to(
    device)

rnn_decoder = LSTM(input_size=input_size, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                   h_FC_dim=RNN_FC_dim, drop_p=0, num_classes=k, load_pretrain=False).to(device)

classifer = Classifier(input_dim, k).to(device)


def train(log_interval, model, device, train_loader, optimizer, epoch, use_pretrained=False):
    # 训练模型函数
    cnn_model, rnn_decoder, classifier = model
    if not use_pretrained:
        cnn_model.train()
        rnn_decoder.train()

    classifier.train()

    losses = []
    scores = []
    N_count = 0  # 已经训练过的视频
    for batch_idx, (X_vedio, X_mel, y) in enumerate(train_loader):

        X_vedio, X_mel, y = X_vedio.to(device), X_mel.to(device), y.to(device).view(-1, )

        N_count += X_vedio.size(0)

        optimizer.zero_grad()

        cnn_out = cnn_model(X_vedio)  # 输出
        lstm_out = rnn_decoder(X_mel)  # 输出
        classifer_input = torch.cat([cnn_out, lstm_out], dim=1).to(device)

        output = classifier(classifer_input)

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


def validation(model, device, optimizer, test_loader, use_pretrained=False):
    # 测试模型函数

    cnn_model, rnn_decoder, classifier = model
    if not use_pretrained:
        rnn_decoder.eval()
        cnn_model.eval()

    classifer.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X_vedio, X_mel, y in test_loader:
            X_vedio, X_mel, y = X_vedio.to(device), X_mel.to(device), y.to(device).view(-1, )

            cnn_out = cnn_model(X_vedio)  # 输出
            lstm_out = rnn_decoder(X_mel)  # 输出
            classifer_input = torch.cat([cnn_out, lstm_out], dim=1).to(device)

            output = classifier(classifer_input)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()  # 计算loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # 收集全部预测类别
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # 计算准确度
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)

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

    with open('combine.txt', 'a', encoding='utf-8') as f:
        f.write(datas)
        f.write('\n')
        f.close()
    # save Pytorch models of best record
    if (epoch + 1) % 10 == 0:
        if not use_pretrained:
            torch.save(cnn_model.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(
                epoch + 1)))  # save motion_encoder
            torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(
                epoch + 1)))  # save motion_encoder
            torch.save(classifer.state_dict(),
                       os.path.join(save_model_path, 'classifier_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        else:
            torch.save(classifer.state_dict(), os.path.join(save_model_path,
                                                            'classifier_with_pretrained_epoch{}.pth'.format(
                                                                epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(),
                   os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))  # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


# 在多个GPU上平行运行
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    rnn_decoder = nn.DataParallel(rnn_decoder)

model_params = list(cnn3d.parameters()) + list(rnn_decoder.parameters()) + list(classifer.parameters())
optimizer = torch.optim.Adam(model_params, lr=learning_rate)

# 记录训练过程
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# 开始训练
if __name__ == '__main__':
    for epoch in range(epochs):
        # 训练，测试
        train_losses, train_scores = train(log_interval, [cnn3d, rnn_decoder, classifer], device, train_loader,
                                           optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation([cnn3d, rnn_decoder, classifer], device, optimizer, valid_loader)

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
        np.save('./3DCNN_Mel_epoch_training_losses.npy', A)
        np.save('./3DCNN_Mel_epoch_training_scores.npy', B)
        np.save('./3DCNN_Mel_epoch_test_loss.npy', C)
        np.save('./3DCNN_Mel_epoch_test_score.npy', D)

    """# 加载预训练模型"""


    # 加载预训练模型

    # 替代加载模型的分类层，使模型只输出最后一层特征
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x


    cnn3d_model_path = "./3dcnnmodel"

    # 加载3dcnn模型
    cnn3d = CNN3D(t_dim=max_frames, img_x=img_x, img_y=img_y,
                  drop_p=dropout, fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, num_classes=k, load_pretrain=True).to(
        device)

    cnn3d.load_state_dict(torch.load(os.path.join(cnn3d_model_path, '3dcnn_epoch41.pth')))

    cnn3d.fc3 = Identity()

    rnn_decoder_model_path = "./melmodel2"

    rnn_decoder = LSTM(input_size=input_size, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                       h_FC_dim=RNN_FC_dim, drop_p=0, num_classes=k, load_pretrain=True).to(device)

    rnn_decoder.load_state_dict(torch.load(os.path.join(rnn_decoder_model_path, 'rnn_decoder_epoch40.pth')))

    rnn_decoder.fc2 = Identity()

    classifer = Classifier(input_dim, k).to(device)

    optimizer = torch.optim.Adam(classifer.parameters(), lr=learning_rate)

    transform = transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5], std=[0.5])])

    max_frames = 30

    train_set = Dataset_Vedio_MelSpec(train_list, train_label, max_frames, transform)
    valid_set = Dataset_Vedio_MelSpec(test_list, test_label, max_frames, transform)

    # 选用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6, 'pin_memory': True} if use_cuda else {}

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # 记录训练过程
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # 开始训练

    for epoch in range(epochs):
        # 训练，测试
        train_losses, train_scores = train(log_interval, [cnn3d, rnn_decoder, classifer], device, train_loader,
                                           optimizer, epoch, use_pretrained=True)
        epoch_test_loss, epoch_test_score = validation([cnn3d, rnn_decoder, classifer], device, optimizer, valid_loader,
                                                       use_pretrained=True)

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
        np.save('./3DCNN_Mel_epoch_training_losses_with_pretained.npy', A)
        np.save('./3DCNN_Mel_epoch_training_scoreswith_pretained.npy', B)
        np.save('./3DCNN_Mel_epoch_test_loss_with_pretained.npy', C)
        np.save('./3DCNN_Mel_epoch_test_score_with_pretained.npy', D)

        # -*- coding: utf-8 -*-
        """3DCNN-MelSpec.ipynb

        Automatically generated by Colaboratory.

        Original file is located at
            https://colab.research.google.com/drive/18N7ro_DsfAtRzcKtoaO6W6fPQLKpSikh
        """
