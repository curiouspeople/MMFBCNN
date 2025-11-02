# coding=utf-8
# -*-coding:gb2312-*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.Index_calculation import testclass
import warnings

from MMBFCNN.mbff import MMBFCNN_bci

from torch.utils import data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.load_bci_2a_dataset import load_train_test_datset_unenhance

warnings.filterwarnings("ignore")

# learning_rate = 0.0001
learning_rate = 0.0001
epochs = 400
batch_size = 72

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_acc_list = []

for subject_i in range(3, 10):
    myModel = MMBFCNN_bci(epochs=3).to(device)
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(myModel.parameters(), lr=learning_rate, weight_decay=0.01)
    max_acc = 0.
    print('-' * 20 + "subject_i : {} ".format(subject_i) + '-' * 20)

    train_set, test_set = load_train_test_datset_unenhance(subject_i)

    # 将tenor数据包装好
    train_dataloader = DataLoader(dataset=TensorDataset(train_set[0], train_set[1]), batch_size=batch_size,
                                  shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=TensorDataset(test_set[0], test_set[1]), batch_size=batch_size, shuffle=True,
                                 drop_last=True)

    EST = testclass()

    train_len = EST.len(len(train_set[0]), batch_size)
    test_len = EST.len(len(test_set[0]), batch_size)
    print(train_len)
    print(test_len)

    Train_Loss_list = []
    Train_Accuracy_list = []
    Test_Loss_list = []
    Test_Accuracy_list = []

    for i in range(epochs):
        # 记录训练和测试次数
        total_train_step = 0
        total_test_step = 0

        # 训练和测试总loss
        total_train_loss = 0
        total_train_acc = 0

        for x, y in train_dataloader:
            x = x.reshape(batch_size, 22, 1125, 1)
            x = x.to(torch.float32).to(device)
            y = y.to(torch.long).to(device)
            outputs = myModel(x)

            train_loss = loss_func(outputs, y)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            train_acc = (outputs.argmax(1) == y).sum().item()

            total_train_loss = total_train_loss + train_loss.item()
            total_train_step = total_train_step + 1

            total_train_acc += train_acc

        Train_Loss_list.append(total_train_loss / (len(train_dataloader)))
        Train_Accuracy_list.append(total_train_acc / train_len)

        total_test_loss = 0
        total_test_acc = 0

        with torch.no_grad():
            pred_output_list = []
            for data in test_dataloader:
                testx, testy = data

                # 测试的batch_size 设置为36 刚好144/36=4
                testx = testx.reshape(batch_size, 22, 1125, 1)
                testx = testx.to(torch.float32).to(device)
                testy = testy.to(torch.long).to(device)

                outputs = myModel(testx)
                testy = testy.squeeze()
                test_loss = loss_func(outputs, testy.long())

                test_acc = (outputs.argmax(1) == testy).sum().item()

                total_test_loss = total_test_loss + test_loss.item()
                total_test_step = total_test_step + 1
                total_test_acc += test_acc

        Test_Loss_list.append(total_test_loss / (len(test_dataloader)))
        Test_Accuracy_list.append(total_test_acc / test_len)

        # 保存准确率最高的模型参数
        if (total_test_acc / test_len) > max_acc:
            max_acc = total_test_acc / test_len

        if i % 20 == 0:
            # if True:
            print("subject_i: {} ".format(subject_i),
                  "Epoch: {}/{} ".format(i + 1, epochs),
                  "Training Loss: {:.8f} ".format(total_train_loss / train_len),
                  "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
                  "Test Loss: {:.8f} ".format(total_test_loss / test_len),
                  "Test Accuracy: {:.4f}".format(total_test_acc / test_len))
            with open('./result_acc.txt', 'a') as f:
                f.write(
                    "subject_i: {} Epoch: {}/{} Training Loss: {:.8f} Training Accuracy: {:.4f} Test Loss: {:.8f} Test Accuracy: {:.4f} MAX_ACC:{:.4f}\n"
                    .format(subject_i, i + 1, epochs, total_train_loss / train_len, total_train_acc / train_len,
                            total_test_loss / test_len, total_test_acc / test_len, max_acc))
    max_acc_list.append(max_acc)
    print("Max Acc: {}".format(max_acc))
    with open('./result_acc.txt', 'a') as f:
        f.write("Sbj: {} Max Acc: {}\n\n".format(subject_i, max_acc))

print("End" + "-" * 40)
with open('./result_acc.txt', 'a') as f:
    sbj = 1
    avg = 0
    f.write("End" + "-" * 40)
    f.write("\n")
    for i in max_acc_list:
        print("max acc in sbj_no: {} Acc: {}".format(sbj, i))
        f.write("max acc in sbj_no: {} Acc: {}\n".format(sbj, i))
        sbj += 1
        avg += i
    print("average acc: {}".format(avg / len(max_acc_list)))
    f.write("End Avg Acc: {}\n".format(avg / len(max_acc_list)))
