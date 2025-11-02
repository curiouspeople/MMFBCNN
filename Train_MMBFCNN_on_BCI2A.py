# coding=utf-8
# -*-coding:gb2312-*-
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, precision_score, recall_score, accuracy_score

from utils.plot2 import plot_confusion_matrix
from utils.Index_calculation import testclass
import warnings

from MMBFCNN.mbff import MMBFCNN_bci

from torch.utils import data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.load_bci_2a_dataset import load_train_test_datset_unenhance


print("库导入成功")
warnings.filterwarnings("ignore")

learning_rate = 0.0001
epochs = 400
batch_size = 72

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

max_acc_list = []
max_kappa_list = []
max_f1Score_list = []
max_precisionScore_list = []
max_recallScore_list = []

for subject_i in range(1, 10):
    myModel = MMBFCNN_bci(epochs=3).to(device)
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(myModel.parameters(), lr=learning_rate, weight_decay=0.01)
    print('-' * 20 + "subject_i : {} ".format(subject_i) + '-' * 20)

    train_set, test_set = load_train_test_datset_unenhance(subject_i)
    print(len(train_set[0]))
    print(len(test_set[0]))
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

    max_acc = 0.
    max_kappa = 0.
    max_f1Score = 0.
    max_precisionScore = 0.
    max_recallScore = 0.
    max_sne_output = []
    max_sne_y = []
    for i in range(epochs):
        # 记录训练和测试次数
        total_train_step = 0
        total_test_step = 0

        # 训练和测试总loss
        total_train_loss = 0
        total_train_acc = 0
        X_TSNE = []
        Y_TSNE = []
        for x, y in train_dataloader:
            x = x.reshape(batch_size, 22, 1125, 1)
            x = x.to(torch.float32).to(device)
            y = y.to(torch.long).to(device)
            outputs = myModel(x)

            # tsne图
            X_TSNE.append(outputs.cpu().detach().numpy())
            Y_TSNE.append(y.cpu().detach().numpy())

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

        predicted_label = []
        true_label = []

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

                predicted_tmp = outputs.argmax(1)
                predicted_tmp = predicted_tmp.cpu().numpy()
                testy_tmp = testy.cpu().numpy()

                predicted_label.append(predicted_tmp)
                true_label.append(testy_tmp)

                test_acc = (outputs.argmax(1) == testy).sum().item()

                total_test_loss = total_test_loss + test_loss.item()
                total_test_step = total_test_step + 1
                total_test_acc += test_acc

        Test_Loss_list.append(total_test_loss / (len(test_dataloader)))
        Test_Accuracy_list.append(total_test_acc / test_len)

        predicted_label = np.array(predicted_label).reshape(-1)
        true_label = np.array(true_label).reshape(-1)
        cm = confusion_matrix(true_label, predicted_label)
        kappa = cohen_kappa_score(true_label, predicted_label)
        f1Score = f1_score(true_label, predicted_label, average='macro')
        precisionScore = precision_score(true_label, predicted_label, average='macro')
        recallScore = recall_score(true_label, predicted_label, average='macro')
        acc_sklearn = accuracy_score(true_label, predicted_label)

        # plot_confusion_matrix(cm, subject_i)

        # 保存准确率最高的模型参数
        if (total_test_acc / test_len) > max_acc:
            max_acc = total_test_acc / test_len
            max_kappa = kappa
            max_f1Score = f1Score
            max_precisionScore = precisionScore
            max_recallScore = recallScore
            early_stop = 0
            Max_CM = cm

            max_sne_output = X_TSNE
            max_sne_y = Y_TSNE

        early_stop += 1
        # if early_stop >= 300:
        #     break
        if i % 20 == 0:
            # if True:
            print("subject_i: {} ".format(subject_i),
                  "Epoch: {}/{} ".format(i + 1, epochs),
                  "Train Loss: {:.8f} ".format(total_train_loss / train_len),
                  "Train Acc: {:.4f} ".format(total_train_acc / train_len),
                  "Test Loss: {:.8f} ".format(total_test_loss / test_len),
                  "Test Acc: {:.4f}".format(total_test_acc / test_len),
                  "Sklearn ACC: {:.4f} ".format(acc_sklearn),
                  "Max ACC: {:.4f}".format(max_acc))
            with open('result/result_BCI2A_acc.txt', 'a') as f:
                f.write(
                    "subject_i: {} Epoch: {}/{} Train Loss: {:.8f} Train Acc: {:.4f}  Test Loss: {:.8f} Test Acc: {:.4f} Sklearn ACC: {:.4f} Max Acc: {:.4f}\n"
                        .format(subject_i, i + 1, epochs, total_train_loss / train_len, total_train_acc / train_len,
                                total_test_loss / test_len, total_test_acc / test_len, acc_sklearn, max_acc))

    # 保存sne图要画的特征
    feats = np.array(max_sne_output)
    feats_y = np.array(max_sne_y)
    feats = feats.reshape(-1, 4)
    feats_y = feats_y.reshape(-1)

    print("here")
    print(feats.shape)
    print(feats_y.shape)
    joblib.dump(feats, 'result/result_tsne_data/bci_2a/feat.pkl')
    joblib.dump(feats_y, 'result/result_tsne_data/bci_2a/feat_y.pkl')

    # # 这里可以画一下图 曲线图和混淆矩阵
    # name = str("sbj_{}".format(subject_i))
    # save_dir = "result/not_cross/result_ts3_PSD/"
    # plot.PLOT4(Train_Accuracy_list, Test_Accuracy_list, Train_Loss_list, Test_Loss_list,
    #            i + 1, name, save_dir)  # i+1是因为epoch从0开始
    #
    # save_dir = "result/not_cross/confusion_matrix/"
    # plot.plot_confusion_matrix(Max_CM, subject_i, save_dir, "")

    max_acc_list.append(max_acc)
    max_kappa_list.append(max_kappa)
    max_f1Score_list.append(max_f1Score)
    max_precisionScore_list.append(max_precisionScore)
    max_recallScore_list.append(max_recallScore)

    print("Max Acc: {}".format(max_acc))
    with open('result/result_BCI2A_acc.txt', 'a') as f:
        f.write("Sbj: {} Max Acc: {}\n\n".format(subject_i, max_acc))

print("End" + "-" * 40)
with open('result/result_BCI2A_acc.txt', 'a') as f:
    k_ = 1
    avg = 0
    f.write("End" + "-" * 40)
    f.write("\n")
    for i in range(9):
        print("In Subject: {} Max Acc: {:.4f} Kappa: {:.4f} F1_Score: {:.4f} Precision: {:.4f} Recall: {:.4f}".
              format(i + 1, max_acc_list[i], max_kappa_list[i], max_f1Score_list[i], max_precisionScore_list[i],
                     max_recallScore_list[i]))
        f.write(
            "In Subject: {} Max Acc: {:.4f} Kappa: {:.4f} F1_Score: {:.4f} Precision: {:.4f} Recall: {:.4f}\n".
                format(i + 1, max_acc_list[i], max_kappa_list[i], max_f1Score_list[i],
                       max_precisionScore_list[i],
                       max_recallScore_list[i]))
    print(
        "end -- all average Acc: {:.4f} Kappa: {:.4f} F1_Score: {:.4f} Precision: {:.4f} Recall: {:.4f}".format(
            sum(max_acc_list) / len(max_acc_list), sum(max_kappa_list) / len(max_kappa_list),
            sum(max_f1Score_list) / len(max_f1Score_list),
            sum(max_precisionScore_list) / len(max_precisionScore_list),
            sum(max_recallScore_list) / len(max_recallScore_list)
        ))
    f.write(
        "end -- all average Acc: {:.4f} Kappa: {:.4f} F1_Score: {:.4f} Precision: {:.4f} Recall: {:.4f}\n".format(
            sum(max_acc_list) / len(max_acc_list), sum(max_kappa_list) / len(max_kappa_list),
            sum(max_f1Score_list) / len(max_f1Score_list),
            sum(max_precisionScore_list) / len(max_precisionScore_list),
            sum(max_recallScore_list) / len(max_recallScore_list)
        ))
