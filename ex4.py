import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



def one_hot(labels, n_class = 4):
    expansion = np.eye(n_class)
    y = expansion[labels[0],:]
    for i in range(1,labels.shape[0]):
        y = np.r_[y,expansion[labels[i],:]]
    assert y.shape[1] == n_class, "Wrong number of labels!"
    return y


def loadData(filePath):
    count = 0  #总样本数目
    for root, dirs, files in os.walk(filePath):  # 遍历统计
        for each in files:
            count += 1
    data_x = np.arange(count*15000*53*3,dtype=float).reshape(count,3,15000,53)
    data_xP = np.arange(count * 15000 * 53 * 3, dtype=float).reshape(count, 3, 15000, 53)
    data_y = np.arange(count,dtype=int).reshape(count,1)
    count = 0
    for eachInfo in os.listdir(filePath):
        file = open(filePath + '\\' + eachInfo, 'r')
        data = pd.read_csv(file,header=None,error_bad_lines=False)
        temp_y = data.iloc[39,2]
        file = open(filePath+'\\' + eachInfo, 'r')
        data = pd.read_csv(file, header= 40,skip_blank_lines=False)
        temp_x = np.arange(15000 * 53 * 3, dtype=float).reshape(3, 15000, 53)
        for i in range(1, 54):
            temp_x[0, :, i - 1] = data['CH' + str(i) + '(690)'].tolist()
            if (i==1)|(i==4)|(i==10):
                temp_x[1, :, i - 1] = data['CH' + str(i) + '(832)'].tolist()
            else:
                temp_x[1, :, i - 1] = data['CH' + str(i) + '(830)'].tolist()
            temp_x[2,:,i-1] = temp_x[0,:,i-1]+temp_x[0,:,i-1]
        data_x[count,:,:,:] = temp_x
        data_y[count,0] = int(temp_y)
        count = count + 1
        print(eachInfo)
    for i in range(count):
        for j in range(0,3):
            for k in range(0,53):
                data_xP[i,j,:,k] = rhythmExtraction(data_x[i,j,:,k],0,0.6,200,len(data_x[i,j,:,k]))
    return data_xP,data_y
    # train_x,test_x,train_y,test_y = train_test_split(data_xP,data_y,test_size=0.3)
    # train_y1 = one_hot(train_y)
    # test_y1 = one_hot(test_y)
    # #plotTend(data_x)
    # return train_x,test_x,train_y1,test_y1,train_y,test_y

#计算每个通道的平均值
def averageFeature(data):
    count = len(data)
    averageData = np.arange(count * 53 * 3,dtype=float).reshape(count,159)
    for i in range(count):
        for j in range(3):
            for k in range(53):
                averageData[i,j*53+k] = np.average(data[i,j,:,k])
    return averageData

def plotTend(trainData):
    plt.figure(figsize=(15,8))
    x = np.linspace(0, 150, 15000)
    chanelNum1 = 0
    chanelNum2 = 53
    plt.subplot(4, 1, 1)   #处理前时域空间图
    plt.xlabel("Time")
    plt.ylabel("Value")
    for i in range(chanelNum1,chanelNum2):
        plt.plot(x, trainData[0, 0, :,i],label=str(i),linewidth=0.5)
    plt.subplot(4, 1, 2)   #处理前频域空间图
    sampling_rate = 200  # 采样频率为200Hz
    for i in range(chanelNum1,chanelNum2):
        fft_size = len(trainData[0, 0, :, i])
        xf = np.fft.fft(trainData[0, 0, :, i])
        freqs = np.linspace(0, 1.0 * sampling_rate / 2, fft_size)
        plt.plot(freqs,xf)
    plt.subplot(4, 1, 3)  #处理后频域空间图
    y_time = np.zeros((2,15000,53),dtype = float)
    for i in range(chanelNum1,chanelNum2):
        fft_size = len(trainData[0, 0, :, i])
        xf = np.fft.fft(trainData[0, 0, :, i])
        y_ = rhythmExtraction1(xf,0,0.3,200,fft_size)
        freqs = np.linspace(0, 1.0 * sampling_rate / 2, fft_size)
        plt.plot(freqs,y_)
        y_time[1,:,i] = np.fft.ifft(y_)
    plt.subplot(4, 1, 4)  #处理后时域空间图
    for i in range(chanelNum1,chanelNum2):
        plt.plot(x, y_time[1,:,i])
    plt.legend()
    plt.show()



def rhythmExtraction1(oneFrame, f_low, f_high, fs, frameLength):
    # data_fft = np.fft.fft(oneFrame)
    f1 = round(frameLength / fs * f_low + 1)
    f2 = round(frameLength / fs * f_high + 1)
    f3 = round(frameLength / fs * (fs - f_high) + 1)
    f4 = round(frameLength / fs * (fs - f_low) + 1)

    oneFrame[1: f1] = 0
    oneFrame[f2: f3] = 0
    oneFrame[f4: frameLength] = 0
    #y_time = np.fft.ifft(oneFrame)
    return oneFrame

def rhythmExtraction(oneFrame, f_low, f_high, fs, frameLength):
    data_fft = np.fft.fft(oneFrame)
    f1 = round(frameLength / fs * f_low + 1)
    f2 = round(frameLength / fs * f_high + 1)
    f3 = round(frameLength / fs * (fs - f_high) + 1)
    f4 = round(frameLength / fs * (fs - f_low) + 1)

    data_fft[1: f1] = 0
    data_fft[f2: f3] = 0
    data_fft[f4: frameLength] = 0
    y_time = np.fft.ifft(data_fft)
    return y_time


def mulPca(trainData,testData):
    # 将特征值进行标准化处理
    std = StandardScaler()
    train_x = std.fit_transform(trainData)
    test_x = std.transform(testData)
    # 进行PCA降维
    pca = PCA(n_components=0.8)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print("PCA+++++++++++++")
    print(train_x.shape)
    # 将降维后的数据进行返回
    return train_x, test_x

def lda(trainData,testData,train_y):
    # 将特征值进行标准化处理
    std = StandardScaler()
    train_x = std.fit_transform(trainData)
    test_x = std.transform(testData)
    # 进行LDA降维
    lda = LinearDiscriminantAnalysis(n_components=0.8)
    train_x = lda.fit_transform(train_x.astype(int),train_y)
    test_x = lda.transform(test_x)
    # 将降维后的数据进行返回
    return train_x, test_x

def plotAcc(kernel,classAcc):
    plt.figure(figsize=(8,5))
    x = np.linspace(0, 53, 53)
    plt.subplot(3, 1, 1)
    plt.ylim(0,1)
    plt.title(kernel)
    plt.xlabel("Channel")
    plt.ylabel("Oxy-Accuracy")
    plt.bar(x,classAcc[0,:])
    plt.subplot(3, 1, 2)
    plt.ylim(0, 1)
    plt.xlabel("Channel")
    plt.ylabel("Deoxy-Accuracy")
    plt.bar(x, classAcc[1,:])
    plt.subplot(3, 1, 3)
    plt.ylim(0, 1)
    plt.xlabel("Channel")
    plt.ylabel("Total-Accuracy")
    plt.bar(x, classAcc[2,:])
    plt.legend()
    plt.show()
#对单个通道使用SVM分别进行分类，观察每个通道的有效性
def singleSVM(train_data,train_y,test_data,test_y):
    classAcc = np.arange(3*3 * 53, dtype=float).reshape(3, 3, 53)
    for type in range(3):
        for channel in range(53):
            random_state = np.random.RandomState(0)
            svmModel1 = OneVsRestClassifier(svm.SVC(kernel='linear', C = 2, probability=True, random_state=random_state))
            svmModel2 = OneVsRestClassifier(svm.SVC(kernel='rbf', C=2, gamma = 1e-3,probability=True, random_state=random_state))
            svmModel3 = OneVsRestClassifier(svm.SVC(kernel='poly', C=2, gamma = 1e-3,probability=True, random_state=random_state))
            clt1 = svmModel1.fit(train_data[:, type * 53 + channel].reshape(-1, 1), train_y)
            score1 = clt1.score(test_data[:, type * 53 + channel].reshape(-1, 1), test_y)
            clt2 = svmModel2.fit(train_data[:, type * 53 + channel].reshape(-1, 1), train_y)
            score2 = clt2.score(test_data[:, type * 53 + channel].reshape(-1, 1), test_y)
            clt3 = svmModel3.fit(train_data[:, type * 53 + channel].reshape(-1, 1), train_y)
            score3 = clt3.score(test_data[:, type * 53 + channel].reshape(-1, 1), test_y)
            classAcc[0][type][channel] = score1
            classAcc[1][type][channel] = score2
            classAcc[2][type][channel] = score3
    f = open('SVM各个通道分析', 'ab')
    np.savetxt(f, classAcc[0], delimiter=',')
    np.savetxt(f, classAcc[1], delimiter=',')
    np.savetxt(f, classAcc[2], delimiter=',')
    plotAcc("Linear",classAcc[0])
    plotAcc("Rbf",classAcc[1])
    plotAcc("Poly",classAcc[2])

#对单个通道使用KNN分别进行分类，观察每个通道的有效性
def singleKNN(train_data,train_y,test_data,test_y):
    classAcc = np.arange(3 * 3 * 53, dtype=float).reshape(3, 3, 53)
    for type in range(3):
        for channel in range(53):
            knnModel1 = KNeighborsClassifier(n_neighbors=3)
            knnModel2 = KNeighborsClassifier(n_neighbors=5)
            knnModel3 = KNeighborsClassifier(n_neighbors=6)
            clt1 = knnModel1.fit(train_data[:, type * 53 + channel].reshape(-1, 1), train_y)
            score1 = clt1.score(test_data[:, type * 53 + channel].reshape(-1, 1), test_y)
            clt2 = knnModel2.fit(train_data[:, type * 53 + channel].reshape(-1, 1), train_y)
            score2 = clt2.score(test_data[:, type * 53 + channel].reshape(-1, 1), test_y)
            clt3 = knnModel3.fit(train_data[:, type * 53 + channel].reshape(-1, 1), train_y)
            score3 = clt3.score(test_data[:, type * 53 + channel].reshape(-1, 1), test_y)
            classAcc[0][type][channel] = score1
            classAcc[1][type][channel] = score2
            classAcc[2][type][channel] = score3
    f = open('KNN各个通道分析.txt', 'ab')
    np.savetxt(f, classAcc[0], delimiter=',')
    np.savetxt(f, classAcc[1], delimiter=',')
    np.savetxt(f, classAcc[2], delimiter=',')
    plotAcc("K = 3", classAcc[0])
    plotAcc("K = 5", classAcc[1])
    plotAcc("K = 6", classAcc[2])

def plotSingle(dataX):
    count = len(dataX)
    for i in range(53):
        channel = "Channel " + str(i)
        plt.figure(figsize=(15,8))
        x = np.linspace(0, 150, 15000)
        plt.subplot(3, 1, 1)
        plt.title(channel)
        plt.ylabel("Oxy-Hb/mMmm")
        for j in range(count):
            plt.plot(x,dataX[j,0,:,i])
        plt.subplot(3, 1, 2)
        plt.ylabel("Deoxy-Hb/mMmm")
        for j in range(count):
            plt.plot(x,dataX[j,1,:,i])
        plt.subplot(3, 1, 3)
        plt.xlabel("Time/s")
        plt.ylabel("Total-Hb/mMmmy")
        for j in range(count):
            plt.plot(x,dataX[j,2,:,i])
        #plt.legend()
        plt.savefig("D:\各通道原始数据图\\"+channel+".png")

#TempDataset
#DepressionDataset - 2

if __name__=="__main__":
    #获取数据
    dataX,dataY = loadData('T:\BaiduNetdiskDownload\DepressionDataset\DepressionDataset - 2')
    train_x, test_x, train_y, test_y = train_test_split(dataX, dataY, test_size=0.3)
    averageTrainX = averageFeature(train_x)
    averateTestX = averageFeature(test_x)
    #对单个通道使用SVM分别进行分类，观察每个通道的有效性
    #singleSVM(averageTrainX,train_y,averateTestX,test_y)
    # 对单个通道使用KNN分别进行分类，观察每个通道的有效性
    #singleKNN(averageTrainX, train_y, averateTestX, test_y)
    # plotSingle(dataX)










    # train_y = list(map(int ,train_y))
    # print(train_y)
    # trainPca, testPca = mulPca(averageFeature(train_x),averageFeature(test_x))
    # #trainLda, testLda = lda(averageFeature(train_x), averageFeature(test_x),train_y)
    # # Learn to predict each class against the other
    # # 训练模型并预测
    # random_state = np.random.RandomState(0)
    # svmModel = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    # clt = svmModel.fit(trainPca, train_y1)
    # print("************************SVMScore:******************************")
    # print(clt.score(testPca, test_y1))

    # knnModel = KNeighborsClassifier()
    # # 设置k的范围
    # k_range = list(range(1, 6))
    # leaf_range = list(range(1, 2))
    # weight_options = ['uniform', 'distance']
    # algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
    # param_gridknn = dict(n_neighbors=k_range, weights=weight_options, algorithm=algorithm_options, leaf_size=leaf_range)
    # gridKNN = GridSearchCV(knnModel, param_gridknn, cv=6, scoring='accuracy', verbose=1)
    # clt = gridKNN.fit(trainPca, train_y1)
    # print('best score is:', str(gridKNN.best_score_))
    # print('best params are:', str(gridKNN.best_params_))
    # clt = clt.best_estimator_
    # print("************************KNNScore:******************************")
    # print(clt.score(testPca, test_y1))

    #differentClassify(trainPca,train_y1,testPca,test_y1)

    # print("++++++++++++++++++++++LDA++++++++++++++++++")
    # for clf_key in clfs.keys():
    #     print('the classifier is :', clf_key)
    #     clf = clfs[clf_key]
    #     try_different_method(clf,trainLda,train_y1,testLda,test_y1)
