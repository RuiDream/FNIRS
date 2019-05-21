import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.decomposition import  PCA
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import tensorflow


def preFilter(filePath):
    file = open(filePath,'r')
    time = np.linspace(0,150,15000)
    # 采样率
    sampleRate = 20
    # 采样点
    nPoint = 15000
    lowCut = 0.01
    HighCut = 5

    #从第32行开始读
    data_origin = pd.read_csv(file, header=32)
    beforeTime_690 = np.arange(15000 * 53,dtype = float).reshape(15000, 53)
    frequency_690 = np.arange(15000 * 53,dtype = float).reshape(15000, 53)
    filter_690 = np.arange(15000 * 53, dtype=float).reshape(15000, 53)
    afterTime_690 = np.arange(15000 * 53, dtype=float).reshape(15000, 53)
    #读入数据
    for i in range(1,54):
        beforeTime_690[:, i-1] = data_origin['CH'+str(i)+'(690)'].tolist()
    for i in range(1,54):
        #傅里叶变换到频域空间
        frequency_690[:, i-1] = np.fft.fft(beforeTime_690[:,i-1])
        # 带通滤波
        filter_690[:,i-1] = butter_bandpass_filter(frequency_690[:,i-1],lowCut,HighCut,sampleRate,order=6)
        #反傅里叶变换到时域空间
        afterTime_690[:,i-1] = np.fft.ifft(filter_690[:,i-1])


    ## 画图显示
    plt.figure(figsize=(15,8))
    plt.subplot(4,1,1)
    plt.xlabel("Time")
    plt.ylabel("Total Concentration")
    plt.plot(time,beforeTime_690[:,1])
    plt.plot(time, beforeTime_690[:, 20])
    plt.plot(time, beforeTime_690[:, 30])

    plt.subplot(4, 1, 2)
    plt.xlabel("Frequency")
    plt.ylabel("Energy")
    #??问题：怎么设置这个区间长度，如何知道变换完后的频率大小
    #fre = np.arange()
    fre = np.linspace(0,20,len(frequency_690[:,1]))
    plt.plot(fre ,frequency_690[:, 1])
    plt.plot(fre, frequency_690[:, 20])
    plt.plot(fre, frequency_690[:, 30])

    plt.subplot(4, 1, 3)
    plt.xlabel("Frequency")
    plt.ylabel("Energy")
    fre = np.arange(len(filter_690[:, 1]))
    plt.plot(fre, filter_690[:, 1])
    plt.plot(fre, filter_690[:, 20])
    plt.plot(fre, filter_690[:, 30])

    plt.subplot(4, 1, 4)
    plt.xlabel("Time")
    plt.ylabel("Total Concentration")
    fre = np.arange(len(afterTime_690[:, 1]))
    plt.plot(time, afterTime_690[:, 1])
    plt.plot(time, afterTime_690[:, 20])
    plt.plot(time, afterTime_690[:, 30])
    plt.show()
    return afterTime_690

def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='bandpass')
  return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y

if __name__ == '__main__':
    train_path = 'D:\EPIC\DailyTask\\2019-Task\\3-2019\\3-7-2019-抑郁症\FNIRS\数据集\\0\曹勇的原始数据 2019-03-06 1745.csv'
    data_690_Path = ''
    data1 = preFilter(train_path)


