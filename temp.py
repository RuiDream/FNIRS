import numpy as np
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
from sklearn.model_selection import GridSearchCV

def rhythmExtraction(oneFrame, f_low, f_high, fs, frameLength, title):
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

t = range(10000)
fs = 1000
t = np.linspace(1,10000,1000)
y = np.sin(2*np.pi*10*t)+np.sin(2*np.pi*50*t)

y_ = rhythmExtraction(y,40,50,1000,10000,'')
plt.subplot(211)

plt.plot(t,y)
plt.subplot(212)
plt.plot(t,y_)
plt.show()