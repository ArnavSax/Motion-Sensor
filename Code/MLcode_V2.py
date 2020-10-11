# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

from tkinter import filedialog

import matplotlib.pyplot as plt

#Import Raw Data


fin = open(r"C:\\Users\\ga405_qlutj2w\\Dropbox\\0000_FTR\\motorSensor\\Rehab\\Jiang_Data\\right hand to left knee default.log","r+")
#fin = open("putty.txt", 'r');
#print(fin)
#logfile = filedialog.askopenfilename()
#fin = open(logfile, 'r')

#store data in 2d array
data = [[]]

fin.readline()
fin.readline()

line = fin.readline()

while(line):
#   data.append(line.split('\n'))
   data.append(line.split(','))
   line = fin.readline()

   
data = data[1:]
#print(data)


#Data Visualization
aX = []
aY = []
aZ = []
t = []


#print(data)

for log in data:
    aX.append(float(log[0]))
    aY.append(float(log[1])) ,
    aZ.append(float(log[2]))
    t.append(float((log[9])[:-4]))
   
plt.plot(t, aZ)
plt.xlabel('time')
plt.ylabel("Acceleration Z")
plt.show()

plt.plot(t, aY)
plt.xlabel('time')
plt.ylabel("Acceleration Y")
plt.show()


plt.plot(t, aX)
plt.xlabel('time')
plt.ylabel("Acceleration X")
plt.show()
   
#Cleaning Data
start = 200
amount = 2000

data = data[start:amount]


#Chunking Data
#3D arrays
windows = [[[]]]

chunkSize = 20
numChunk = int(len(data)/chunkSize)-1

print(numChunk)
print(len(data))
for i in range(numChunk):
    startIndex = i*chunkSize + 1
    stopIndex = (i+1)*chunkSize
  #  print(startIndex)
 #   print(stopIndex)
 #   print(i)
    windows.append(data[startIndex:stopIndex])
windows = windows[1:]


#Feature Extraction

#Mean Average Value
meanX = []
meanY = []
meanZ = []
t = []

for w in windows:
    sum = 0
    sum2 = 0
    for i in w:
        sum += abs(float(i[2]))
       # print(i[9][:-4])
        sum2 += float((i[9])[:-4])
    meanX.append(sum/chunkSize)
    t.append(sum2/chunkSize)
    
meanX = meanX[1:]
t = t[1:]
    
plt.plot(t, meanX)
plt.xlabel('time')
plt.ylabel("Mean X")
plt.show()


#Root Mean Square
rmsX = []

for w in windows:
    sum = 0
    for i in w:
        sum += float(i[2]) ** 2
    rmsX.append(sum/chunkSize)
rmsX = rmsX[1:]

plt.plot(t, rmsX)
plt.xlabel('time')
plt.ylabel("RMS X")
plt.show()


#Slope Sign Change
sscX = []
for x in range(len(windows)):
    w = windows[x]
    flag = -1
    change = 0
    for i in range(len(w)-1):
        if(w[i+1][2] > w[i][2]):
            curr = 1
        elif(w[i+1][2] < w[i][2]):
            curr = 0
        if(flag != curr):
            change += 1
        flag = change
    sscX.append(change)

sscX = sscX[1:]

plt.plot(t, sscX)
plt.xlabel('time')
plt.ylabel("Slope Sign Change X")
plt.show()


#Positive Peak
posThresh = .1
posX = []
for x in range(len(windows)):
    w = windows[x]
    count = 0
    for i in range(len(w)-1):
        if(float(w[i][2]) >= posThresh):
            count += 1
    posX.append(count)

posX = posX[1:]
plt.plot(t, posX)
plt.xlabel('time')
plt.ylabel("Positive Peak X")
plt.show()
        

#Negative Peak
negThresh = .1
negX = []
for x in range(len(windows)):
    w = windows[x]
    count = 0
    for i in range(len(w)-1):
        if(float(w[i][2]) <= negThresh):
            count += 1
    negX.append(count)

negX = negX[1:]
plt.plot(t, negX)
plt.xlabel('time')
plt.ylabel("Neagtive Peak X")
plt.show()


#Zero Crossing
crossX = []
for x in range(len(windows)):
    w = windows[x]
    change = 0
    for i in range(len(w)-1):
        if(float(w[i+1][2]) > 0 and float(w[i][2]) < 0 or float(w[i+1][2]) < 0 and float(w[i][2]) > 0):
            change += 1
        flag = change
    crossX.append(change)

crossX = crossX[1:]

plt.plot(t, crossX)
plt.xlabel('time')
plt.ylabel("Zero Crossing X")
plt.show()


#splitting trainnig and tresting data



feature_matrix = np.stack([meanX,rmsX,sscX,posX,negX,crossX], axis=1) # Feature Matrix
#print(feature_matrix)


from sklearn import neighbors
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib.colors import ListedColormap
df2 = pd.DataFrame( feature_matrix,
                   columns=['MeanX', 'RMSX', 'SSCX', 'POSX', 'NEGX', 'CROSSX']) # Label Added
print(df2)
rows_nbr = df2.shape[0]

X = np.array(df2.iloc[:rows_nbr, [0,4]])
x_rows, x_columns=X.shape
print(X)
Y = np.array(df2.iloc[:rows_nbr, 5])
#X = X.shape[1:]
print(X.shape)
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size= 0.20)

#oversampler=SMOTE(kind='regular',k_neighbors=2)
#make_pipeline_imb
h = .02
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF']) # colors for the KNN plot 
cmap_bold  = ListedColormap(['#FF0000', '#0000FF'])
def accuracy(k, X_train, y_train, X_test, y_test):
    '''
    compute accuracy of the classification based on k values 
    '''
    # instantiate learning model and fit data
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)    
    knn.fit(X_train, y_train)
    
    # predict the response
    pred = knn.predict(X_test)
    
    # evaluate and return  accuracy
    return accuracy_score(y_test, pred)
for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(10, weights=weights)
        clf.fit(X_train, y_train)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
        # Plot also the training points, x-axis = 'Glucose', y-axis = "BMI"
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)   
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("0/1 outcome classification (k = %i, weights = '%s')" % (5, weights))
    
        fig.savefig(weights +'.png')
    
        # evaluate
        y_expected  = y_test
        y_predicted = clf.predict(X_test)
        
        # print results
print('----------------------------------------------------------------------')
print('Classification report')
print('----------------------------------------------------------------------')
print('\n', classification_report(y_expected, y_predicted))
print('----------------------------------------------------------------------')
print('Accuracy = %5s' % round(accuracy(6, X_train, y_train, X_test, y_test), 3))
print('----------------------------------------------------------------------')
plt.show()


