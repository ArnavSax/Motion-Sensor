import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

filename = "../Data/Right Arm Straight Reach/Fast Sampling Rate/default.log"

data = pd.read_csv(str(filename), names=["accelX", "accelY", "accelZ", "magneX", "magneY", "magneZ", "gyrosX", "gyrosY", "gyrosZ", "time"]).iloc[1:-1].astype(float)
data["time(ms)"] -= data.iloc[0, data.columns.get_loc('time(ms)')] # scale time on the sensor down

# plt.plot(data["time"], data["accelZ"])
# plt.xlabel('time')
# plt.ylabel("Acceleration Z")
# plt.show()

# plt.plot(data["time"], data["accelY"])
# plt.xlabel('time')
# plt.ylabel("Acceleration Y")
# plt.show()


# plt.plot(data["time"], data["accelX"])
# plt.xlabel('time')
# plt.ylabel("Acceleration X")
# plt.show()



def genMatrix(col):
    # feature extraction --> matrix
    




feature_matrix = np.stack([meanX,rmsX,sscX,posX,negX,crossX], axis=1) # Feature Matrix

def knn(feature_matrix): # knn for arbitrary axis
    df2 = pd.DataFrame( feature_matrix,
                       columns=['Mean', 'RMS', 'SSC', 'POSX', 'NEG', 'CROSS']) # Label Added
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


knn(feature_matrix)

    


            
    
    
    

