import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA

def preprocess(data):
    data['buyprice'] = data['buyprice'].map({'vhigh':4, 'high':3, 'med':2, 'low':1})
    data['maintenance'] = data['maintenance'].map({'vhigh':4, 'high':3, 'med':2, 'low':1})
    data['doors'] = data['doors'].map({'2':2, '3':3, '4':4, '5more':5})
    data['persons'] = data['persons'].map({'2':2, '4':4, 'more':5})
    data['lug_boot'] = data['lug_boot'].map({'small':1, 'med':2, 'big':3})
    data['safety'] = data['safety'].map({'high':3, 'med':2, 'low':1})
    data['class'] = data['class'].map({'unacc':0, 'acc':1, 'good':2, 'vgood':3})
    return data

def reducePCA(data):
    pca = PCA()
    pca.fit(data)
    variances = pca.explained_variance_ratio_
    toKeep = len(variances)
    for i in range(len(variances)):
        if variances[i] < 0.1:
            toKeep = i
            break
    pca = PCA(n_components=toKeep, random_state=0)
    print ("New dimensions: " + str(toKeep))
    return pca.fit_transform(data)


def fit_and_test(dataX, dataY):
    # splitting data into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=0)

    # decision tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(dataX, dataY)
    test_score = dt.score(X_test, Y_test)
    print ("DT test score is: " + str(test_score))

    # naive bayes
    bayes = GaussianNB()
    bayes.fit(dataX, dataY)
    test_score = bayes.score(X_test, Y_test)
    print("NB test score is: " + str(test_score))

    # neural network
    layers = (data.shape[1], data.shape[1])
    nn = MLPClassifier(hidden_layer_sizes=layers, max_iter=1000)
    nn.fit(dataX, dataY)
    test_score = nn.score(X_test, Y_test)
    print("NN test score is: " + str(test_score))


filePath = "./data/car.csv"
colNames = ['buyprice','maintenance','doors','persons','lug_boot','safety','class']


data = pd.read_csv(filePath, names=colNames)

# prepping & preprocessing data, converting strings into numerical values
data = preprocess(data)

# splitting data into X and Y
print ("Base dataset results:")
dataX = data.drop('class', axis=1)
dataY = data['class']
# running models against base dataset
fit_and_test(dataX, dataY)

# dimensionality transformation on X
print ("Reduced Dataset results:")
dataX_pca = reducePCA(dataX)
# running models against transformed dataset
fit_and_test(dataX_pca, dataY)