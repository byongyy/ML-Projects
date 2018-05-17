import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

print ("Conducting PCA Analysis:")
# loading dataset
voice = pd.read_csv("./winebinary.csv")
voice_X = voice.drop(['label'], axis=1)
voice_Y = voice['label']

# initializing and fitting initial PCA model
pca = PCA()
pca.fit(voice_X)
# obtaining explained variances 
variances = pca.explained_variance_ratio_
componentsToKeep = 0
# finding n_component threshold
for i in range(len(variances)):
    if variances[i] < variances[0]/1000:
        componentsToKeep = i + 1
        break

# creating new reduced dataset 
pca = PCA(n_components = componentsToKeep)
voice_reduced_X = pd.DataFrame(pca.fit_transform(voice_X))
voice_reduced = pd.concat([voice_reduced_X, voice_Y], axis=1)
voice_reduced.to_csv('winebinary_reduced.csv', index=False)

print("Number of components kept: {} \n".format(componentsToKeep))