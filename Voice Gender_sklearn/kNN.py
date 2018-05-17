import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score


def run(file_path):
    print("K-NEAREST NEIGHBORS ON {}:".format(file_path))
    # loading dataset
    voice = pd.read_csv(file_path)
    voice_X = voice.drop(['label'], axis=1)
    voice_Y = voice['label']
    voice_train_X, voice_test_X, voice_train_Y, voice_test_Y = train_test_split(voice_X, voice_Y, test_size=0.25, stratify=voice_Y)

    # initializing best model to keep for final prediction
    best_model = KNeighborsClassifier()
    best_model_score = 0
    best_k = 0
    best_metric = ''
    best_weight = ''
    # training & predicting over each combination of parameters
    for distMetric in ['manhattan','euclidean','chebyshev']:
        for weight in ['uniform','distance']:
            for k in range(1,21):
                # initializing and training classifier
                clf = KNeighborsClassifier(n_neighbors=k,
                                            weights=weight,
                                            metric=distMetric)
                clf.fit(voice_train_X, voice_train_Y)
                # obtaining test score and saving if best model
                score = clf.score(voice_test_X, voice_test_Y) * 100
                if score > best_model_score:
                    best_model = clf
                    best_model_score = score
                    best_k = k
                    best_metric = distMetric
                    best_weight = weight

    print ("Test Accuracy for kNN Model: {:.2f}%".format(best_model_score))
    print ("Parameters: k = {}, {} distance, {} weight".format(best_k, best_metric, best_weight))

    # computing kappa statistic
    final_prediction = best_model.predict(voice_test_X)
    kappa = cohen_kappa_score(final_prediction, voice_test_Y.as_matrix())
    print ("Kappa Statistic: {:.5f} \n".format(kappa))


#run("./voice.csv")
#run(".voice_reduced.csv")
run("./winebinary.csv")
run("./winebinary_reduced.csv")