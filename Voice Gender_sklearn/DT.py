import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score

def run(file_path):
    print("DECISION TREE ON {}:".format(file_path))
    # static parameters
    folds = 10

    # loading dataset
    voice = pd.read_csv(file_path)
    voice_X = voice.drop(['label'], axis=1)
    voice_Y = voice['label']
    # splitting into test for later
    voice_train_X, voice_test_X, voice_train_Y, voice_test_Y = train_test_split(voice_X, voice_Y, test_size=0.25, stratify=voice_Y)

    # initializing indices for stratified k-fold CV
    skf = StratifiedKFold(n_splits=folds)
    validationScores = []
    models = {}

    # initializing best model to keep for final prediction
    best_model = DecisionTreeClassifier()
    best_model_score = 0

    # training & predicting over each fold
    for train, test in skf.split(voice_train_X, voice_train_Y):
        # splitting into training and validation sets
        X_train = voice_train_X.iloc[train]
        Y_train = voice_train_Y.iloc[train]
        X_test = voice_train_X.iloc[test]
        Y_test = voice_train_Y.iloc[test]
        # initializing and training classifier
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(X_train, Y_train)
        # compiling scores
        validation = clf.score(X_test, Y_test)*100
        validationScores.append(validation)
        # saving best model for use in final prediction
        if validation > best_model_score:
            best_model = clf
            best_model_score = validation

    # finding average model accuracy
    validAcc = np.mean(validationScores)
    print ("Average Test Accuracy of 10-Fold Cross-Validated Model: {:.2f}%".format(validAcc))

    # testing against test set using best model found
    final_prediction = best_model.predict(voice_test_X)
    compare = (final_prediction == voice_test_Y.as_matrix())
    num_correct = len(compare[compare == True])
    final_accuracy = 100 * num_correct / len(voice_test_Y)
    print ("Final Model Test Accuracy: {:.2f}%".format(final_accuracy))

    # computing kappa statistic
    kappa = cohen_kappa_score(final_prediction, voice_test_Y.as_matrix())
    print ("Kappa Statistic: {:.5f} \n".format(kappa))


run("./voice.csv")
run("./voice_reduced.csv")
#run("./winebinary.csv")
#run("./winebinary_reduced.csv")