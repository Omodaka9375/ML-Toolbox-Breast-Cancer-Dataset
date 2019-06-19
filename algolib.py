import sys
import random
import csv
from itertools import izip_longest
import pickle as pk
import struct
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from featureselector import FeatureSelector
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix

# General ML Algorithms toolbox for machine learning training and testing
# Author: Branislav Djalic
# Email: branislav.djalic@gmail.com

algo_list = ['dtc', 'linsvc', 'svc', 'mlp', 'knn', 'gaus', 'lda', 'logreg']

export_path = './model/'

# Data processing tools


def testAlgo(path='', samples=0, algo='', export=False, log=False, standard_scaler=False, minmax_scaler=False):

    if path == '' or algo == '' or samples < 1:
        print('You need to specify all required parameters!')
        sys.exit(0)

    print('Training started . . .')
    X, y = encodelabels(path, row_count=samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=111, shuffle=True)

    if standard_scaler and not minmax_scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif minmax_scaler and not standard_scaler:
            mm = MinMaxScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif standard_scaler and minmax_scaler:
            print('You can only use one scaler at a time- minmax or standard!')
            sys.exit(0)

    if algo == 'dtc':
        model = DecisionTreeClassifier()
    if algo == 'linsvc':
        model = svm.LinearSVC()
    if algo == 'svc':
        model = svm.SVC()
    if algo == 'mlp':
        model = MLPClassifier()
    if algo == 'knn':
        model = KNeighborsClassifier()
    if algo == 'gaus':
        model = GaussianNB()
    if algo == 'lda':
        model = LinearDiscriminantAnalysis()
    if algo == 'logreg':
        model = LogisticRegression()

    model.fit(X_train, y_train)

    trainY = np.array(y_train)
    prediction_train = model.predict(X_train)

    testY = np.array(y_test)
    prediction_test = model.predict(X_test)

    training_accuracy = accuracy_score(trainY, prediction_train)
    test_accuracy = accuracy_score(testY, prediction_test)

    if export:
        export_trained_model(model, algo + '_classifier')
    return training_accuracy, test_accuracy


def testHypersOnAlgo(path='', samples=0, algo=[], hparameters={}, standard_scaler=False, minmax_scaler=False, folds=3):

    if path == '' or len(algo) != len(hparameters) or samples < 1 or len(hparameters) < 1:
        print('You need to specify all required parameters!')
        sys.exit(0)

    print('Hyper training started . . .')

    X, y = encodelabels(path, row_count=samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=111, shuffle=True)

    if standard_scaler and not minmax_scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif minmax_scaler and not standard_scaler:
            mm = MinMaxScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    elif standard_scaler and minmax_scaler:
            print('You can only use one scaler at a time- minmax or standard!')
            sys.exit(0)

    best_score = {}

    if 'dtc' in hparameters:
                tree = DecisionTreeClassifier()
                model = RandomizedSearchCV(tree, hparameters['dtc'], cv=folds)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                par = "\nTuned DTC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('DTC confusion matrix:')
                print(cm)

    if 'linsvc' in hparameters:
                tree = svm.LinearSVC()
                model = RandomizedSearchCV(tree, hparameters['linsvc'], cv=folds)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                par = "\nTuned LinearSVC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('LinearSVC confusion matrix:')
                print(cm)
    if 'svc' in hparameters:
                tree = svm.SVC()
                model = RandomizedSearchCV(tree, hparameters['svc'],cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "\nTuned SVC model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('SVC confusion matrix:')
                print(cm)

    if 'mlp' in hparameters:
                tree =  MLPClassifier()
                model = RandomizedSearchCV(tree, hparameters['mlp'],cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "\nTuned MLP model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('MLP confusion matrix:')
                print(cm)

    if 'knn' in hparameters:
                tree = KNeighborsClassifier()
                model = RandomizedSearchCV(tree, hparameters['knn'],cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "\nTuned KNN model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('KNN confusion matrix:')
                print(cm)

    if 'gaus' in hparameters:
                tree = GaussianNB()
                model = RandomizedSearchCV(tree, hparameters['gaus'],cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par ="\nTuned Gaussian model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('Gaussian confusion matrix:')
                print(cm)

    if 'lda' in hparameters:
                tree = LinearDiscriminantAnalysis()
                model = RandomizedSearchCV(tree, hparameters['lda'],cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "\nTuned LDA model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('LDA confusion matrix:')
                print(cm)

    if 'logreg' in hparameters:
                tree = LogisticRegression()
                model = RandomizedSearchCV(tree, hparameters['logreg'],cv=folds)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                par = "\nTuned LogReg model parameters: {}".format(model.best_params_)
                best_score.update({par: model.best_score_})
                cm = confusion_matrix(y_test, y_pred)
                print(par)
                print('Best score: ' + str(model.best_score_*100) + ' %')
                print('Logreg confusion matrix:')
                print(cm)
        
    return best_score


def run_multiple(path='', algos=[], sample_count=0, log=False):
    print('\n########## Multi-algorithm testing started ##########\n')
    train_score = {}
    test_score = {}
    for i in range(len(algos)):
        training_accuracy, test_accuracy = testAlgo(path=path, algo=algos[i], samples=sample_count, export=False, log=False)
        train_score.update({algos[i]: training_accuracy})
        test_score.update({algos[i]: test_accuracy})
        print('########## ' + algos[i] + ' ##########')
        print("Train set score: {0} %".format(training_accuracy*100))
        print("Test set score: {0} %".format(test_accuracy*100)+ "\n")
    print('\nBest training score: ' + str(max(train_score.items(), key=lambda k: k[1])))
    print('Best test score: '+ str(max(test_score.items(), key=lambda k: k[1])))

def plot(path, sample_size=10):
      df = pd.read_csv(path,nrows=sample_size)
      # print(df["HasDetections"].value_counts())
      df.groupby(['HasDetections',"ProductName"])['MachineIdentifier'].size().unstack().plot(kind='bar',stacked=True)
      # df.groupby(['HasDetections','Census_OSEdition','Platform','Processor','DefaultBrowsersIdentifier','UacLuaenable'])['MachineIdentifier'].size().unstack().plot(kind='bar',stacked=True)

      plt.show()

def analyze(path, plot=False, sample_count=10, save=False):
       print('\nIdentifing bad features:\n')
       df= pd.read_csv(path, sep = ",", nrows=sample_count, low_memory=False)
       X= df.drop(df.columns[-1], axis='columns')
       y= df[df.columns[-1]]
       fs = FeatureSelector(data = X, labels = y)
       fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'classification', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})

       fs.plot_feature_importances(threshold = 0.99, plot_n = 5)

       if plot:
                x_data =X['Platform']
                y_data = y
                _, ax = plt.subplots()
                # Draw bars, position them in the center of the tick mark on the x-axis
                ax.bar(x_data, y_data, color = '#539caf', align = 'center')
                # Draw error bars to show standard deviation, set ls to 'none'
                # to remove line between points
                ax.errorbar(x_data, y_data, color = '#297083', ls = 'none', lw = 2, capthick = 2)
                ax.set_ylabel(y_data)
                ax.set_xlabel(x_data)
                ax.set_title('My plot')
                plt.show()
       if save:
               print('\nRemoving all bad features...')
               df = fs.remove(methods = 'all', keep_one_hot = True)
               df.to_csv('./clean_data/clean_data_to_train.csv')
 

                  
       return None

def encodelabels(path, row_count, log=False):
    df= pd.read_csv(path, sep = ",", nrows=row_count, low_memory=False)

    inputs = df.drop(df.columns[-1], axis='columns')
    target = df[df.columns[-1]]
    feature_list = list(inputs.columns)
    for i in range(len(feature_list)):
        names = LabelEncoder()
        inputs[feature_list[i] + '_n'] = names.fit_transform(inputs[feature_list[i]])

    inputs_n = inputs.drop(feature_list, axis='columns')
    if log:
        print('Features head:' + inputs_n.head())
        print('Target ' + target.head())

    return inputs_n,target

def export_trained_model(model, name):
    import pickle as pk
    path = export_path + name + '.csv'
    pk.dump(model, open(path, 'wb'))
    print('\nModel ' + name + ' saved at ' + path + '\n')

def predict_on_model(path, feature):
    # load the model from disk and predict
    loaded_model = pk.load(open(path, 'rb'))
    result = loaded_model.predict([feature])
    print ('Model: ' + path + ' prediction: ' + result)

def extractTrainData(path, savepath, columns=[], row_count=None):
      print('\n########## ML toolbox starting ##########\n')
      print('Extracting training data...\n')

      if savepath == '' or len(columns) <1 or path == '':
            print('You need to specify all required parameters!')
            sys.exit(0)

      train_list = []
      for i in range (len(columns)):   
            train_list.append(pd.read_csv(path, sep = ",",header=0, nrows=row_count, low_memory=False) [columns[i]])

      export_train_data = izip_longest(*train_list, fillvalue = '')
      with open(savepath, 'w') as myfile:
                  wr = csv.writer(myfile)
                  wr.writerow(columns)
                  wr.writerows(export_train_data)
      myfile.close()
      print('\nRaw train data extracted and cleaned at ' + savepath + '\n')

def extractTestData(path, savepath, columns=[], row_count=None):
      print('\nExtracting testing data . . .')

      if savepath == '' or len(columns) <1 or path == '':
            print('You need to specify all required parameters!')
            sys.exit(0)

      test_list=[]
      for i in range (len(columns)):   
            test_list.append(pd.read_csv(path, sep = ",",header=0, nrows=row_count, low_memory=False) [columns[i]])
      
      export_test_data = izip_longest(*test_list, fillvalue = '')
      with open(savepath, 'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(columns)
            wr.writerows(export_test_data)
      myfile.close()
      print('Raw test data extracted and cleaned at ' + savepath)

# Algorithms

# def autoregressione_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.ar_model import AR
    
#     # shuffle our data use 80:20 for train:test
#     random.shuffle(train_dataset)
#     n = int(len(train_dataset)*.80)

#     # create train and test cases
#     trainData = train_dataset[:n]
#     testData = train_dataset[n:]
#     # fit model
#     model = AR(trainData)
#     model_fit = model.fit()
#     # make prediction
#     train_prediction = model_fit.predict(len(trainData), len(trainData))
#     test_prediction = model_fit.predict(len(testData), len(testData))
#     print('Auto-regression results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'autoregressione_classifier')

# def movingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARMA
#     from random import random
#     # fit model
#     model = ARMA(train_dataset, order=(0, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Moving-average results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'movingaveragee_classifier')

# def ARmovingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARMA
#     from random import random
#     # fit model
#     model = ARMA(data, order=(2, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Autoregressive Moving Average (ARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'ARmovingaveragee_classifier')

# def ARImovingaveragee_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.arima_model import ARIMA
#     from random import random
#     # fit model
#     model = ARIMA(data, order=(1, 1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset), typ='levels')
#     print('Autoregressive Moving Integrated Average (ARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'ARImovingaveragee_classifier')

# def sarima_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     from random import random
#     # fit model
#     model = SARIMAX(train_dataset, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Seasonal Autoregressive Integrated Moving-Average (SARIMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'sarima_classifier')

# def sarimax_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     from random import random
#     # contrived dataset
#     train_dataset = [x + random() for x in range(1, 100)]
#     data2 = [x + random() for x in range(101, 200)]
#     # fit model
#     model = SARIMAX(train_dataset, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     exog2 = [200 + random()]
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset),exog=[exog2])
#     print('Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors(SARIMAX) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'sarimax_classifier')

# def vector_autoregressione_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.vector_ar.var_model import VAR
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = i + random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     # fit model
#     model = VAR(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.forecast(model_fit.y, steps=1)
#     print('Vector Autoregression (VAR) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'vector_autoregressione_classifier')

# def vector_autoregression_movingavr_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.varmax import VARMAX
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     # fit model
#     model = VARMAX(train_dataset, order=(1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     prediction = model_fit.forecast()
#     print('Vector Autoregression Moving-Average (VARMA) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'vector_autoregression_movingavr_classifier')

# def varmaxe_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.statespace.varmax import VARMAX
#     from random import random
#     # contrived dataset with dependency
#     train_dataset = list()
#     for i in range(100):
#         v1 = random()
#         v2 = v1 + random()
#         row = [v1, v2]
#         train_dataset.append(row)
#     data_exog = [x + random() for x in range(100)]
#     # fit model
#     model = VARMAX(train_dataset, exog=data_exog, order=(1, 1))
#     model_fit = model.fit(disp=False)
#     # make prediction
#     data_exog2 = [[100]]
#     prediction = model_fit.forecast(exog=data_exog2)
#     print('Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'varmaxe_classifier')

# def simple_expo_smoothinge_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.holtwinters import SimpleExpSmoothing
#     # fit model
#     model = SimpleExpSmoothing(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Simple Exponential Smoothing (SES) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'simple_expo_smoothinge_classifier')

# def holtwintere_classifier(train_dataset, logs=False, export_model=False):
#     print('Training started . . .')
#     from statsmodels.tsa.holtwinters import ExponentialSmoothing
#     # fit model
#     model = ExponentialSmoothing(train_dataset)
#     model_fit = model.fit()
#     # make prediction
#     prediction = model_fit.predict(len(train_dataset), len(train_dataset))
#     print('Holt Winters Exponential Smoothing (HWES) results: ' + prediction)
#     if export_model:
#         export_model(model_fit, 'holtwintere_classifier')



