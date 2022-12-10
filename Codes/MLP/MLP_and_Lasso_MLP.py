from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import time

Fullset = True
if (Fullset):
    filename_end = ''
else:
    filename_end = '_1000'

# load data set from file
path = "/home/xinwang/python/part2/"
print("load data starts")
input = pd.read_csv(path + 'INPUT_preprocessed_data'+filename_end+'.csv')
output = pd.read_csv(path + 'OUTPUT_preprocessed_data'+filename_end+'.csv')

X = input.drop(["SMILES"],axis=1).to_numpy()
Y = output.drop(["SMILES"],axis=1).to_numpy()
print("load data done")
#examine and replace missing values -- Done from Excel
# imputer = impute.SimpleImputer(strategy='median')
# imputer.fit(X)
# X = imputer.transform(X)

###################################### ADD MOLECULAR DESCRIPTORS ###########################
DoMD = False
if (DoMD):
    print("Molecular descriptor starts")
    MD = pd.read_csv(path + 'MD_preprocessed_data'+filename_end+'.csv')
    X_MD = MD.drop(["SMILES"],axis=1).to_numpy()
    X = np.hstack([X,X_MD])
    print("Molecular descriptor done")
##############################################END###########################################
###################################### ADD FingerPrints ###########################
DoFP = False
if (DoFP):
    print("Fingerprints starts")
    FP = pd.read_csv(path + 'FP_preprocessed_data'+filename_end+'.csv')
    X_FP = FP.drop(["SMILES"],axis=1).to_numpy()
    #print(len(X))
    #print(len(X_FP))
    X = np.hstack([X,X_FP])
    #X = X_FP
    #print(len(X[0])) # 1024 + 7
    print("Fingerprints done")
##############################################END###########################################

# Check the length of data
print("Dataset length:")
print("X=({}, {})".format(len(X),len(X[0])))
print("Y=({}, {})".format(len(Y),len(Y[0])))


#normalization
DoNorm=True
if (DoNorm):
    print("Normalization starts")
    scaler = MinMaxScaler(feature_range=(-1,1))
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y)
    print("Normalization done")




DoAnn = True
if (DoAnn):
    # Find best param using Cross validation
    # Cross-Validation
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # https://towardsdatascience.com/cross-validation-and-grid-search-efa64b127c1b
    FindBestParam = False
    if (FindBestParam):
        print("ANN starts")
        param_grid={
            'hidden_layer_sizes':[3,(3,2),(5,2)],
            'activation':['tanh','relu'],
            'alpha':[0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1],
            }

        ann = MLPRegressor()
        ANN_cv = GridSearchCV(estimator = ann,param_grid=param_grid, cv =5, n_jobs=64)
        print("(in ANN) CV starts\n")
        start_cv = time.time()
        ANN_cv.fit(X,Y.ravel())
        end_cv = time.time()
        print("(in ANN) CV  done\n")
        print("computing time: {}min.".format((end_cv - start_cv)/60))

        # Results
        print("--ANN RESULTS--")
        print("best param:")
        print(ANN_cv.best_params_)
        #print("results:")
        #print(ANN_cv.cv_results_)

    # Implement ANN again with the best parameters
    DoANNwithBestParam = True
    if (DoANNwithBestParam):
        print("ANN w/ best param starts")
        alpha = 3e-05
        hl = (5,2)
        activation= 'tanh'
        epochs = 1000
        ann = MLPRegressor(hidden_layer_sizes=hl, alpha = alpha, activation=activation,solver = 'adam',max_iter = epochs,random_state = 5)

        trainingLoss = []
        validationLoss = []

        #partition
        trainX, testX, trainY, testY = train_test_split(X,Y,test_size = 0.3, random_state = 5)
        for epoch in range(epochs):
            ann.partial_fit(trainX, trainY.ravel())
            #training loss and validation loss
            trainingLoss.append(1-ann.score(trainX, trainY))
            validationLoss.append(1-ann.score(testX,testY))
        trainmse = metrics.mean_squared_error(trainY, ann.predict(trainX))
        testmse = metrics.mean_squared_error(testY, ann.predict(testX))
        print("hidden_layer_sizes={}, activation={}, alpha={}\n Tr MSE={:7.5f}, Te MSE={:7.5f}, Sc={:7.5f}".format(hl,activation, alpha, trainmse, testmse, ann.score(testX, testY)))

        #loss plot
        plt.figure(1)
        #plt.subplot(121)
        plt.plot(trainingLoss,label='train', linewidth = 1.0)
        plt.plot(validationLoss,label='test', linewidth = 1.0)
        plt.ylabel('loss',fontsize = 18)
        plt.xlabel('epochs',fontsize = 18)
        #titlestr = "MLP model with set 1 input"
        #plt.title(titlestr,fontsize = 18)
        plt.xlim(right=epochs)
        plt.yscale('log')
        plt.legend(loc = 'upper right',prop = {'size':15})
        plt.ylim(3*1e-3,1)
        plt.tick_params(labelsize = 15)
        plt.savefig('1.png',dpi=300,bbox_inches = 'tight')

        plt.figure(2)
        plt.scatter(scaler.inverse_transform(testY.reshape(-1,1)),scaler.inverse_transform(ann.predict(testX).reshape(-1,1)),color = 'b',s = 10)
        plt.ylabel('Predicted value',fontsize = 18)
        plt.xlabel('True value',fontsize = 18)
        plt.xlim(-10,2)
        plt.ylim(-10,2)
        plt.tick_params(labelsize = 15)
        plt.savefig('1_y.png',dpi=300,bbox_inches = 'tight')


        print("ANN w/ best param done")
        
