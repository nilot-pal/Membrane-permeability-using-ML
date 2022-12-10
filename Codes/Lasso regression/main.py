# This code performs the following tasks:
# 1. Based on which set of inputs (set 1-7) the user wants, prepares dataframe (preprocessed and normalized) from the raw dataset --- by Anri Karanovich
# 2.  Computes the optimal alpha (l1 regularization) using cross validation --- by Nilotpal Chakraborty
# 3. Plots learning curve for training and cross validation --- by Nilotpal Chakraborty
# 4. Computes and plots the feature weights vs. alpha. Also finds the features with non-zero weights which will be used as input to the MLP model --- by Anri Karanovich
# 5. Writes performance of the model as output to a file --- by Nilotpal Chakraborty


import numpy as np
import os
from joblib import dump, load
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## User defined functions
from module_1 import Get_Data
from module_2 import plot_curve


# 1. Model Charateristics

## UNCOMMENT the line correspoinding to the set of input data features
# that you are calculating (see the project report, Table 1, for the definition of sets)

ModelType, include_Inp, include_FP, include_MD = ("Set_1", True, False,  False)
#ModelType, include_Inp, include_FP, include_MD = ("Set_2", True, True, False)
#ModelType, include_Inp, include_FP, include_MD = ("Set_3", True, False, True)
#ModelType, include_Inp, include_FP, include_MD = ("Set_4", True, True, True)
#ModelType, include_Inp, include_FP, include_MD = ("Set_5", False, True, False)
#ModelType, include_Inp, include_FP, include_MD = ("Set_6", False, False, True)
#ModelType, include_Inp, include_FP, include_MD = ("Set_7", False, True, True)

subset = False          #True: use only 1000 rows; 
                        #False: use the whole dataset
                        
source='csv'            # 'csv': read data from the pre-compiled files by Seokgyum
                        # 'txt': read data directly from the raw_data_file

          
raw_data_file = 'oc8b00718_si_002.txt'  
                        #the raw data file name (ONLY USED for source='txt') 
                        
skiprows = 28           # how many rows tro skip in raw_data_file

read_from_saved = True  #True: read the pre-computed data from an atuomatically
                        #     saved file (saves time when adjusting the plots etc.)
                        #False: do calculations from scratch;
                        
tol = 1e-5              # Tolerance used for the LASSO models
max_iter = 50000        # maximum number of iterations foe the LASSO models

N_print_w = 20          # print (at most) this many largest weigths from the best model
N_color_w = 10          # in the evolution curve, color (at most) this many weight curves
                           
random_seed_1 = 401356  #random_state values for data splitting and LassoCV
random_seed_2 = 79346


############################################################
############################################################

# 2. Designate an output subfolder
slash = os.sep         #exact OS-dependent path format will be provided by pathlib-Path()
outchar = ''
if include_Inp: outchar = outchar + '_Inp'
if include_FP: outchar = outchar + '_FP'
if include_MD: outchar = outchar + '_MD'
if subset: outchar = outchar + '_1000'

outdir = os.getcwd() + slash + ModelType + outchar + slash
outdir_p = Path(outdir)
if not os.path.exists(outdir_p): os.mkdir(outdir_p)

# 3. Data Preparation
        #Get_Data now returns dataframes, but ALREADY SCALED
        
scaler = MinMaxScaler()

X,Y = Get_Data(raw_data_file = raw_data_file, skiprows=skiprows, subset=subset,
               include_Inp=include_Inp, include_FP=include_FP, include_MD=include_MD,
               scaler=scaler, source='csv', savepkl = False)       

X_cols = X.columns
N_cols = X_cols.shape[0]

X_new = X.to_numpy()
Y_new = Y.to_numpy()

trainX, testX, trainY, testY = train_test_split(X_new, Y_new, test_size = 0.3, 
                                                random_state = random_seed_1)

print("{}: data is preprocessed and split".format(ModelType))

# 4. Lasso search over alphas with 5 fold cross-validation

lcv_file = Path(outdir + ModelType + outchar + "_LASSO_CV.joblib")

if read_from_saved and os.path.exists(lcv_file):
    model = load(lcv_file)
else:
    model = LassoCV(cv=5, random_state=random_seed_2, tol = tol, max_iter=max_iter)
    model.fit(trainX,trainY.ravel())
    dump(model, lcv_file)
    
plot_curve(X,Y,model.alpha_, outdir)   ## Function call to plot learning curve
   
lasso_best = Lasso(alpha=model.alpha_)   # Find best alpha
lasso_best.fit(trainX,trainY.ravel())
y_pred = lasso_best.predict(testX)

print("{}: LassoCV is finished".format(ModelType))


# 5. Find largest weights

coefs = lasso_best.coef_
intercept = lasso_best.intercept_
ncoef = coefs.size
N_nonzero = np.count_nonzero(coefs)
feat_coefs = [(i, X_cols[i], coefs[i]) for i in range(ncoef)]
            #triples of column indices, column names and their weights for given alpha
feat_nonzero = [x for x in feat_coefs if x[2] != 0]
                                    #sort by absolute value
feat_sorted = sorted(feat_nonzero , key=lambda c: abs(c[2]), reverse=True)
total_weight = sum(np.absolute(feat_sorted[i][2]) for i in range(N_nonzero))
#print(total_weight)

N_print_w = min(N_print_w, N_nonzero)      #N_print_w cannot be larger than 
                                          # number of nonzero columns
feat_largest = feat_sorted[0:N_print_w]
                                    
 # 6. Print  model performance characteristics into a file 
    
perf_file = Path(outdir + ModelType + "_performance.txt")
with open(perf_file, 'w') as f:
    f.write(f'(tol = {tol:.2e}, max_iter = {max_iter})\n')
    f.write('Best l1 regularization (alpha): {:.3e}'.format(model.alpha_))
    print('R squared on training set', round(lasso_best.score(trainX, trainY)*100, 2))
    print('R squared on test set', round(lasso_best.score(testX, testY)*100, 2))
    f.write('\nR squared on training set: {:.6f}'.format(lasso_best.score(trainX, trainY)))
    f.write('\nR squared on test set: {:.6f}'.format(lasso_best.score(testX, testY)))
    f.write('\nMean_squared_error on test set: {:.6f}'.format(mean_squared_error(testY, y_pred)))
    f.write("\n\nNumber of npnzero weights: {}".format(N_nonzero))
#    f.write("\n\nNumber of weights that make up 80% of total weight: {}".format(i))
#    for i in range(N_print_w):
    f.write("\nIntercept: {:.6f}".format(intercept))
#    f.close()
    
print("{}: performance measures recorded".format(ModelType))
    
# 7. Plot MSE vs alphas

plt.semilogx(model.alphas_, model.mse_path_, ":")
plt.plot(
    model.alphas_ ,
    model.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(
    model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
)

plt.legend()
plt.xlabel("alphas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

ymin, ymax = 0, 4
plt.ylim(ymin, ymax);
plt.savefig(Path(outdir+"MSE_vs_alpha_cv_5.png"), dpi=300)
plt.show()

print("{}: MSE vs alpha plot created".format(ModelType))



# 8. Compute weight evolution
al_ex = np.linspace(-6,1,100)
alphas = [10**ex for ex in al_ex]

lasso = Lasso(max_iter=max_iter, warm_start=True, tol=tol)

we_file = Path(outdir + ModelType + outchar + "_weight_evol.npy")

if read_from_saved and os.path.exists(we_file):
    coef_arr = np.load(we_file)
else:
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(trainX,trainY)
        coefs.append(lasso.coef_)
    coef_arr = np.asarray(coefs)
    np.save(we_file, coef_arr)
            #we sort hte coefficient by decreasing magnitude AS they appear in the optimal model

N_color_w = min(N_color_w, N_nonzero)

coef_large_arr =  np.zeros((len(alphas), N_color_w))

for i in range(N_color_w):
    coef_large_arr[:,i] = coef_arr[:,feat_sorted[i][0]]
        #here we pick those coefficients that are top-10 lrgest in the OPTIMAL model
        
labels = [feat_sorted[i][1] for i in range(N_color_w)]    

ax = plt.gca()
ax.plot(alphas, coef_arr, color='lightgray')
for i in range(N_color_w):
    ax.plot(alphas, coef_large_arr[:,i], linewidth=2, label=labels[i])
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha');
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=8)
plt.axvline(
    model.alpha_, linestyle="--", color="k", label="alpha: CV estimate")
    
plt.savefig(Path(outdir+"Weight_Evolution.png"), dpi=300, bbox_inches = 'tight')
plt.show()

print("{}: Weight evolution plot is created".format(ModelType))
print("{}:Analysis complete".format(ModelType))
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")




