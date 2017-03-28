# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.cross_validation import train_test_split


# Flags (set true or false as necessary)
plot_flag = True # plots for skewness and correlation
verbose = True # prints details
ROC_plot = True # plot for ROC curves

# ********************** Data cleaning of training set ********************************

# Read train data
df_train = pd.read_csv("C:\Users\Hiranya\Downloads\MLChallenge/train_indessa.csv")

# Get the dtypes
if verbose:
    df_train.dtypes

# Text cleaning (extract digits from string)
df_train['zip_code'] = df_train['zip_code'].str[0:3]
df_train['zip_code'] = df_train['zip_code'].astype(int)
df_train['last_week_pay'] = df_train['last_week_pay'].str.extract('(\d+)', expand=True)
df_train['last_week_pay'] = df_train['last_week_pay'].astype(float)

# Missing data
if verbose:
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(21)

# Delete variables with more than 15% missing data or high cardinality
df_train.drop(df_train.columns[[38, 15, 24, 36, 23, 5, 9, 17]], axis=1, inplace=True)

# Treat missing data
df_train['collections_12_mths_ex_med']=df_train['collections_12_mths_ex_med'].fillna(0)
df_train['tot_coll_amt']=df_train['tot_coll_amt'].fillna(0)
df_train['tot_cur_bal']=df_train['tot_cur_bal'].fillna(df_train['tot_cur_bal'].min())
df_train['total_rev_hi_lim']=df_train['total_rev_hi_lim'].fillna(df_train['total_rev_hi_lim'].min())
df_train['last_week_pay']=df_train['last_week_pay'].fillna(0)
df_train['revol_util']=df_train['revol_util'].fillna(0)
df_train['delinq_2yrs']=df_train['delinq_2yrs'].fillna(0)
df_train['acc_now_delinq']=df_train['acc_now_delinq'].fillna(0)
df_train['inq_last_6mths']=df_train['inq_last_6mths'].fillna(0)
df_train['open_acc']=df_train['open_acc'].fillna(0)
df_train['pub_rec']=df_train['pub_rec'].fillna(0)
df_train['total_acc']=df_train['total_acc'].fillna(0)
df_train['annual_inc']=df_train['annual_inc'].fillna(0)


if verbose:
    # Just checking that there's no missing data missing
    df_train.isnull().sum().max()

    # Find all unique levels
    print(df_train['term'].unique()) 
    print(df_train['grade'].unique())
    print(df_train['sub_grade'].unique())
    print(df_train['emp_length'].unique()) 
    print(df_train['home_ownership'].unique()) 
    print(df_train['verification_status'].unique()) 
    print(df_train['pymnt_plan'].unique()) 
    print(df_train['purpose'].unique()) 
    print(df_train['addr_state'].unique())
    print(df_train['initial_list_status'].unique()) 
    print(df_train['application_type'].unique()) 

    # Find count of each levels
    print(df_train['term'].value_counts()) 
    print(df_train['grade'].value_counts())
    print(df_train['sub_grade'].value_counts())
    print(df_train['emp_length'].value_counts()) 
    print(df_train['home_ownership'].value_counts()) 
    print(df_train['verification_status'].value_counts()) 
    print(df_train['pymnt_plan'].value_counts()) 
    print(df_train['purpose'].value_counts()) 
    print(df_train['addr_state'].value_counts())
    print(df_train['initial_list_status'].value_counts()) 
    print(df_train['application_type'].value_counts()) 

# Consolidate levels to reduce cardinality
df_train.loc[df_train['home_ownership']=='NONE', 'home_ownership'] = 'OTHER'
df_train.loc[df_train['home_ownership']=='ANY', 'home_ownership'] = 'OTHER'

# Convert emp_length into values from 0 to 10
di = {'< 1 year':0, 'n/a':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5,
'6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}
df_train['emp_length'].replace(di, inplace=True)

# pymnt_plan and application_type do not contribute to prediction
del df_train['pymnt_plan']
del df_train['application_type']

# Label encoding
cols = ['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
'purpose', 'addr_state', 'initial_list_status']
df_cat = df_train[cols] # columns with categorical data
d = defaultdict(preprocessing.LabelEncoder)
# Encode
fit = df_cat.apply(lambda x: d[x.name].fit_transform(x))
# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))
# Using the dictionary to label future data
new_cat = df_cat.apply(lambda x: d[x.name].transform(x))
df_train[cols] = new_cat

# Create new variables
df_train['loan_ratio'] = df_train['annual_inc']/df_train['loan_amnt']
df_train['interest'] = df_train['loan_amnt']*df_train['int_rate']

# Check skewness of variable
if plot_flag:
    plt.figure()
    sns.distplot(df_train['annual_inc'])
    plt.figure()
    sns.distplot(df_train['interest'])
    plt.figure()
    sns.distplot(df_train['dti'])
    plt.figure()
    sns.distplot(df_train['loan_ratio'])

# Treat skewed variables
df_train['annual_inc']=np.log(df_train['annual_inc']+10)
df_train['interest']=np.log(df_train['interest']+10)
df_train['dti']=np.log(df_train['dti']+10)
df_train['loan_ratio']=np.log(df_train['loan_ratio']+10)

if plot_flag:
    # correlation matrix of entire dataset
    plt.figure()
    corrmat = df_train.corr()
    f, ax = plt.subplots()
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')

    # Selected dataset correlation matrix
    plt.figure()
    colls = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'interest', 'int_rate', 'grade', 'sub_grade',
    'total_acc', 'open_acc', 'total_rev_hi_lim', 'revol_bal', 'recoveries', 'collection_recovery_fee']
    cm = np.corrcoef(df_train[colls].values.T)
    sns.heatmap(cm, annot=True, square=True, fmt='.2f', yticklabels=colls, xticklabels=colls)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')

# Delete redundant variables
del df_train['loan_amnt']
del df_train['funded_amnt']
#del df_train['sub_grade']
#del df_train['total_rev_hi_lim']
#del df_train['collection_recovery_fee']
#del df_train['total_acc']
#del df_train['int_rate']

# ********************** Data cleaning of test set ********************************

# Read test data
df_test = pd.read_csv("C:\Users\Hiranya\Downloads\MLChallenge/test_indessa.csv")

# Text cleaning
df_test['zip_code'] = df_test['zip_code'].str[0:3]
df_test['zip_code'] = df_test['zip_code'].astype(int)
df_test['last_week_pay'] = df_test['last_week_pay'].str.extract('(\d+)', expand=True)
df_test['last_week_pay'] = df_test['last_week_pay'].astype(float)

# Delete variables with more than 15% missing data or high cardinality
df_test.drop(df_test.columns[[38, 15, 24, 36, 23, 5, 9, 17]], axis=1, inplace=True)

# Treat missing data
df_test['collections_12_mths_ex_med']=df_test['collections_12_mths_ex_med'].fillna(0)
df_test['tot_coll_amt']=df_test['tot_coll_amt'].fillna(0)
df_test['tot_cur_bal']=df_test['tot_cur_bal'].fillna(df_test['tot_cur_bal'].min())
df_test['total_rev_hi_lim']=df_test['total_rev_hi_lim'].fillna(df_test['total_rev_hi_lim'].min())
df_test['last_week_pay']=df_test['last_week_pay'].fillna(0)
df_test['revol_util']=df_test['revol_util'].fillna(0)
df_test['delinq_2yrs']=df_test['delinq_2yrs'].fillna(0)
df_test['acc_now_delinq']=df_test['acc_now_delinq'].fillna(0)
df_test['inq_last_6mths']=df_test['inq_last_6mths'].fillna(0)
df_test['open_acc']=df_test['open_acc'].fillna(0)
df_test['pub_rec']=df_test['pub_rec'].fillna(0)
df_test['total_acc']=df_test['total_acc'].fillna(0)
df_test['annual_inc']=df_test['annual_inc'].fillna(0)

# Just checking that there's no missing data missing
if verbose:
    df_test.isnull().sum().max()

# Consolidate levels
df_test.loc[df_test['home_ownership']=='NONE', 'home_ownership'] = 'OTHER'
df_test.loc[df_test['home_ownership']=='ANY', 'home_ownership'] = 'OTHER'

# Convert emp_length into values from 0 to 10
di = {'< 1 year':0, 'n/a':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5,
'6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}
df_test['emp_length'].replace(di, inplace=True)

# pymnt_plan and application_type do not contribute to prediction
del df_test['pymnt_plan']
del df_test['application_type']

# Label encoding
cols = ['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
'purpose', 'addr_state', 'initial_list_status']
df_cat_test = df_test[cols]
# Using the old dictionary to label test data
new_cat_test = df_cat_test.apply(lambda x: d[x.name].transform(x))
df_test[cols] = new_cat_test

# Create new variables
df_test['loan_ratio'] = df_test['annual_inc']/df_test['loan_amnt']
df_test['interest'] = df_test['loan_amnt']*df_test['int_rate']

# Treat skewed variables
df_test['annual_inc']=np.log(df_test['annual_inc']+10)
df_test['interest']=np.log(df_test['interest']+10)
df_test['dti']=np.log(df_test['dti']+10)
df_test['loan_ratio']=np.log(df_test['loan_ratio']+10)

# Delete redundant variables
del df_test['loan_amnt']
del df_test['funded_amnt']
#del df_test['sub_grade']
#del df_test['total_rev_hi_lim']
#del df_test['collection_recovery_fee']
#del df_test['total_acc']
#del df_test['int_rate']

# Independent variables
predictors = df_train[df_train.columns.difference(['member_id', 'loan_status'])].columns

#************************ Training and test set ***************************

X = df_train[predictors] # independent variables
y = df_train['loan_status'] # dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#************************ Logistic Regression ***************************

# Create linear regression object
alg_lr = LogisticRegression(random_state=1)

# Train the model using the training sets
alg_lr.fit(X_train, y_train)

# Predict Output on test set
pred_lr = alg_lr.predict(X_test)

# Check score
score_lr = accuracy_score(y_test,pred_lr) # 0.80096463747210889

# True positive rate, False positive rate and ROC-AUC
fpr_lr, tpr_lr, _ = roc_curve(y_test, pred_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

#************************ Neural Networks ***************************

alg_mlp = MLPClassifier(hidden_layer_sizes=(20)) 
alg_mlp.fit(X_train, y_train)
alg_mlp.score(X_train, y_train) 
pred_mlp = alg_mlp.predict(X_test)
score_mlp = accuracy_score(y_test,pred_mlp) # 0.67451749344512313
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, pred_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

#*************************** Random forest *******************************

alg_rf = RandomForestClassifier(random_state=1, n_estimators=300, min_samples_split=4, min_samples_leaf=2)
alg_rf.fit(X_train, y_train)
alg_rf.score(X_train, y_train)
pred_rf = alg_rf.predict(X_test)
score_rf = accuracy_score(y_test,pred_rf) # 0.85525178991337791
fpr_rf, tpr_rf, _ = roc_curve(y_test, pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

#************************ AdaBoosting ***************************

alg_ada = AdaBoostClassifier(n_estimators=300, random_state=1)
alg_ada.fit(X_train, y_train)
alg_ada.score(X_train, y_train)
pred_ada = alg_ada.predict(X_test)
score_ada = accuracy_score(y_test,pred_ada) # 0.85711495263209303
fpr_ada, tpr_ada, _ = roc_curve(y_test, pred_ada)
roc_auc_ada = auc(fpr_ada, tpr_ada)

#************************ Gradient Boosting ***************************

alg_gb = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=1, random_state=0)
alg_gb.fit(X_train, y_train)
alg_gb.score(X_train, y_train)
pred_gb = alg_gb.predict(X_test)
score_gb = accuracy_score(y_test,pred_gb) # 0.88322176895279736
fpr_gb, tpr_gb, _ = roc_curve(y_test, pred_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

#******************** ML submission ********************************
# Predict outcomes of test data and create submission file in the required format

# Train algorithm using original entire training data using the best model
alg_gb.fit(df_train[predictors], df_train['loan_status'])

# Make predictions using original entire test set
predictions_gb = alg_gb.predict_proba(df_test[predictors])[:,1]

# Predictions should be between 0 and 1 (requirement of the challenge)
predictions_gb[predictions_gb == 0] = 0.00001
predictions_gb[predictions_gb == 1] = 0.99999

# Create new dataframe for submission
submission_gb = pd.DataFrame({"member_id":df_test["member_id"], "loan_status":predictions_gb})

# Output to csv
submission_gb.to_csv("submission_gb.csv", index=False, header=True)

#******************** ROC Curves ********************************
# Plot ROC curves of all models on a single plot

if ROC_plot:
    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.plot(fpr_mlp, tpr_mlp, label='Neural Networks (area = %0.2f)' % roc_auc_mlp)
    plt.legend(loc="lower right")
    plt.plot(fpr_rf, tpr_rf, label='Random Forests (area = %0.2f)' % roc_auc_rf)
    plt.legend(loc="lower right")
    plt.plot(fpr_ada, tpr_ada, label='Ada Boosting (area = %0.2f)' % roc_auc_ada)
    plt.legend(loc="lower right")
    plt.plot(fpr_gb, tpr_gb, label='Gradient Boosting (area = %0.2f)' % roc_auc_gb)
    plt.legend(loc="lower right")
    