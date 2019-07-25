###################################################
# Girish Narayanswamy
# CSCSI 5622 - Intro To Machine Learning
# Homework 5
###################################################

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as GBC

from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np

# Converting Input Data to NP Array Type
# Training and Test Data Feature Columns
cont_cols = [0, 2, 4, 10, 11, 12] # continuous feature columns in train.data file
categ_cols = [1, 3, 5, 6, 7, 8, 9, 13] # categorical feature columns in train.data file

std_scalar = preprocessing.StandardScaler()
one_hot_encode = preprocessing.OneHotEncoder(sparse=False) # one hot encode cat feats

with open("train.data") as file: # reads in training data
    train = [line.rstrip('\n').split(', ') for line in file] # adapted from stack overflow https://stackoverflow.com/questions/3142054/python-add-items-from-txt-file-into-a-list
    train = np.asarray(train) # convert to np array


with open("test.data") as file: # reads in test data
    test = [line.rstrip('\n').split(', ') for line in file] # remove delimiter, EOL, and place into list
    test = np.asarray(test) # convert to np array

# Processing Data from test and train file data arrays
# Train data
train_cont_cols = train[:, cont_cols] # grab the continuous features
train_categ_cols = train[:, categ_cols] # grab the categorical features
train_cont_cols = std_scalar.fit_transform(train_cont_cols) # scale cont features
train_categ_cols = one_hot_encode.fit_transform(train_categ_cols) # one hot encode categorical features
train_x = np.concatenate([train_cont_cols, train_categ_cols], axis=1) # recombine features into single array
train_y = train[:, 14] # last column is classification

# Test data
test_cont_cols = test[:, cont_cols] # grab the continuous features
test_categ_cols = test[:, categ_cols] # grab the categorical features
test_cont_cols = std_scalar.transform(test_cont_cols) # scale cont features
test_categ_cols = one_hot_encode.transform(test_categ_cols) # one hot encode categorical features
test_x = np.concatenate([test_cont_cols, test_categ_cols], axis=1) # recombine features into single array


train_ratio = 0.7 # percent of original set to use for training data
size_train = int(train_x.shape[0]*train_ratio) # size of new training set
rand_idx = np.random.permutation(train_x.shape[0]) # randomly shuffle indexes

train_idx = rand_idx[0:size_train] # get random indexes for training data
test_idx = rand_idx[size_train:train_x.shape[0]] # get random indexes for test data

# train_idx = range(size_train)
# test_idx = range(size_train, train_x.shape[0])

# create new random training and test sets
train_x1 = train_x[train_idx,:]
train_y1 = train_y[train_idx]
test_x1 = train_x[test_idx,:]
test_y1 = train_y[test_idx]

# Using PCA for Dim Reduction
# pca = PCA(n_components=60)
# train_x1 = pca.fit_transform(train_x1)
# test_x1 = pca.transform(test_x1)

#################### TEST CODE ####################
#################### Random Forest ################ 86.4 % accuracy
# rfc = RFC(n_estimators=200, max_depth=20, random_state=0) # create random forest classifier instance
# rfc.fit(train_x1, train_y1) # train with train_x and train_y data
# y_hat = rfc.predict(test_x1) # get output for predicted test_x data
# scr = 0
# for i in range(len(y_hat)):
#     if y_hat[i] == test_y1[i]:
#         scr += 1
#
# print(scr/len(y_hat))
###################################################

#################### SVM ######################### 85.2 % accuracy
# svc = SVC(gamma='auto')
# svc.fit(train_x1, train_y1)
# y_hat = svc.predict(test_x1)
# scr = 0
# for i in range(len(y_hat)):
#     if y_hat[i] == test_y1[i]:
#         scr += 1
#
# print(scr/len(y_hat))
###################################################

#################### Gradient Boosting ################
# gbc = GBC(learning_rate=0.2, n_estimators=200)
# gbc.fit(train_x1, train_y1)
# y_hat = gbc.predict(test_x1)
# scr = 0
# for i in range(len(y_hat)):
#     if y_hat[i] == test_y1[i]:
#         scr += 1
#
# print(scr/len(y_hat))
###################################################

#################### OUTPUT CODE ##################
#################### Random Forest ################
# rfc = RFC(n_estimators=200, max_depth=20, random_state=0) # create random forest classifier instance
# rfc.fit(train_x, train_y) # train with train_x and train_y data
# y_hat = rfc.predict(test_x) # get output for predicted test_x data
#
# # Format to correct output CSV file
# y_hat_cols = np.reshape(y_hat, (y_hat.shape[0], 1))
# index = np.arange(y_hat.shape[0]).reshape((y_hat.shape[0], 1))
# csv_output = np.concatenate((index, y_hat_cols), axis=1)
# np.savetxt("output_predictions.csv", csv_output, fmt="%s", delimiter=',')

#################### SVM #########################
# svc = SVC(gamma='auto')
# svc.fit(train_x, train_y)
# y_hat = svc.predict(test_x)
#
# # Format to correct output CSV file
# y_hat_cols = np.reshape(y_hat, (y_hat.shape[0], 1))
# index = np.arange(y_hat.shape[0]).reshape((y_hat.shape[0], 1))
# csv_output = np.concatenate((index, y_hat_cols), axis=1)
# np.savetxt("output_predictions.csv", csv_output, fmt="%s", delimiter=',')

#################### GBC #########################
gbc = GBC(learning_rate=0.2, n_estimators=200)
gbc.fit(train_x, train_y)
y_hat = gbc.predict(test_x)

# Format to correct output CSV file
y_hat_cols = np.reshape(y_hat, (y_hat.shape[0], 1))
index = np.arange(y_hat.shape[0]).reshape((y_hat.shape[0], 1))
csv_output = np.concatenate((index, y_hat_cols), axis=1)
np.savetxt("output_predictions.csv", csv_output, fmt="%s", delimiter=',')
print("DONEEEE !!!!!!!")







