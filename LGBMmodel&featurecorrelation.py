#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem import SnowballStemmer

df1 = pd.read_excel("hotelfinal.xlsx")

df1.info()


# In[19]:


# Calculate the percentage of empty values for each column
percent_missing = df1.isnull().mean() * 100

# Find the columns with more than 80% missing values
cols_to_drop = percent_missing[percent_missing > 80].index

# Drop the columns with more than 80% missing values
df1 = df1.drop(cols_to_drop, axis=1)

df1.info()


# In[26]:


df = df1.select_dtypes(include='number')
df = df.drop(columns = ['row_id','aggregaterating/bestrating', 'aggregaterating/worstrating', 'currenciesaccepted'], axis=1)
df['amenityfeaturecount'].dropna(inplace=True)
df.fillna(df.mean(), inplace=True)
df.info()
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
kmo_model


# In[3]:


from factor_analyzer import FactorAnalyzer
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.fit(df)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev


# In[4]:


# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# In[31]:


corr_df = df1.select_dtypes(include='number')

corr_df= corr_df.drop(columns = ['row_id','aggregaterating/bestrating', 'aggregaterating/worstrating', 'currenciesaccepted'] , axis=1)

corr_df.fillna(corr_df.mean(), inplace=True)

# Calculate the correlation matrix using kendall correlation coefficient
corr_matrix = corr_df.corr(method='pearson')

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True, square=True)
corr_matrix.to_csv('corr.csv')

# Add pairplot for more information
sns.pairplot(corr_df, diag_kind='kde', corner=True)
plt.figure(figsize=(20,15))
plt.show()


# In[20]:


#onehotencoding the pricecurrency
encoder = OneHotEncoder(handle_unknown='ignore')

encoder_curr = pd.DataFrame(encoder.fit_transform(df1[['makesoffer/pricecurrency']]).toarray())

encoder_curr.columns = encoder.get_feature_names_out(['makesoffer/pricecurrency'])

final_df = pd.concat([df1, encoder_curr], axis=1)
final_df.info()


# In[16]:


#run this if didn't drop the columns with more than 80% missing values

# Convert the column to lowercase
#final_df['paymentaccepted_translated'] = final_df['paymentaccepted_translated'].str.lower()

# Remove numbers and punctuation, but keep the comma
#final_df['paymentaccepted_translated'] = final_df['paymentaccepted_translated'].apply(lambda x: re.sub(r'[^\w\s,]', '', str(x)))

# Stem the words using the SnowballStemmer
#stemmer = SnowballStemmer('english')
#final_df['paymentaccepted_list'] = final_df['paymentaccepted_translated'].apply(lambda x: [stemmer.stem(word) for word in x.split(',')])


# Remove white space
#final_df['paymentaccepted_list'] = final_df['paymentaccepted_list'].apply(lambda x: [word.strip() for word in x])

#encoding payment
#final_df['paymentaccepted_list'] = final_df['paymentaccepted_translated'].str.split(',')


#for row in final_df.loc[final_df.paymentaccepted_list.isnull(), 'paymentaccepted_list'].index:
#    final_df.at[row, 'paymentaccepted_list'] = []
# Creating an instance of the MultiLabelBinarizer class
#mlb = MultiLabelBinarizer()

# Fitting and transforming the 'paymentaccepted_list' column using the MultiLabelBinarizer
#one_hot_encoded = pd.DataFrame(mlb.fit_transform(final_df['paymentaccepted_list']), columns=mlb.classes_, index=final_df.index)

# Concatenating the one-hot encoded column with the original dataframe
#final_df = pd.concat([final_df, one_hot_encoded], axis=1)

# Dropping the original 'paymentaccepted' column
#final_df = final_df.drop(columns=['paymentaccepted_list','paymentaccepted_translated', 'paymentaccepted'], axis=1)

#print(len(one_hot_encoded.columns.tolist()))
#print(one_hot_encoded)


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


# Clean up the strings by removing non-ASCII characters and replacing commas with spaces
final_df['amenityfeature_combined'] = final_df['amenityfeature_combined'].str.lower()

# Remove numbers and punctuation, but keep the comma
final_df['amenityfeature_combined'] = final_df['amenityfeature_combined'].apply(lambda x: re.sub(r'[^\w\s,]', '', str(x)))

# Stem the words using the SnowballStemmer
stemmer = SnowballStemmer('english')
final_df['amenityfeature_list'] = final_df['amenityfeature_combined'].apply(lambda x: [stemmer.stem(word) for word in x.split(',')])

# Remove white space
final_df['amenityfeature_list'] = final_df['amenityfeature_list'].apply(lambda x: [word.strip() for word in x])

# Fill missing values with an empty list
for row in final_df.loc[final_df.amenityfeature_list.isnull(), 'amenityfeature_list'].index:
    final_df.at[row, 'amenityfeature_list'] = []

final_df['amenityfeature_list'] = final_df['amenityfeature_list'].apply(','.join)

stop_words = stopwords.words("english")

vectorizer = CountVectorizer(stop_words=stop_words, max_features=100)

features = vectorizer.fit_transform(final_df['amenityfeature_list'].values.astype('U'))

features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())

# Combine the original dataframe with the embeddings
final_df = pd.concat([final_df, features_df], axis=1)

# Dropping the original 'amenityfeature' column
final_df = final_df.drop('amenityfeature_combined', axis=1)


final_df.info()


# In[43]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Prepare a multiple regression model
X = one_hot_encoded.dropna(axis=1)
y = df1['aggregaterating/ratingvalue']

# Fit a linear regression model to the data
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())


# In[44]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Prepare a multiple regression model
X = features_df.dropna(axis=1)
y = df1['aggregaterating/ratingvalue']

# Fit a linear regression model to the data
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())


# In[22]:


dff = final_df.select_dtypes(include='number')

dff = dff.drop(columns=['row_id', 'aggregaterating/bestrating','aggregaterating/worstrating', 
                                   'nan', 'pricecurrency'], axis=1)



dff.fillna(dff.mean(), inplace=True)
dff.reset_index()
dff.info()

nan_columns = dff.columns[dff.isnull().any()].tolist()
print(nan_columns)

print(dff[nan_columns].isnull().sum())


# In[23]:



# spit the data into 70% training 10% validation and 20% testing
train_size=0.7

X = dff.drop(columns = ['aggregaterating/ratingvalue'], axis = 1).copy()

X.fillna(X.mean(), inplace=True)


nan_columns = X.columns[X.isnull().any()].tolist()
print(nan_columns)

print(X[nan_columns].isnull().sum())


X.info()

feature_list = list(X.columns)

X = np.array(X)

Y = np.array(final_df['aggregaterating/ratingvalue'])

Y_classed = pd.qcut(Y, 5, labels=False)


# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X, Y_classed, train_size=0.7)


# we have to define test_size=0.66(that is 66% of remaining data)
test_size = 0.66
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.66)

print('X train shape: ', (X_train.shape)), print('y train shape: ', (y_train.shape))
print('X valid shape: ', (X_valid.shape)), print('y valid shape: ', (y_valid.shape))
print('X test shape: ', (X_test.shape)), print('y test shape: ', (y_test.shape))


# In[32]:


from sklearn.experimental import enable_halving_search_cv
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
X_test = sc.transform(X_test)

# Define the number of folds for k-fold cross-validation
n_folds = 5

# Create the KFold object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Define a range of hyperparameters to tune
param_dist = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'min_child_samples': [2, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 1e-1, 1],
    'reg_lambda': [0, 1e-1, 1],
    'min_split_gain': [0, 1e-3, 1e-2],
    'min_child_weight': [1e-5, 1e-3, 1e-2]
}

# Define a range of hyperparameters to tune
#param_grid = {
#    'n_estimators': [100, 500, 1000],
#    'max_depth': [5, 7, 10],
#    'min_child_samples': [2, 5, 10]
#}

# Initialize the LGBMClassifier
lgb_clf = lgb.LGBMClassifier()

# Perform random search
random_search = RandomizedSearchCV(estimator=lgb_clf, param_distributions=param_dist, n_iter=10, cv=kf, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

print(random_search.best_estimator_)

# Train a LGBMClassifier with the best hyperparameters found
lgb_best = random_search.best_estimator_

# Perform grid search using HalvingGridSearchCV
#grid_search = HalvingGridSearchCV(estimator=lgb_clf, param_grid=param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train, y_train)

# Train a LGBMClassifier with the best hyperparameters found
#lgb_best = grid_search.best_estimator_

# Train the LightGBM classifier on the training set
lgb_best.fit(X_train, y_train)

# Evaluate the performance of the trained model on the validation set
y_pred_valid = lgb_best.predict(X_valid)
accuracy_valid = accuracy_score(y_valid, y_pred_valid)
precision_valid = precision_score(y_valid, y_pred_valid, average='weighted')
recall_valid = recall_score(y_valid, y_pred_valid, average='weighted')
f1_valid = f1_score(y_valid, y_pred_valid, average='weighted')
print("Accuracy on Validation Set:", accuracy_valid)
print("Precision on Validation Set:", precision_valid)
print("Recall on Validation Set:", recall_valid)
print("F1-score on Validation Set:", f1_valid)

# Evaluate the performance of the trained model on the testing set
y_pred_test = lgb_best.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted')
recall_test = recall_score(y_test, y_pred_test, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print("Accuracy on Testing Set:", accuracy_test)
print("Precision on Testing Set:", precision_test)
print("Recall on Testing Set:", recall_test)
print("F1-score on Testing Set:", f1_test)


# In[29]:


from sklearn.model_selection import KFold

# Initialize the LGBMClassifier
lgb_clf = lgb.LGBMClassifier()


lgb_clf.fit(X_train, y_train)

# Evaluate the performance of the trained model on the validation set
y_pred_valid = lgb_clf.predict(X_valid)
accuracy_valid = accuracy_score(y_valid, y_pred_valid)
precision_valid = precision_score(y_valid, y_pred_valid, average='weighted')
recall_valid = recall_score(y_valid, y_pred_valid, average='weighted')
f1_valid = f1_score(y_valid, y_pred_valid, average='weighted')
print("Accuracy on Validation Set:", accuracy_valid)
print("Precision on Validation Set:", precision_valid)
print("Recall on Validation Set:", recall_valid)
print("F1-score on Validation Set:", f1_valid)

# Evaluate the performance of the trained model on the testing set
y_pred_test = lgb_clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted')
recall_test = recall_score(y_test, y_pred_test, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print("Accuracy on Testing Set:", accuracy_test)
print("Precision on Testing Set:", precision_test)
print("Recall on Testing Set:", recall_test)
print("F1-score on Testing Set:", f1_test)


# In[25]:


# Get the most frequent class in the training set
baseline_class = np.argmax(np.bincount(y_train))

# Predict the baseline class for the validation and testing sets
y_pred_baseline = np.full(y_valid.shape, baseline_class)

# Calculate the accuracy of the baseline model on the validation set
baseline_accuracy = accuracy_score(y_valid, y_pred_baseline)

# Print the baseline accuracy
print("Baseline Accuracy:", baseline_accuracy)


# In[49]:


plt.hist(Y, bins=50, edgecolor='k')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.title('Distribution of Target Variable')
plt.show()


# In[27]:


import matplotlib.cm as cm
# Create a categorical color map
cmap = cm.get_cmap('Set1')
colors = cmap(y_pred_test / max(y_pred_test))

# Plot the predicted vs actual values
plt.scatter(y_test, y_pred_test, c=colors)

# Add a line for the perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')

# Show the plot
plt.show()


# In[11]:


# Convert X_train to a DataFrame
X_train_df = pd.DataFrame(X_train)

# Plot the feature importance
feature_importance = lgb_best.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.barh(X_train_df.columns[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance Plot")
plt.show()



# In[36]:


# Get numerical feature importances
importances = list(lgb_best.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[12]:




# Calculate the prediction error
error = y_valid - y_pred_valid

plt.scatter(y_valid, error, alpha=0.5)
plt.xlabel("True Value")
plt.ylabel("Prediction Error")
plt.title("Prediction Error Plot")
plt.show()


# In[13]:



# Calculate the residuals
residuals = y_valid - y_pred_valid

plt.scatter(y_pred_valid, residuals, alpha=0.5)
plt.xlabel("Predicted Value")
plt.ylabel("Residual")
plt.title("Residuals Plot")
plt.hlines(y=0, xmin=0, xmax=1, color='red')
plt.show()


# In[52]:


hist, bins = np.histogram(Y, bins=50)
print("hist:", hist)
print("bins:", bins)


# In[26]:


# Calculate the confusion matrix for the testing set
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
# Plot the confusion matrix using seaborn
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Testing Set')
plt.show()


# In[19]:


print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

