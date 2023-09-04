import re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

df1 = pd.read_excel("hotelfinal.xlsx")


# select the numeric variables for the correlation coefficient
#corr_df = df1.select_dtypes(include='number')

#corr_df = corr_df.drop(columns = ['row_id','aggregaterating/bestrating', 'aggregaterating/worstrating', 'currenciesaccepted'] , axis=1)

#corr_df.fillna(corr_df.mean(), inplace=True)

# Calculate the correlation matrix using kendall correlation coefficient
#corr_matrix = corr_df.corr(method='pearson')

# Plot the correlation matrix as a heatmap
#plt.figure(figsize=(20,15))
#sns.heatmap(corr_matrix, annot=True, square=True)
#plt.show()
#corr_matrix.to_csv('corr.csv')

# Add pairplot for more information
#sns.pairplot(corr_df, diag_kind='kde', corner=True)
#plt.figure(figsize=(20,15))
#plt.show()

#features = corr_df.drop('aggregaterating/ratingvalue', axis=1)
#target = df1['aggregaterating/ratingvalue']

# Create the scatter matrix
#fig = px.scatter_matrix(pd.concat([features, target], axis=1))

# Show the plot
#fig.show()

# Calculate the percentage of empty values for each column
percent_missing = df1.isnull().mean() * 100

# Find the columns with more than 80% missing values
cols_to_drop = percent_missing[percent_missing > 80].index

# Drop the columns with more than 80% missing values
df1 = df1.drop(cols_to_drop, axis=1)

df1.info()

#onehotencoding the pricecurrency
encoder = OneHotEncoder(handle_unknown='ignore')

encoder_curr = pd.DataFrame(encoder.fit_transform(df1[['makesoffer/pricecurrency']]).toarray())

encoder_curr.columns = encoder.get_feature_names_out(['makesoffer/pricecurrency'])

encoder_curr = encoder_curr.dropna(axis=1)

final_df = pd.concat([df1, encoder_curr], axis=1)


#onehotencoding the addresscountry

#final_df['address/addresscountry'] = final_df['address/addresscountry'].str.lower()

# Remove numbers and punctuation, but keep the comma
#final_df['address/addresscountry'] = final_df['address/addresscountry'].apply(lambda x: re.sub(r'[^\w\s,]', '', str(x)))

# Remove white space
#final_df['address/addresscountry'] = final_df['address/addresscountry'].str.strip()

#encoder = OneHotEncoder(handle_unknown='ignore')

#encoder_add = pd.DataFrame(encoder.fit_transform(df1[['address/addresscountry']]).toarray())

#encoder_add.columns = encoder.get_feature_names(['address/addresscountry'])

#final_df = pd.concat([final_df, encoder_add], axis=1)



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


# Fill missing values with an empty list
#for row in final_df.loc[final_df.paymentaccepted_list.isnull(), 'paymentaccepted_list'].index:
#    final_df.at[row, 'paymentaccepted_list'] = []

# Creating an instance of the MultiLabelBinarizer class
#mlb = MultiLabelBinarizer()

# Fitting and transforming the 'paymentaccepted_list' column using the MultiLabelBinarizer
#one_hot_encoded = pd.DataFrame(mlb.fit_transform(final_df['paymentaccepted_list']), columns=mlb.classes_, index=final_df.index)

#one_hot_encoded = one_hot_encoded.dropna(axis=1)
# Concatenating the one-hot encoded column with the original dataframe
#final_df = pd.concat([final_df, one_hot_encoded], axis=1)

# Dropping the original 'paymentaccepted' column
#final_df = final_df.drop(columns=['paymentaccepted_list', 'paymentaccepted_translated', 'paymentaccepted'], axis=1)


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Clean up the strings by lowercase
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

features_df = features_df.dropna(axis=1)

# Combine the original dataframe with the embeddings
final_df = pd.concat([final_df, features_df], axis=1)

# Dropping the original 'amenityfeature' column
final_df = final_df.drop('amenityfeature_combined', axis=1)


dff = final_df.select_dtypes(include='number')

dff = dff.drop(columns=['row_id', 'aggregaterating/bestrating','aggregaterating/worstrating',
                                   'nan', 'pricecurrency'], axis=1)



dff.fillna(dff.mean(), inplace=True)

dff.info()

# Threshold for the number of zeros that a column can contain
#zero_threshold = 0.8 * len(final_df)

# Get the count of zeros for each column
#zero_count = (final_df == 0).sum(axis=0)

# Get the columns that contain more zeros than the threshold
#drop_columns = zero_count[zero_count >= zero_threshold].index

# Drop the columns from the dataframe
#final_df = final_df.drop(drop_columns, axis=1)


# spit the data into 70% training 10% validation and 20% testing
train_size = 0.7

X = dff.drop(columns=['aggregaterating/ratingvalue'], axis=1).copy()

feature_list = list(X.columns)

X = np.array(X)

Y = np.array(dff['aggregaterating/ratingvalue'])


# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=0.7)


# we have to define test_size=0.66(that is 66% of remaining data)
test_size = 0.66
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.66)

print('X train shape: ', (X_train.shape)), print('y train shape: ', (y_train.shape))
print('X valid shape: ', (X_valid.shape)), print('y valid shape: ', (y_valid.shape))
print('X test shape: ', (X_test.shape)), print('y test shape: ', (y_test.shape))

from sklearn.experimental import enable_halving_search_cv
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
X_test = sc.transform(X_test)

from sklearn.model_selection import KFold
# Define the number of folds for k-fold cross-validation
n_folds = 5

# Create the KFold object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Creating a baseline model using the mean strategy
dummy = DummyRegressor(strategy='mean')

# Training the baseline model
dummy.fit(X_train, y_train)

# Predicting on the testing set using the baseline model
y_pred_dummy = dummy.predict(X_test)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred_dummy)
print("Mean Squared Error:", mse)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred_dummy)
print("Mean Absolute Error:", mae)

# R-squared
r2 = r2_score(y_test, y_pred_dummy)
print("R-squared:", r2)

# Create a random forest regressor object
rf = RandomForestRegressor()

# Define a range of hyperparameters to tune
#param_grid = {
#    'n_estimators': [100, 500, 1000],
#    'max_depth': [3, 5, 7],
#    'min_samples_split': [2, 5, 10]
#}

# Perform cross-validation to find the best hyperparameters
#grid_search = HalvingGridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
#grid_search.fit(X_train, y_train)


#print(grid_search.best_params_)


# Define a range of hyperparameters to tune
param_grid = {'bootstrap': [True, False],
 'max_depth': [5, 10, 15],
 'max_features': [1.0, 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 600, 1000]}

# Perform random search
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=kf, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

print(random_search.best_estimator_)


# Train a LGBMClassifier with the best hyperparameters found
rf_best = random_search.best_estimator_
#rf_best = grid_search.best_estimator_

# Perform cross-validation to find the best hyperparameters
#grid_search = HalvingGridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
#grid_search.fit(X_train, y_train)

# Train a random forest regressor with the best hyperparameters found
#rf_best = grid_search.best_estimator_


#clf = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_split=2)

# Fit the classifier to the training data
#clf.fit(X_train, y_train)
rf_best.fit(X_train, y_train)
# Evaluate the performance of the trained model on the validation set
#y_pred = clf.predict(X_valid)
y_pred_valid = rf_best.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred_valid)
mae = mean_absolute_error(y_valid, y_pred_valid)
r2 = r2_score(y_valid, y_pred_valid)
print("Validation Set Results:")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Evaluate the performance of the trained model on the testing set
##y_pred = clf.predict(X_test)
y_pred_test = rf_best.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print("Testing Set Results:")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Create a categorical color map
cmap = cm.get_cmap('Set1')
colors = cmap(y_test / max(y_test))

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

# Convert X_train to a DataFrame
X_train_df = pd.DataFrame(X_train)

# Plot the feature importance
feature_importance = rf_best.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.barh(X_train_df.columns[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance Plot")
plt.show()

# Get numerical feature importances
importances = list(rf_best.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



# Calculate the prediction error
error = y_valid - y_pred_valid

plt.scatter(y_valid, error, alpha=0.5)
plt.xlabel("True Value")
plt.ylabel("Prediction Error")
plt.title("Prediction Error Plot")
plt.show()

# Calculate residuals
residuals = y_test - y_pred_test

# Plot fitted vs residual values
plt.scatter(y_pred_test, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Fitted vs Residual Values for Test Set')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()




from sklearn.tree import plot_tree
# Plot the first decision tree in the random forest model
plot_tree(rf_best.estimators_[0], filled=True)

# Add title
plt.title("First Decision Tree in Random Forest Regression Model")

# Show the plot
plt.show()
#import lightgbm as lgb
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Map the target variable to the 5 classes
#y_train_classes = np.digitize(y_train, [0.2, 0.4, 0.6, 0.8]) - 1
#y_test_classes = np.digitize(y_test, [0.2, 0.4, 0.6, 0.8]) - 1
#y_valid_classes = np.digitize(y_valid, [0.2, 0.4, 0.6, 0.8]) - 1

# Define a LightGBM classifier and a range of hyperparameters to tune
#lgb_clf = lgb.LGBMClassifier()
#param_grid = {
#    'n_estimators': [100, 500, 1000],
#    'max_depth': [3, 5, 7],
#    'min_child_samples': [2, 5, 10]
#}

# Perform grid search cross-validation to find the best hyperparameters
#grid_search = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train, y_train_classes)

# Train a LightGBM classifier with the best hyperparameters found and early stopping
#lgb_best = lgb.LGBMClassifier(n_estimators=grid_search.best_params_['n_estimators'],
#                              max_depth=grid_search.best_params_['max_depth'],
#                              min_child_samples=grid_search.best_params_['min_child_samples'],
#                              early_stopping_rounds=10,
 #                             verbose=0)
#lgb_best.fit(X_train, y_train_classes, eval_set=[(X_valid, y_valid_classes)])

# Evaluate the performance of the trained model on the validation set
#y_pred_valid_classes = lgb_best.predict(X_valid)
#accuracy_valid = accuracy_score(y_valid_classes, y_pred_valid_classes)
#precision_valid = precision_score(y_valid_classes, y_pred_valid_classes, average='weighted')
#recall_valid = recall_score(y_valid_classes, y_pred_valid_classes, average='weighted')
#f1_valid = f1_score(y_valid_classes, y_pred_valid_classes, average='weighted')
#print("Accuracy on Validation Set:", accuracy_valid)
#print("Precision on Validation Set:", precision_valid)
#print("Recall on Validation Set:", recall_valid)
#print("F1-score on Validation Set:", f1_valid)

# Evaluate the performance of the trained model on the testing set
#y_pred_test_classes = lgb_best.predict(X_test)
#accuracy_test = accuracy_score(y_test_classes, y_pred_test_classes)
#precision_test = precision_score(y_test_classes, y_pred_test_classes, average='weighted')
#recall_test = recall_score(y_test_classes, y_pred_test_classes, average='weighted')
#f1_test = f1_score(y_test_classes, y_pred_test_classes, average='weighted')
#print("Accuracy on Testing Set:", accuracy_test)
#print("Precision on Testing Set:", precision_test)
#print("Recall on Testing Set:", recall_test)
#print("F1-score on Testing Set:", f1_test)


# Calculate the confusion matrix for the testing set
#cm = confusion_matrix(y_test_classes, y_pred_test_classes)

# Plot the confusion matrix using seaborn
#sns.heatmap(cm, annot=True, cmap='Blues')
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.title('Confusion Matrix for Testing Set')
#plt.show()
