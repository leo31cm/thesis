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

#df = pd.read_excel("preprocessing_hotel.xlsx")

# normalize the rating value
#file_df['aggregaterating/ratingvalue'].replace('', np.nan, inplace=True)
#df = file_df.copy()
#df.dropna(subset=['aggregaterating/ratingvalue'], inplace=True)
#x = df['aggregaterating/ratingvalue'].values.reshape(-1, 1) #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#df['aggregaterating/ratingvalue'] = x_scaled


#df['pricerange_All'] = df.pricerange.str.findall(r'([0-9,.]+(?:\.[0-9,.]+)?)')


df1 = pd.read_excel("hotel.xlsx")

df1.info()

#df1['aggregaterating/bestrating'] = df1['aggregaterating/bestrating'].replace(',', '.').astype(np.float64)

#df1['aggregaterating/ratingcount'] = df1['aggregaterating/ratingcount'].str.replace(r'[^0-9]+', '')

#df1['aggregaterating/ratingcount'] = df1['aggregaterating/ratingcount'].astype(np.float64)

#df1.replace(0, np.nan, inplace=True)

#df1.to_excel('hotel14.xlsx')






#encoder_curr = pd.DataFrame(encoder.fit_transform(df1[['makesoffer/pricecurrency']]))

#encoder_curr.columns = encoder.get_feature_names(['makesoffer/pricecurrency'])

#curr_df = pd.concat([df1, encoder_curr ], axis=1)

#onehotencoding the pricecurrency
encoder = OneHotEncoder(handle_unknown='ignore')

encoder_curr = pd.DataFrame(encoder.fit_transform(df1[['makesoffer/pricecurrency']]).toarray())

encoder_curr.columns = encoder.get_feature_names_out(['makesoffer/pricecurrency'])

final_df = pd.concat([df1, encoder_curr], axis=1)

# Convert the column to lowercase
final_df['paymentaccepted_translated'] = final_df['paymentaccepted_translated'].str.lower()

# Remove numbers and punctuation, but keep the comma
final_df['paymentaccepted_translated'] = final_df['paymentaccepted_translated'].apply(lambda x: re.sub(r'[^\w\s,]', '', str(x)))

# Stem the words using the SnowballStemmer
stemmer = SnowballStemmer('english')
final_df['paymentaccepted_list'] = final_df['paymentaccepted_translated'].apply(lambda x: [stemmer.stem(word) for word in x.split(',')])


# Remove white space
final_df['paymentaccepted_list'] = final_df['paymentaccepted_list'].apply(lambda x: [word.strip() for word in x])

#encoding payment
final_df['paymentaccepted_list'] = final_df['paymentaccepted_translated'].str.split(',')


for row in final_df.loc[final_df.paymentaccepted_list.isnull(), 'paymentaccepted_list'].index:
    final_df.at[row, 'paymentaccepted_list'] = []
# Creating an instance of the MultiLabelBinarizer class
mlb = MultiLabelBinarizer()

# Fitting and transforming the 'paymentaccepted_list' column using the MultiLabelBinarizer
one_hot_encoded = pd.DataFrame(mlb.fit_transform(final_df['paymentaccepted_list']), columns=mlb.classes_, index=final_df.index)

# Concatenating the one-hot encoded column with the original dataframe
final_df = pd.concat([final_df, one_hot_encoded], axis=1)

# Dropping the original 'paymentaccepted' column
final_df = final_df.drop(columns=['paymentaccepted_list','paymentaccepted_translated', 'paymentaccepted'], axis=1)

#print(len(one_hot_encoded.columns.tolist()))
#print(one_hot_encoded)


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

dff = final_df.select_dtypes(include='number')

dff = dff.drop(columns=['row_id', 'aggregaterating/bestrating','aggregaterating/worstrating',
                                   'nan', 'pricecurrency'], axis=1)



dff.fillna(dff.mean(), inplace=True)

dff.info()

#final_df.to_excel("hotel.xlsx", index=False)

#final_df.info()


#correlation

#correl = df1.corr()

#df1 = df1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_regression

#X = df1.drop(columns = ['aggregaterating/ratingvalue', 'row_id', 'makesoffer/pricecurrency']).copy()
#Y = df1['aggregaterating/ratingvalue'].astype('float64')


#best_features = SelectKBest(score_func=f_regression, k=3)
#fit = best_features.fit(X, Y)
#df_scores = pd.DataFrame(fit.scores_)
#df_columns = pd.DataFrame(X.columns)
#features_scores = pd.concat([df_columns, df_scores], axis=1)
#features_scores.columns = ['Features', 'Score']

#print(features_scores.nlargest(5,'Score'))

#correl.to_csv("correlation1.csv")

#df1['paymentaccepted_list'] = df1['paymentaccepted'].str.split(',')
#df1['paymentaccepted_list'].apply(pd.Series)
#df1 = pd.concat([df1,df1['paymentaccepted_list'].apply(pd.Series)], axis=1)
#df_payment = pd.DataFrame(df1['paymentaccepted_list'].tolist()).fillna('').add_prefix('payment_')

#df1 = pd.concat([df1, df_payment], axis=1)
#df1['paymentaccepted'] = df1['paymentaccepted'].astype(str)

#model = EasyNMT('Opus-MT')

#df1['paymentaccepted_en'] = df1['paymentaccepted'].apply(lambda x: model.translate(x, target_lang='en'))


#df1.to_excel("hotel13.xlsx", index=False)



#df1 = df1.replace(r'^\s*$', np.nan, regex=True)
#df['pricerange_All'] = [float(str(i).replace(",", "")) for i in df['pricerange_All']]
#df2 = df1[['makesoffer/pricecurrency']]
#df3 = df1['openinghours/workday']
#df4 = df1['openinghours/weekend']
#df1[['openinghours/weekend_start','openinghours/weekend_closed']] = df4.str.split('-', n=1, expand=True)

#df1['openinghours/workday_start'] = pd.to_datetime(df1['openinghours/workday_start'], errors='coerce')
#df1['openinghours/workday_start_hour'] = df1['openinghours/workday_start'].dt.hour
#df1['openinghours/workday_start_minute'] =df1['openinghours/workday_start'].dt.minute
#df1['openinghours/workday_closed'] = pd.to_datetime(df1['openinghours/workday_closed'], errors='coerce')
#df1['openinghours/workday_closed_hour'] = df1['openinghours/workday_closed'].dt.hour
#df1['openinghours/workday_closed_minute'] =df1['openinghours/workday_closed'].dt.minute
#df1['openinghours/weekend_start'] = pd.to_datetime(df1['openinghours/weekend_start'], errors='coerce')
#df1['openinghours/weekend_start_hour'] = df1['openinghours/weekend_start'].dt.hour
#df1['openinghours/weekend_start_minute'] =df1['openinghours/weekend_start'].dt.minute
#df1['openinghours/weekend_closed'] = pd.to_datetime(df1['openinghours/weekend_closed'], errors='coerce')
#df1['openinghours/weekend_closed_hour'] = df1['openinghours/weekend_closed'].dt.hour
#df1['openinghours/weekend_closed_minute'] =df1['openinghours/weekend_closed'].dt.minute




def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df




#df1['pricecurrency'] = df2


#df1[['openinghours/workday','openinghours/weekend']] = df3.str.split(',', n=1, expand=True)


#print(df2.head())
#sns.countplot(y="aggregaterating/ratingvalue", data=df1)
#plt.title("Counting by stars")

#df1['pricerange_All'] = df1.pricerange_All.str.findall(r"[-+]?(?:\d*\.*\d+)")

#for y in df1.pricerange_All:

    #np_array = np.array(y, dtype=np.float64)
    #print(type(y))

#df1['pricerange_All_new'] = df1['pricerange_All'].apply(lambda row: row[0] if len(row) == 1 else str(np.mean([float(p) for p in row])))

#df1.to_excel('payment.xlsx', index=False)






#des = df.describe()
#des.to_csv('describe.csv', index=False)

#df.to_excel("preprocessing_hotel.xlsx", index=False)

# spit the data into 70% training 10% validation and 20% testing
train_size=0.7


#final_df = final_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)



X = dff.drop(columns = ['aggregaterating/ratingvalue'], axis = 1).copy()

X.fillna(X.mean(numeric_only=True), inplace=True)

X.info()

feature_list = list(X.columns)

X = np.array(X)

Y = np.array(final_df['aggregaterating/ratingvalue'])




# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=0.7)


# we have to define test_size=0.66(that is 66% of remaining data)
test_size = 0.66
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.66)

print('X train shape: ', (X_train.shape)), print('y train shape: ', (y_train.shape))
print('X valid shape: ', (X_valid.shape)), print('y valid shape: ', (y_valid.shape))
print('X test shape: ', (X_test.shape)), print('y test shape: ', (y_test.shape))

# Use the KNN classifier to fit data:
#classifier = KNeighborsClassifier(n_neighbors=5)
#classifier.fit(X_train, y_train)

# Predict y data with classifier:
#y_predict = classifier.predict(X_test)

# Print results:
#print(confusion_matrix(y_test, y_predict))
#print(classification_report(y_test, y_predict))


from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


# Creating a baseline model using the mean strategy
dummy = DummyRegressor(strategy='mean')

# Training the baseline model
dummy.fit(X_train, y_train)

# Predicting on the testing set using the baseline model
y_pred_dummy = dummy.predict(X_test)

# Create a random forest regressor object
rf = RandomForestRegressor()

# Define a range of hyperparameters to tune
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Perform cross-validation to find the best hyperparameters
grid_search = HalvingGridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


print(grid_search.best_params_)

# Train a random forest regressor with the best hyperparameters found
rf_best = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'],
                                 max_depth=grid_search.best_params_['max_depth'],
                                 min_samples_split=grid_search.best_params_['min_samples_split'])
rf_best.fit(X_train, y_train)

# Evaluate the performance of the trained model on the validation set
y_pred = rf_best.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)
print("Validation Set Results:")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Evaluate the performance of the trained model on the testing set
y_pred = rf_best.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Testing Set Results:")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Plot the predicted vs. actual values
plt.scatter(y_test, y_pred)
plt


# Creating a random forest model
#rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Training the random forest model
#rf.fit(X_train, y_train)

# Predicting on the testing set using the random forest model
#y_pred_rf = rf.predict(X_test)

# Computing the mean squared error for the baseline model and the random forest model
#mse_dummy = mean_squared_error(y_test, y_pred_dummy)
#mse_rf = mean_squared_error(y_test, y_pred_rf)

# Outputting the mean squared error for the baseline model and the random forest model
#print('Dummy MSE: %.3f, RF MSE: %.3f' % (mse_dummy, mse_rf))




# Use the forest's predict method on the test data
#predictions = rf.predict(X_test)
# Calculate the absolute errors
#errors = abs(y_pred_rf - y_test)
# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Map the target variable to the 5 classes
y_train_classes = np.digitize(y_train, [1, 2, 3, 4]) - 1
y_test_classes = np.digitize(y_test, [1, 2, 3, 4]) - 1
y_valid_classes = np.digitize(y_valid, [1, 2, 3, 4]) - 1

# Define a random forest classifier and a range of hyperparameters to tune
rf = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train_classes)

# Train a random forest classifier with the best hyperparameters found
rf_best = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                  max_depth=grid_search.best_params_['max_depth'],
                                  min_samples_split=grid_search.best_params_['min_samples_split'])
rf_best.fit(X_train, y_train_classes)

# Evaluate the performance of the trained model on the validation set
y_pred_valid_classes = rf_best.predict(X_valid)
accuracy_valid = accuracy_score(y_valid_classes, y_pred_valid_classes)
classification_report_valid = classification_report(y_valid_classes, y_pred_valid_classes)
print("Accuracy on Validation Set:", accuracy_valid)
print("Classification Report on Validation Set:\n", classification_report_valid)

# Evaluate the performance of the trained model on the testing set
y_pred_test_classes = rf_best.predict(X_test)
accuracy_test = accuracy_score(y_test_classes, y_pred_test_classes)
classification_report_test = classification_report(y_test_classes, y_pred_test_classes)
print("Accuracy on Testing Set:", accuracy_test)
print("Classification Report on Testing Set:\n", classification_report_test)