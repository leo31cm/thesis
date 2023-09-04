# Importing the required libraries
from langdetect import detect
from textblob import TextBlob
import pandas as pd



# Importing the required libraries
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from translate import Translator
from unidecode import unidecode
from google.cloud import translate_v3 as translate
import os
import pandas as pd
from google.api_core.retry import Retry
from sklearn.preprocessing import OneHotEncoder

#df.read_excel('total_hotel.xlsx')

# Set the path to your credentials JSON file
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\jovial-meridian-377819-7fe13328ba30.json'

# Initialize the Translation API client
#client = translate.TranslationServiceClient()


#def translate_column_values(column, target_language='en'):
#    translate_client = translate.TranslationServiceClient()
#    result = []
#    for value in column:
#        text = str(value)
#        parent = translate_client.location_path('jovial-meridian-377819', 'global')
#        response = translate_client.translate_text(parent=parent, contents=[text], mime_type='text/plain',
#                                                   target_language_code=target_language)
#        result.append(response.translations[0].translated_text)
#    return result

# Choose the column to translate
#column_to_translate = df['amenityfeature']

# Translate the values in the chosen column
#translated_values = translate_column_values(column_to_translate)

# Assign the translated values back to the original DataFrame
#df['amenityfeature'] = translated_values




from easynmt import EasyNMT

# Load the data into a pandas DataFrame
#df = pd.read_excel('total_hotel.xlsx')

#df = df.dropna(subset=['aggregaterating/ratingvalue'])

#printing

#print(df['addresscountry'] ,':', len(df['addresscountry'].unique()))


#print(df.addresscountry.value_counts().sort_values(ascending=False).head(20))

#onehotencoding the address
#encoder = OneHotEncoder(handle_unknown='ignore')

#encoder_curr = pd.DataFrame(encoder.fit_transform(df[['address/addresscountry']]).toarray())

#encoder_curr.columns = encoder.get_feature_names_out(['address/addresscountry'])

#final_df = pd.concat([df, encoder_curr], axis=1)

#final_df.info()



# Concatenate all non-empty string columns into a single column
#string_cols = df.select_dtypes(include=['object']).columns.tolist()
#df['amenityfeature_combined'] = df[string_cols].apply(lambda x: ', '.join(x[x.notnull()].astype(str)), axis=1)

# Finally, set the data type of the new column to string
#df['amenityfeature_combined'] = df['amenityfeature_combined'].astype(str)
#df['paymentaccepted'] = df['paymentaccepted'].astype(str)


# Translate each row in the dataframe
#for i, row in df.iterrows():
#    original_text = row['amenityfeature_combined']
#    original_text = row['paymentaccepted']
#    if original_text.strip():
#        parent = client.location_path('jovial-meridian-377819', 'global')
#        response = client.translate_text(
#            parent=parent,
#            contents=[original_text],
#            target_language_code='en',
#     )

#        translations = response.translations
#        translated_text = translations[0].translated_text
#        df.at[i, 'paymentaccepted_translated'] = translated_text
#        df.at[i, 'amenityfeature_translated'] = translated_text

# Save the translated dataframe to a new Excel file
#df[['amenityfeature_translated']].to_excel('amenityfeature_combined.xlsx', index=False)
#df[['paymentaccepted_translated']].to_excel('paymentaccepted_trans.xlsx', index=False)

# create a new column with the translated text
#df['amenityfeature_combined_translated'] = df['amenityfeature_combined'].apply(lambda x: translator.translate(x))



# First, select the columns containing "amenityfeature" in the column names
#amenity_cols = [col for col in df.columns if 'amenityfeature' in col]

# Create a new dataframe containing only the selected columns
#new_df = df[amenity_cols]

# Write the new dataframe to a new Excel file
#new_df.to_excel('new_file.xlsx', index=False)



# Then, concatenate those columns along the rows to create a new column
#df['amenityfeature_combined'] = df[amenity_cols].apply(lambda x: ''.join(str(e) if e != '' else '[EMPTY]' for e in x), axis=1)



#df['amenityfeature_combined'].to_excel('amenityfeature_combined.xlsx', index=False)
#df.head()

#for row in df.loc[df.paymentaccepted_list.isnull(), 'paymentaccepted_list'].index:
#    df.at[row, 'paymentaccepted_list'] = []
# Creating an instance of the MultiLabelBinarizer class
#mlb = MultiLabelBinarizer()

# Fitting and transforming the 'paymentaccepted_list' column using the MultiLabelBinarizer
#one_hot_encoded = pd.DataFrame(mlb.fit_transform(df['paymentaccepted_list']), columns=mlb.classes_)

# Concatenating the one-hot encoded column with the original dataframe
#df = pd.concat([df, one_hot_encoded], axis=1)

# Dropping the original 'paymentaccepted' column
#df = df.drop('paymentaccepted_list', axis=1)

# Outputting the resulting dataframe
#print(df)

#df.to_excel('onehotpayment.xlsx')

#final_df = pd.read_excel('amenityfeature_combined.xlsx')

# Clean up the strings by removing non-ASCII characters and replacing commas with spaces
#df['amenityfeature_combined'] = df['amenityfeature_combined'].str.replace('[^\x00-\x7F]+', '', regex=True)
#df['amenityfeature_combined'] = df['amenityfeature_combined'].str.replace(',', ' ')

# Fill missing values with an empty string
#df['amenityfeature_combined'] = df['amenityfeature_combined'].fillna('')

# Instantiate the MultiLabelBinarizer
#mlb = MultiLabelBinarizer()

# Fit and transform the cleaned-up column
#df_encoded = pd.DataFrame(mlb.fit_transform(df['amenityfeature_combined'].str.split()), columns=mlb.classes_)

# Concatenate the encoded data with the original data
#df_final = pd.concat([df, df_encoded], axis=1)

# Print the final dataframe
#print(df_final)
#df_final.to_excel("amenityfeature_encoding.xlsx", index=False)
# Fitting and transforming the 'paymentaccepted_list' column using the MultiLabelBinarizer
#one_hot_encoded = pd.DataFrame(mlb.fit_transform(df['paymentaccepted_list']), columns=mlb.classes_)

# Concatenating the one-hot encoded column with the original dataframe
#df = pd.concat([df, one_hot_encoded], axis=1)

