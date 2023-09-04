## importing pandas as pd
import pandas as pd
import numpy as np

# Creating the dataframe
#df = pd.read_excel("total_hotel.xlsx")

# Print the dataframe
#df.info()


#combine features that name include amenityfeature
#df['amenity_feature_all'] = df[[col for col in df.columns if "amenityfeature" in col]].apply(lambda x: ', '.join(x[x.notnull()]), axis = 1)

import openpyxl
import math
from openpyxl import Workbook



# read file
#file_df = pd.read_excel('total_hotel.xlsx')


# load your data into a pandas dataframe
#df = pd.read_excel('total_hotel.xlsx')

# create a new column for the combined rating value
#df['aggregaterating/ratingvalue_new'] = ''

# iterate over the rows of the dataframe and combine the values
#for i, row in df.iterrows():
#    if row['aggregaterating/ratingvalue']:
#        df.at[i, 'aggregaterating/ratingvalue_new'] = row['aggregaterating/ratingvalue']
#    elif row['aggregaterating/1/ratingvalue']:
#        df.at[i, 'aggregaterating/ratingvalue_new'] = row['aggregaterating/1/ratingvalue']
#    elif row['aggregaterating/0/ratingvalue']:
#        df.at[i, 'aggregaterating/ratingvalue_new'] = row['aggregaterating/0/ratingvalue']

# output the updated dataframe

#df = df.drop(columns=['aggregaterating/ratingvalue', 'aggregaterating/1/ratingvalue', 'aggregaterating/0/ratingvalue'])

#df.to_excel('updated_total_hotel.xlsx', index=False)



#file_df = pd.read_excel('updated_total_hotel.xlsx')
# remove rows where aggregaterating/ratingvalue is null
#file_df = file_df[file_df['aggregaterating/ratingvalue_new'].notnull()]

# remove duplicate rows in the name column, keeping the first value
# Keep only FIRST record from set of duplicates
#file_df_first_record = file_df.drop_duplicates(subset=["name", 'aggregaterating/ratingvalue'], keep="first")
#file_df_first_record.to_excel("Duplicates_First_Record.xlsx", index=False)

# Keep only LAST record from set of duplicates
#file_df_last_record = file_df.drop_duplicates(subset=["name", 'aggregaterating/ratingvalue'], keep="last")
#file_df_last_record.to_excel("Duplicates_Last_Record.xlsx", index=False)

# Remove ALL records of a set of duplicates
#file_df_remove_all = file_df.drop_duplicates(subset=["name", 'aggregaterating/ratingvalue'], keep=False)
#file_df_remove_all.to_excel("Duplicates_All_Removed.xlsx", index=False)

# Find what the duplicate were
#duplicate_row_index = file_df.duplicated(subset=["name", 'aggregaterating/ratingvalue'], keep="first")
#all_duplicate_rows = file_df[duplicate_row_index]
#duplicate_rows = all_duplicate_rows.drop_duplicates(subset=["name"], keep="first")
#duplicate_rows.to_excel("Duplicate_Rows.xlsx", index=False)

