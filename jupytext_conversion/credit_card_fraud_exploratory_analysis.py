# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 1. Import Libraries

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype, is_bool_dtype, is_categorical_dtype

pd.set_option('display.max_columns', None)
# -

# # 2. Load Data

fraud_train = pd.read_csv('data/fraudTrain.csv')
fraud_test = pd.read_csv('data/fraudTest.csv')


# ## a. Check if the both datasets have the same columns:

def match_dataframe_columns(df_train, df_test):
    try:
        return(all(df_train.columns == df_test.columns))
    except:
        return(False)


match_dataframe_columns(fraud_train, fraud_test)

# ## b. Append train and test dataset

fraud_data = pd.concat([fraud_train, fraud_test], ignore_index=False)

fraud_data.head()

fraud_data.tail()

fraud_data.info()

# # 4. Exploratory Data Analysis

# ## Data Quality

# ###  a. Handling duplicates

duplicate_rows_data = fraud_data[fraud_data.duplicated()]
print("Number of duplicated rows: ", duplicate_rows_data.shape)

# ### b. Uniqueness

for column in fraud_data.columns:
    num_distinct_values = len(fraud_data[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")

# ### c. Missing values

print(fraud_data.isnull().sum())

# ### d. Describe data

fraud_data.describe().style.format('{:.2f}')

# # Univariate analysis

# ## Plot distribution of non-numerical features

for column in fraud_data.columns:
    df_column = fraud_data[column]
    if is_categorical_dtype(df_column):
        sns.countplot(data=fraud_data, x=df_column)
        plt.show()

fraud_data.head()

sns.countplot(x='gender', data=fraud_train)
plt.title('Gender Distribution')
plt.show()

# ## Histogram for `amt`

sns.histplot(x='amt', data=fraud_train, bins=30)
plt.title("amt Distribution")
plt.show()


