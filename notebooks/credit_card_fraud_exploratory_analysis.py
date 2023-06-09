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
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pandas.api.types import is_string_dtype, is_bool_dtype, is_categorical_dtype

pd.set_option('display.max_columns', None)
# -

# # 2. Load Data

# +
local_path = '../data/'
kaggle_path = '/kaggle/input/fraud-detection/'

def identify_data_path():
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(kaggle_path):
        return kaggle_path


# +
avaiable_path = identify_data_path()
train_data = os.path.join(avaiable_path, 'fraudTrain.csv')
test_data =  os.path.join(avaiable_path, 'fraudTest.csv')

fraud_train = pd.read_csv(train_data)
fraud_test = pd.read_csv(test_data)


# -

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

# ## A) Data Quality

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

# ## C) Data Preparation

transaction_date = pd.to_datetime(fraud_data['trans_date_trans_time'])
birth_date =  pd.to_datetime(fraud_data['dob'])
fraud_data['age'] = (transaction_date -birth_date) / np.timedelta64(1, 'Y')

# ## D) Univariate Analysis

# ### a. Bar plot for merchant

sns.countplot(x='category', data=fraud_data)
plt.title('Category distribution')
plt.xticks(rotation=90)
plt.show()

# ### b. Histogram for amt

plt.hist(fraud_data['amt'], edgecolor='black')
plt.title('Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Count')
plt.show()

# ### c. Bar plot for gender

sns.countplot(x='gender', data=fraud_data)
plt.title('Gender distribution')
plt.show()

# ### d. Bar plot for state

plt.figure(figsize=(20, 10))
sns.countplot(x='state', data=fraud_data)
plt.title('State distribution')
plt.xticks(rotation=90)
plt.show()

# ### Histogram for age

plt.hist(fraud_data['age'], edgecolor='black')
plt.title('Age Distribution')
plt.show()

# ### e. Bar plot for is_fraud

sns.countplot(x='is_fraud', data=fraud_data)
plt.title('Fraud distribution')
plt.show()

# ## C) Bivariate Analysis

# ### a. Boxplot Amount vs Fraud Classification

sns.boxplot(x='is_fraud', y='amt', data=fraud_data)
plt.title("Amount vs Fraud")
plt.show()

# ### b. Count plot for gender vs fraud

sns.countplot(x='gender', hue='is_fraud', data=fraud_data)
plt.title("Gender vs Fraud")
plt.show()

# ### c. Count plot for category

sns.countplot(x='category', hue='is_fraud', data=fraud_data)
plt.title("Category vs Fraud")
plt.xticks(rotation=90)
plt.show()

# ### d. State vs Fraud 

plt.figure(figsize=(15,10))
sns.countplot(y='state', hue='is_fraud', data=fraud_data)
plt.title("State vs Fraud")
plt.show()

# ### e. Boxplot Age vs Fraud

sns.boxplot(y='age', x='is_fraud', data=fraud_data)
plt.title('Age vs Fraud')
plt.show()

# ## D) Multivariate Analysis

# ### a. Violinplot of Amount against fraud classfication split by gender

sns.violinplot(x='is_fraud', y='amt', hue='gender', data=fraud_data)
plt.title('Fraud vs Amount split by Gender')
plt.show()

# ### b. Violin plot of Age against fraud classification split by gender

sns.violinplot(x='is_fraud', y='age', hue='gender', data=fraud_data)
plt.title('Fraud vs Age Split By Gender')
plt.show()

# ### c. Scatterplot Age vs Amounts vs Fraud

sns.scatterplot(data=fraud_data, x='amt', y='age', hue='is_fraud')
plt.title('Amount vs Age vs Fraud')
plt.show()

# ## E) Correlation

correlation_matrix = fraud_data[['is_fraud', 'age', 'amt']].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(data=correlation_matrix, annot=True)
