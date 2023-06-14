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

import pandas as pd
import seaborn as sns

fraud_train = pd.read_csv('data/fraudTrain.csv')
fraud_test = pd.read_csv('data/fraudTest.csv')

# # Data Overview

# ## Train

fraud_train.head()

fraud_train.info()

# **Highlights:**
# - There is no null data in the train dataset.
# - There are 1,296,675 transactions in the train dataset.

# ## Test

fraud_test.head()

fraud_test.info()

# **Highlights:**
# - There is also no null data in the test dataset.
# - There are 555,719 transactions in the test dataset.
# - The test dataset has almost 50% less data than train dataset.

# # Analysis

# ## Target: `is_fraud`

# +
print('Absolute Values:\n')
print('Train:')
print(fraud_train['is_fraud'].value_counts())

print('\n')

print('Test:')
print(fraud_test['is_fraud'].value_counts())

print('\n=======================================\n')

print('Normalized Values:\n')
print('Train:')
print(fraud_train['is_fraud'].value_counts(normalize=True))

print('\n')

print('Test:')
print(fraud_test['is_fraud'].value_counts(normalize=True))
# -

# **Highlight:**
# * Less than 0.5% of all transactions in both datasets (train and test) were fraud.

# ### Charts

sns.countplot(data=fraud_train, x='is_fraud')

sns.countplot(data=fraud_test, x='is_fraud')

# # Univariate analysis


