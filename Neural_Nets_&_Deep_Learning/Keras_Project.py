import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import random

# Load data information
data_info = pd.read_csv('TensorFlow_FILES/DATA/lending_club_info.csv', index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


feat_info('mort_acc')

# Load the dataset
df = pd.read_csv('TensorFlow_FILES/DATA/lending_club_loan_two.csv')
df.info()


# Exploratory Data Analysis (EDA)
# Countplot
sns.countplot(x='loan_status', data=df)
plt.show()

# Histogram of the loan_amt column
plt.figure(figsize=(12, 4))
sns.histplot(df['loan_amnt'], kde=False, bins=40)
plt.show()

# Correlation between the continuous feature variables
df.corr()
print(df)

# Correlation Heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.ylim(len(df.corr())-1, 0)
plt.show()

# Explore the feature with the "installment" feature
feat_info('installment')
feat_info('loan_amnt')

sns.scatterplot(x='installment', y='loan_amnt', data=df)
plt.show()

# Boxplot showing the relationship
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.show()

# Calculate the summary statistics for the loan amount, grouped by the loan_status
df.groupby('loan_status')['loan_amnt'].describe()

# Examining Grade and SubGrade
df['grade'].unique()
df['sub_grade'].unique()
feat_info('sub_grade')

# Countplot per grade
sns.countplot(x='grade', data=df, hue='loan_status')
plt.show()

# Countplot per subgrade
plt.figure(figsize=(12, 4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm', hue='loan_status')
plt.show()

# F and G subgrades
f_and_g = df[(df['grade'] == 'G') | (df['grade'] == 'F')]
plt.figure(figsize=(12, 4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade', data=f_and_g, order=subgrade_order, palette='coolwarm', hue='loan_status')
plt.show()

# Creating 'loan_repaid' column
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
print(df[['loan_repaid', 'loan_status']])

# Barplot for the correlation
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.show()


# Data Pre Processing
# Remove or fill any missing data
df.head()

# Missing Data
# Length of the dataframe
print(len(df))

# Handling Missing Data
df.isnull().sum()

# Convert this Series to be in terms of percentage of the total DataFrame
100 * df.isnull().sum() / len(df)

# Examine what will be okay to drop
feat_info('emp_title')
feat_info('emp_length')

df['emp_title'].nunique()
df['emp_title'].value_counts()

# Convert a dummy variable feature
df.drop('emp_title', axis=1, inplace=True)

# Create a countplot of the emp_length feature column
sorted(df['emp_length'].dropna().unique())
emp_length_order = ['< 1 year',
                    '1 year',
                    '2 years',
                    '3 years',
                    '4 years',
                    '5 years',
                    '6 years',
                    '7 years',
                    '8 years',
                    '9 years',
                    '10 years'
                    ]
plt.figure(figsize=(12, 4))
sns.countplot(x='emp_length', data=df, order=emp_length_order, hue='loan_status')
plt.show()

# Percentage of charge offs per category
emp_co = df[df['loan_status'] == 'Charged Off'].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status'] == 'Fully Paid'].groupby("emp_length").count()['loan_status']
emp_len = emp_co / (emp_co + emp_fp)
print(emp_len)
emp_len.plot(kind='bar')
plt.show()

# Remove the column
df = df.drop('emp_length', axis=1)

# Revisit the DataFrame to see for missing data
df.isnull().sum()

# Drop title column
df = df.drop('title', axis=1)

# Handling mort_acc feature
feat_info('mort_acc')
df['mort_acc'].value_counts()

# Correlation with the mort_acc column
df.corr()['mort_acc'].sort_values()

# Fill the missing values
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df.isnull().sum()

df = df.dropna()
df.isnull().sum()


# Categorical Variables

df['term'] = df['term'].apply(lambda term: int(term[:3]))
df = df.drop('grade', axis=1)

# Convert the subgrade into dummy variables
dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)

# Convert other columns into dummmy variables
dummies = pd.get_dummies(df[['sub_grade', 'verification_status', 'application_type', 'initial_list_status', 'purpose']],
                         drop_first=True)
df = pd.concat([df.drop(['sub_grade', 'verification_status', 'application_type', 'initial_list_status', 'purpose'],
                        axis=1), dummies], axis=1)

# Review the home_ownership column and convert it into dummy variables
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)

# Create a column that extracts zip codes from the address column
df['zip_code'] = df['address'].apply(lambda address: address[-5:])
dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = pd.concat([df.drop('zip_code', axis=1), dummies], axis=1)
df = df.drop(['address', 'issue_d'], axis=1)
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))


# Train-Test Split
df = df.drop('loan_status', axis=1)

X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# Normalizing the Data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Building the Model
# Build a sequential model on the data and fit the model into the training data
model = Sequential([
    Dense(78, activation='relu'),
    Dropout(0.2),
    Dense(39, activation='relu'),
    Dropout(0.2),
    Dense(19, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))
model.save('myfavoritemodel.h5')

# Evaluating Model Performance
# Plot out the validation loss vs the training loss
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()


# Predictions
# Create predictions and display a classification report and confusion matrix
predictions = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# Making a Prediction for a New Customer
random.seed(101)
random_ind = random.randint(0, len(df) - 1)
new_customer = df.drop('loan_repaid', axis=1).iloc[random_ind]
new_customer = scaler.transform(new_customer.values.reshape(1, -1))
print(model.predict(new_customer).round())
print(df.iloc[random_ind]['loan_repaid'])
