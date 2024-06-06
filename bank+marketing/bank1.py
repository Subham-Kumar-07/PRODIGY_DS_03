import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings('ignore')
%matplotlib inline

# Load the dataset
df = pd.read_csv('bank-additional.csv', delimiter=';')
df.rename(columns={'y':'deposit'}, inplace=True)

# Display basic information
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.dtypes.value_counts())
print(df.info())
print(df.duplicated().sum())
print(df.isna().sum())

# Identify categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(exclude='object').columns
print(cat_cols)
print(num_cols)

# Describe data
print(df.describe())
print(df.describe(include='object'))

# Plot histograms
df.hist(figsize=(10, 10), color='#cc5500')
plt.show()

# Plot count plots for categorical columns
for feature in cat_cols:
    plt.figure(figsize=(5, 5))
    sns.countplot(x=feature, data=df, palette='Wistia')
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

# Plot box plots for numerical columns
df.plot(kind='box', subplots=True, layout=(2, 5), figsize=(20, 10), color='#7b3f00')
plt.show()

# Remove outliers
column = df[['age', 'campaign', 'duration']]
q1 = column.quantile(0.25)
q3 = column.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df[['age', 'campaign', 'duration']] = column[(column > lower_bound) & (column < upper_bound)]

# Plot box plots again after removing outliers
df.plot(kind='box', subplots=True, layout=(2, 5), figsize=(20, 10), color='#808000')
plt.show()

# Drop non-numeric columns for correlation matrix calculation
df_numeric = df.drop(columns=cat_cols)
corr = df_numeric.corr()

# Display correlation matrix
print(corr)

# Filter correlation matrix for absolute values >= 0.90
corr_filtered = corr[abs(corr) >= 0.90]
sns.heatmap(corr_filtered, annot=True, cmap='Set3', linewidths=0.2)
plt.show()

# Drop highly correlated columns
high_corr_cols = ['emp.var.rate', 'euribor3m', 'nr.employed']
df1 = df.copy()
df1.drop(high_corr_cols, inplace=True, axis=1)

# Encode categorical variables
lb = LabelEncoder()
df_encoded = df1.apply(lb.fit_transform)
print(df_encoded['deposit'].value_counts())

# Split data into independent (X) and dependent (y) variables
x = df_encoded.drop('deposit', axis=1)
y = df_encoded['deposit']
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Define evaluation functions
def eval_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy Score:', acc)
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n', cm)
    print('Classification Report\n', classification_report(y_test, y_pred))

def mscore(model):
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print('Training Score:', train_score)
    print('Testing Score:', test_score)

# Train and evaluate Decision Tree model
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10)
dt.fit(x_train, y_train)
mscore(dt)
ypred_dt = dt.predict(x_test)
print(ypred_dt)
eval_model(y_test, ypred_dt)

# Plot Decision Tree
cn = ['no', 'yes']
fn = x_train.columns
print(fn)
print(cn)
plot_tree(dt, class_names=cn, filled=True)
plt.show()

# Train and evaluate another Decision Tree model with different parameters
dt1 = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=15)
dt1.fit(x_train, y_train)
mscore(dt1)
ypred_dt1 = dt1.predict(x_test)
eval_model(y_test, ypred_dt1)

# Plot the second Decision Tree
plt.figure(figsize=(15, 15))
plot_tree(dt1, class_names=cn, filled=True)
plt.show()