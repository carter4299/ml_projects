"""
originally jupyter notebook
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from google.colab import drive
#drive.mount('/content/drive')
file_name = '/content/drive/MyDrive/Colab Notebooks/hr_data_.csv' #you may need to change this line depending on the location of your file in Google Drive
with open(file_name, 'r') as file:
    df = pd.read_csv(file_name)

num_rows, num_columns = df.shape
print(f"Dataset \"{file_name}\", has {num_rows} rows and {num_columns} columns.\n")
print(f"Top 7 Rows:\n{df.head(7)}\n")
print(f"Bottom 7 Rows:\n{df.tail(7)}\n")

if df.isnull().any().any():
    print('Columns have null values\n')
else:
    print('Columns have no null values\n')

print(df.columns)
label_counts = df['target'].value_counts()
label_counts.plot(kind='bar')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()
print("\n\t#Since there is almost 5x more 0.0 than 1.0, the program will be biased toward 0.0, predicting target = 0.0 better than target = 1.0, possibly overfitting 0.0\n")

city_counts = df['city'].value_counts().sort_values(ascending=False)
city_counts.plot(kind='bar')
plt.xlabel('City')
plt.ylabel('Count')
plt.show()

top_cities = city_counts.index[:4]
top_cities_count = df.loc[df['city'].isin(top_cities)].shape[0]
other_cities = df['city'].nunique()
other_cities_count = df.shape[0] - top_cities_count
print(f"Number of rows in the top 4 cities:  {top_cities_count}")
print(f"Number of rows in the remaining {other_cities - 4} cities: {other_cities_count}\n")

df['city'] = df['city'].apply(lambda x: 'city_others' if x not in top_cities else x)

print(f"There is now 5 city names, {df['city'].unique()}, while still containing data for {df['city'].shape[0]} cities\n")

print(f"There is {df['education_level'].nunique()} levels of Education, {df['education_level'].unique()}\n")


def replace_labels(dataframe,col_name,dict_label):
    dataframe[col_name] = dataframe[col_name].replace(dict_label)
    return dataframe


df = replace_labels(df, 'education_level', {'Graduate': 0, 'Masters': 1, 'Phd': 2})

print(f"There is {df['education_level'].nunique()} levels of Education, {df['education_level'].unique()}\n")

print(f"There is {df['company_size'].nunique()} company sizes, {df['company_size'].unique()}\n{df['company_size'].value_counts()}\n")

company_size_dict = {'<10': 0, '10/49': 1, '50-99': 2, '100-500': 3, '500-999': 4, '1000-4999': 5, '5000-9999': 6, '10000+': 7}
df = replace_labels(df, 'company_size', company_size_dict)

print(f"There is {df['company_size'].nunique()} company sizes, {df['company_size'].unique()}\n{df['company_size'].value_counts()}\n")

print(f"There is {df['last_new_job'].nunique()} last new job time intervals, {df['last_new_job'].unique()}\n{df['last_new_job'].value_counts()}\n")

lnj_dict = {'never': 0, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5}
df = replace_labels(df, 'last_new_job', lnj_dict)

print(f"There is {df['last_new_job'].nunique()} last new job time intervals, {df['last_new_job'].unique()}\n{df['last_new_job'].value_counts()}\n")

df = df.drop(['enrollee_id'], axis=1)
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
df = df.loc[:, ~df.T.duplicated()]
print(df.columns)

from sklearn.preprocessing import MinMaxScaler
numeric_col = df.select_dtypes(include=['int', 'float']).columns
df[numeric_col] = MinMaxScaler().fit_transform(df[numeric_col])

print(f"Here is 6 of the scaled records:\n{df[numeric_col].sample(n=6)}")

from sklearn.preprocessing import LabelEncoder
categorical_col = df.select_dtypes(include=['object']).columns
for col in categorical_col:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(['target'], axis=1)
Y = df['target']

value_counts = Y.value_counts()
print(f"\nThere is {value_counts[1]} 1's, and {value_counts[0]} 0's")
print(f"Ratio of 1 and 0 in Y is: {value_counts[1] / value_counts[0]:.2f}\n")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
print(f"There is {sum(y_train==1)} 1's, and {sum(y_train==0)} 0's in y_train")
print(f"Ratio of 1 and 0 in y_train is: {sum(y_train==1) / sum(y_train==0):.2f}")
print(f"There is {sum(y_test==1)} 1's, and {sum(y_test==0)} 0's in y_test")
print(f"Ratio of 1 and 0 in y_test is: {sum(y_test==1) / sum(y_test==0):.2f}\n")

from imblearn.over_sampling import SMOTENC
smote = SMOTENC(categorical_features=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

print(f"Before over-sampling there is {sum(y_train==1)} 1's, and {sum(y_train==0)} 0's in y_train")
print(f"Before over-sampling the ratio of 1 and 0 in original y_train: {sum(y_train==1) / sum(y_train==0):.2f}")
print(f"After over-sampling there is {sum(y_train_resampled==1)} 1's, and {sum(y_train_resampled==0)} 0's in y_train_resampled")
print(f"\t#The ratio changes to 1, since SMOTENC creates synthetic data to improve imbalance ratios.\nAfter over-sampling the ratio of 1 and 0 in resampled y_train: {sum(y_train_resampled==1) / sum(y_train_resampled==0):.2f}\n")

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(x_train, y_train)

from sklearn.model_selection import GridSearchCV
grid_dtc = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 4, 6, 8, 10],
    'max_features': ['sqrt', 'log2', None]
}
grid_search = GridSearchCV(dtc, grid_dtc, cv=5)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
print(f"The best combination of values for the parameters are: {best_params}\n")

dtc.set_params(**grid_search.best_params_)
dtc.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
y_pred = dtc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"\tDecision Tree Classifier\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}\nConfusion Matrix:\n{cm}\n")

from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(dtc, filled=True, feature_names=x_train.columns, class_names=['0', '1'])
plt.show()

dtc_new = DecisionTreeClassifier(**best_params, random_state=42)
dtc_new.fit(x_train_resampled, y_train_resampled)

y_pred_new = dtc_new.predict(x_test)
accuracy_new = accuracy_score(y_test, y_pred_new)
precision_new = precision_score(y_test, y_pred_new)
recall_new = recall_score(y_test, y_pred_new)
f1_new = f1_score(y_test, y_pred_new)
roc_auc_new = roc_auc_score(y_test, y_pred_new)
cm_new = confusion_matrix(y_test, y_pred_new)
print(f"\tDecision Tree Classifier Using Resampled Data\nAccuracy: {accuracy_new:.4f}, Precision: {precision_new:.4f}, Recall: {recall_new:.4f}, F1 Score: {f1_new:.4f}, ROC AUC: {roc_auc_new:.4f}\nConfusion Matrix:\n{cm_new}\n")
print("\t#The results have a lowered precision, but a higher recall, number of positive predictions, f1 score, and ROC AUC\n")

from sklearn.ensemble import RandomForestClassifier
param_grid_rfc = {
    'n_estimators': [25, 50, 100],
    'max_depth': [1, 5, 10],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [2, 3, 4],
    'criterion': ['gini']
}
rfc = RandomForestClassifier(random_state=42)
grid_search_rfc = GridSearchCV(rfc, param_grid=param_grid_rfc, scoring='accuracy', cv=5)
grid_search_rfc.fit(x_train_resampled, y_train_resampled)
print(f"The best combination of values for the parameters are: {grid_search_rfc.best_params_}\n")

rfc_new = RandomForestClassifier(**grid_search_rfc.best_params_, random_state=42)
rfc_new.fit(x_train_resampled, y_train_resampled)

y_pred_rfc = rfc_new.predict(x_test)
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
precision_rfc = precision_score(y_test, y_pred_rfc)
recall_rfc = recall_score(y_test, y_pred_rfc)
f1_rfc = f1_score(y_test, y_pred_rfc)
roc_auc_rfc = roc_auc_score(y_test, y_pred_rfc)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
print(f"\tRandom Forest Classifier\nAccuracy: {accuracy_rfc:.4f}, Precision: {precision_rfc:.4f}, Recall: {recall_rfc:.4f}, F1 Score: {f1_rfc:.4f}, ROC AUC: {roc_auc_rfc:.4f}\nConfusion Matrix:\n{cm_rfc}\n")

from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(random_state=42)
param_grid_ada = {
    'n_estimators': [25, 50, 100],
    'learning_rate': [0.1, 0.5, 1.0]
}
grid_search_ada = GridSearchCV(adaboost, param_grid=param_grid_ada, cv=5, scoring='accuracy')
grid_search_ada.fit(x_train_resampled, y_train_resampled)

abc = AdaBoostClassifier(**grid_search_ada.best_params_)
abc.fit(x_train_resampled, y_train_resampled)

y_pred_abc = abc.predict(x_test)
accuracy_abc = accuracy_score(y_test, y_pred_abc)
precision_abc = precision_score(y_test, y_pred_abc)
recall_abc = recall_score(y_test, y_pred_abc)
f1_abc = f1_score(y_test, y_pred_abc)
roc_auc_abc = roc_auc_score(y_test, y_pred_abc)
cm_abc = confusion_matrix(y_test, y_pred_abc)
print(f"\tAda Boost Classifier\nAccuracy: {accuracy_abc:.4f}, Precision: {precision_abc:.4f}, Recall: {recall_abc:.4f}, F1 Score: {f1_abc:.4f}, ROC AUC: {roc_auc_abc:.4f}\nConfusion Matrix:\n{cm_abc}\n")

from sklearn.ensemble import GradientBoostingClassifier
param_grid_gbc = {
    'n_estimators': [25, 50, 100],
    'max_depth': [1, 2, 3, 4],
    'learning_rate': [0.01, 0.1, 0.5]
}
gbc = GradientBoostingClassifier(random_state=42)
grid_search_gbc = GridSearchCV(gbc, param_grid=param_grid_gbc, scoring='accuracy', cv=5)
grid_search_gbc.fit(x_train_resampled, y_train_resampled)

gbc_new = GradientBoostingClassifier(**grid_search_gbc.best_params_)
gbc_new.fit(x_train_resampled, y_train_resampled)

y_pred_gbc = gbc_new.predict(x_test)
print(f"\tGradient Boosting Classifier\nAccuracy: {accuracy_score(y_test, y_pred_gbc):.4f}, Precision: {precision_score(y_test, y_pred_gbc):.4f}, Recall: {recall_score(y_test, y_pred_gbc):.4f}, F1 Score: {f1_score(y_test, y_pred_gbc):.4f}, ROC AUC: {roc_auc_score(y_test, y_pred_gbc):.4f}\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_gbc)}\n")





















