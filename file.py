import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle


data = pd.read_csv("Creditcard_data.csv")
# X = data.iloc[:, :-1]
# Y = data.iloc[:, -1]

data_class0 = data[data['Class'] == 0]
data_class1 = data[data['Class'] == 1]
# print(data_class0.shape)
# print(data_class1.shape)

data_class1_over = data_class1.sample(data_class0.shape[0], replace=True)
df_balanced = pd.concat([data_class0, data_class1_over], axis=0)
# print(df_balanced.Class.value_counts())


def cluster_sampling(df, number_of_clusters):
    try:
        # Divide the units into cluster of equal size
        df['cluster_id'] = np.repeat(
            [range(1, number_of_clusters+1)], len(df)/number_of_clusters)

        # Create an empty list
        indexes = []

        # Append the indexes from the clusters that meet the criteria
        # For this formula, clusters id must be an even number
        for i in range(0, len(df)):
            if df['cluster_id'].iloc[i] % 3 == 0:
                indexes.append(i)
        cluster_sample = df.iloc[indexes]
        cluster_sample.drop(['cluster_id'], axis=1, inplace=True)
        return (cluster_sample)

    except:
        print("The population cannot be divided into clusters of equal size!")


sample = []

# Simple Random Sampling
sample.append(df_balanced.sample(385, replace=False))

# Systematic Sampling
indexes = np.arange(0, len(df_balanced), 4)
sample.append(df_balanced.iloc[indexes])
# print(sample[1].shape[0])

# Stratified Sampling
sample.append(df_balanced.groupby(
    'Class', group_keys=False).apply(lambda x: x.sample(192)))

# Convenience Sampling
shuffled = shuffle(df_balanced)
sample.append(shuffled.head(385))

# Cluster Sampling
sample.append(cluster_sampling(df_balanced, 7))
# print(sample[2].shape[0])


X = []
Y = []
accuracy_matrix = pd.DataFrame(columns=['Simple Random', 'Systematic Sampling', 'Stratified Sampling', 'Convenience Sampling', 'Cluster Sampling'], index=[
                               'XGB', 'Logistic', 'Random Forest', 'Decision Tree', 'KNN'])

for i in range(5):
    X.append(sample[i].iloc[:, :-1])
    Y.append(sample[i].iloc[:, -1])
    x_train, x_test, y_train, y_test = train_test_split(
        X[i], Y[i], random_state=104, test_size=0.25, shuffle=True)

    # Model 1 XGB
    xgb_model = XGBClassifier().fit(x_train, y_train)
    xgb_y_predict = xgb_model.predict(x_test)
    xgb_score = accuracy_score(xgb_y_predict, y_test)
    accuracy_matrix.iloc[0, i] = xgb_score

    # Model 2 Logistic Regression
    logr = linear_model.LogisticRegression(max_iter=1000)
    logr.fit(x_train, y_train)
    y_pred_log = logr.predict(x_test)
    score_log_1 = accuracy_score(y_test, y_pred_log)
    accuracy_matrix.iloc[1, i] = score_log_1

    # Model 3 RANDOM FOREST
    clf = RandomForestClassifier(n_estimators=15)
    clf.fit(x_train, y_train)
    y_pred_rf = clf.predict(x_test)
    score_rf_1 = accuracy_score(y_test, y_pred_rf)
    accuracy_matrix.iloc[2, i] = score_rf_1

    # Model 4 DECISION TREE
    model = tree.DecisionTreeClassifier()
    model = model.fit(x_train, y_train)
    y_pred_tree = model.predict(x_test)
    score_tree_1 = accuracy_score(y_test, y_pred_tree)
    accuracy_matrix.iloc[3, i] = score_tree_1

    # Model 5 KNN
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred_gnb = gnb.predict(x_test)
    score_gnb = accuracy_score(y_test, y_pred_gnb)
    accuracy_matrix.iloc[4, i] = score_gnb


print(accuracy_matrix)

accuracy_matrix.to_csv('result.csv')
