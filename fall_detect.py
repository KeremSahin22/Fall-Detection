# -*- coding: utf-8 -*-
"""fall_detect.ipynb

Transformed to .py from .ipynb by collab
Kerem Şahin 

PLEASE INSTALL TABULAR LIBRARY
Imports
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate# YOU NEED TABULATE LIBRARY

"""Function Definitions"""

#method is defined to plot clusters of different cluster sizes
def plot_clusters(X, labels):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.show()

#method is defined to plot principal components
def plot_pca(X, labels):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        if label == 1 or label == 'F':
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label='F')
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label='NF')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D Principal Component Analysis')
    plt.legend()
    plt.show()

#this emthod maps the clustering labels of the kmeans into 1 and 0's according to th majority label of that cluster. If a cluster had more Fall than Non Fall, that label of the cluster will be mapped to Fall.
def cluster_prediction_mapping(labels,predictions,cluster_num):
    mapped_predictions = predictions
    for i in range(cluster_num):
        indexes = np.where(predictions == i)[0]#indexes of the ith cluster members
        F_count = np.count_nonzero(labels[indexes] == 1)
        NF_count = np.count_nonzero(labels[indexes] == 0)
        if F_count > NF_count:
            mapped_predictions[indexes] = 1
        else:
            mapped_predictions[indexes] = 0

    return mapped_predictions

#Returns the accuracy according to the cluster mapping methodology
def cluster_accuracy(labels,predictions,cluster_num):
    mapped_predictions = cluster_prediction_mapping(labels,predictions,cluster_num)
    return metrics.accuracy_score(labels, mapped_predictions)

"""Data Read & Preprocessing"""

cur_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(cur_dir,"falldetection_dataset.csv")
data_df = pd.read_csv(path, header=None)
features_df = data_df.drop([0,1], axis=1)
labels_df = data_df.iloc[:,1]
features = features_df.values
labels = labels_df.values

"""PART A"""

#Use PCA as explained. Also, it can be used to detect outliers so remove them.
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)

explained_variance_ratio = pca.explained_variance_ratio_
eig_vals = pca.explained_variance_
sorted_eigenvalues = sorted(eig_vals, reverse=True)
cumulative_var_ratio = np.cumsum(explained_variance_ratio)*100
print(cumulative_var_ratio)

plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Principal Component Analysis')
plt.show()

plot_pca(principal_components,labels)

outlier_sample_idx1 = np.argmax(principal_components[:,0])#selects the outlier in column 0
outlier_sample_idx2 = np.argmax(principal_components[:,1])#selects the outlier in column 1

features_no_outlier = np.delete(features, outlier_sample_idx1, axis=0)
features_no_outlier = np.delete(features_no_outlier, outlier_sample_idx2, axis=0)
labels_no_outlier = np.delete(labels, outlier_sample_idx1, axis=0)
labels_no_outlier = np.delete(labels_no_outlier, outlier_sample_idx2, axis=0)

#Scale the data with the removed outliers
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_no_outlier)#features are scaled
labels_mapped = np.where(labels_no_outlier == 'F', 1, 0)#labels are mapped to 1 and 0

pca2 = PCA(n_components=2)
pc = pca2.fit_transform(features_scaled)
explained_variance_ratio = pca2.explained_variance_ratio_
eig_vals = pca2.explained_variance_
sorted_eigenvalues = sorted(eig_vals, reverse=True)
cumulative_var_ratio = np.cumsum(explained_variance_ratio)*100
print(cumulative_var_ratio)

plt.scatter(pc[:, 0], pc[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Principal Component Analysis')
plt.show()

plot_pca(pc, labels_mapped)

#kmeans with 2 clusters
cluster_num = 2
kmeans = KMeans(n_clusters=cluster_num, random_state=22, n_init='auto')
cluster_predictions = kmeans.fit_predict(pc)
cluster_mapping_score = cluster_accuracy(labels_mapped, cluster_predictions, cluster_num=cluster_num)
print(f"Cluster Mapping Method Accuracy: {cluster_mapping_score}")
plot_clusters(pc, cluster_predictions)

#Try out different k numbers for K-Means
for i in range(2,8):
    print(f"K-Means with {i + 1} Clusters")
    cluster_num = i + 1
    kmeans = KMeans(n_clusters=cluster_num, random_state=22, n_init='auto')
    cluster_predictions = kmeans.fit_predict(pc)
    plot_clusters(pc, cluster_predictions)
    cluster_mapping_score = cluster_accuracy(labels_mapped, cluster_predictions, cluster_num=cluster_num)    
    print(f"Cluster Mapping Method Accuracy: {cluster_mapping_score}\n")

"""PART B"""

pca2 = PCA(n_components=10, random_state=510)
features_projected = pca2.fit_transform(features_scaled)
X_train, X_test, y_train, y_test = train_test_split(features_projected, labels_mapped, test_size=0.3, random_state=777)

svc = SVC()
svc_pol = SVC()

parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[1e-3,1e-2,1e-1,1e0], 'gamma':('scale', 'auto'), 'degree':[0]}
parameters_pol = {'kernel':['poly'], 'C':[1e-3,1e-2,1e-1,1e0], 'gamma':('scale', 'auto'), 'degree': [2,3,4,5,6]}

#cross validation
clf = GridSearchCV(svc, parameters, cv=3) #Grid Search
clf.fit(X_train, y_train)
results = clf.cv_results_
sorted_indices = np.argsort(-results['mean_test_score'])

#cross validation
clf_pol = GridSearchCV(svc_pol, parameters_pol, cv=3) #Grid Search
clf_pol.fit(X_train, y_train)
results_pol = clf_pol.cv_results_
sorted_indices_pol = np.argsort(-results_pol['mean_test_score'])
table_data = []

# Print the accuracy values and parameters in descending order
for index in sorted_indices:
    mean_score = results['mean_test_score'][index]
    params = results['params'][index]
    row = [mean_score]
    for key in sorted(params.keys()):
        row.append(params[key])
    table_data.append(row)

# Print the accuracy values and parameters in descending order
for index in sorted_indices_pol:
    mean_score = results_pol['mean_test_score'][index]
    params = results_pol['params'][index]
    row = [mean_score]
    for key in sorted(params.keys()):
        row.append(params[key])
    table_data.append(row)

# Define the table headers
headers = ["Accuracy"] + sorted(results['params'][0].keys())

# Generate the table in Markdown format
table = tabulate(table_data, headers, tablefmt="pipe")

# Sort the table data based on accuracy in descending order
table_data.sort(reverse=True, key=lambda x: x[0])

# Create a DataFrame from the table data
df = pd.DataFrame(table_data, columns=headers)

# Sort the DataFrame based on accuracy in descending order
df = df.sort_values(by='Accuracy', ascending=False)

print(df)

# to convert df to html table
table_html = df.to_html(index=False)

""" Write the HTML table to a file, use for creating the table
with open('table.html', 'w') as file:
    file.write(table_html)"""

mlp = MLP(max_iter=100000)

parameters_mlp = {'solver' : ('lbfgs', 'adam'), 'alpha':[1e-3,1e-2,1e-1,1e0], 'activation' : ('logistic','relu'), 'hidden_layer_sizes': [(64),(4,4),(8,8),(16,16),(32,32)]}

#cross validation
search = GridSearchCV(mlp, parameters_mlp, cv=3) #Grid Search
search.fit(X_train, y_train)
results_mlp = search.cv_results_
sorted_indices_mlp = np.argsort(-results_mlp['mean_test_score'])
table_data_mlp = []

# Print the accuracy values and parameters in descending order
for index in sorted_indices_mlp:
    mean_score = results_mlp['mean_test_score'][index]
    params = results_mlp['params'][index]
    row = [mean_score]
    for key in sorted(params.keys()):
        row.append(params[key])
    table_data_mlp.append(row)

# Define the table headers and generate table
headers_mlp = ["Accuracy"] + sorted(results_mlp['params'][0].keys())
table_mlp = tabulate(table_data_mlp, headers_mlp, tablefmt="pipe")

# Sorting table based on descending accuracy
table_data_mlp.sort(reverse=True, key=lambda x: x[0])

# Create dataframe for better visualization
df_mlp = pd.DataFrame(table_data_mlp, columns=headers_mlp)

df_mlp = df_mlp.sort_values(by='Accuracy', ascending=False)

print(df_mlp)

# to convert df to html table
table_html_mlp = df_mlp.to_html(index=False)

#Write the HTML table to a file
"""with open('table_mlp.html', 'w') as file:
    file.write(table_html_mlp)"""

"""TEST"""

svc_final = SVC(C=1,gamma='scale',kernel='rbf')
mlp_final = MLP(activation='relu',alpha=0.1,hidden_layer_sizes=(32,32),solver='lbfgs', max_iter=100000)
svc_final = svc_final.fit(X_train,y_train)
mlp_final = mlp_final.fit(X_train,y_train)
svc_predictions = svc_final.predict(X_test)
mlp_predictions = mlp_final.predict(X_test)
print(f"Accuracy of SVC on Test Dataset: {metrics.accuracy_score(y_test,svc_predictions)}")
print(f"Accuracy of MLP on Test Dataset: {metrics.accuracy_score(y_test,mlp_predictions)}")