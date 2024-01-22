# EDAMI PROJECT WINTER 2023
# Agglomerative hierarchical clustering for clustering objects described by nominal and numerical attributes
#
# Arnau Moran Riera               K-7430
#---------------------------------------

# Load required libraries
import time
import numpy as np                                      # pip install numpy
import pandas as pd                                     # pip install pandas
import matplotlib.pyplot as plt                         # pip install matplotlib
from scipy.cluster.hierarchy import dendrogram, linkage # pip install scipy


def calculate_distance(cluster1, cluster2, link):
    # Implementation of distance calculation based on the chosen linkage method
    if link == 'single':
        # The distance between two clusters is defined as the shortest distance between any two points in the two clusters
        return np.min(np.sqrt(np.sum((cluster1[:, np.newaxis] - cluster2)**2, axis=2)))
    elif link == 'complete':
        # The distance between two clusters is defined as the longest distance between any two points in the two clusters
        return np.max(np.sqrt(np.sum((cluster1[:, np.newaxis] - cluster2)**2, axis=2)))
    elif link == 'average':
        # The distance between two clusters is defined as the average distance between all pairs of points in the two clusters
        return np.mean(np.sqrt(np.sum((cluster1[:, np.newaxis] - cluster2)**2, axis=2)))
    elif link == 'ward':
        # Ward linkage computes the sum of squared differences between each data point and the mean of its cluster, then sums these values over all clusters
        mean1 = np.mean(cluster1, axis=0)
        mean2 = np.mean(cluster2, axis=0)
        return np.sum((mean1 - mean2)**2)
    else:
        raise ValueError("Invalid linkage method")


# Function to encode nominal data 
def encode_nominal(data, nominal_columns):
    # Encode nominal attributes using simple label encoding
    encoded_data = data.copy()
    for column in nominal_columns:
        unique_values = data[column].unique()
        encoding_dict = {value: i for i, value in enumerate(unique_values)}
        encoded_data[column] = data[column].map(encoding_dict)
    return encoded_data

# Main algorith process
def agglomerative_hierarchical_clustering(data, link, nominal_columns, nclusters):

    # Store start time
    start_time = time.time()

    # Encode nominal attributes
    if nominal_columns:
        data = encode_nominal(data, nominal_columns)

    print (data)

    # Initialize clusters, each containing a single data point
    clusters = [np.array([point]) for _, point in data.iterrows()]

    # Initialize a list to store information about merges
    merge_history = []
    
    # Main loop
    while len(clusters) > nclusters:
        min_distance = float('inf')
        merge_indices = None

        # Find the closest pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = calculate_distance(clusters[i], clusters[j], link)
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        merged_cluster = np.concatenate((clusters[merge_indices[0]], clusters[merge_indices[1]]))
        merge_history.append((merge_indices[0], merge_indices[1], min_distance))
        del clusters[merge_indices[1]]
        clusters[merge_indices[0]] = merged_cluster

    # Store end time
    end_time = time.time()

    # Print execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return clusters, merge_history

# Function to check if a given input is numerical
def is_numerical(input_str):
    try:
        int(input_str)
        return True
    except ValueError:
        return False

# Parameters declaration
nclusters = 1    #Final number of clusters
link = 1         #Linkage method to calculate distances

# Define a dictionary to map numeric values to linkage methods
linkage_mapping = {1: 'single', 2: 'complete', 3: 'average', 4: 'ward'}

# Interface
print("-----------------------------------------------")
print("Alglomerative Hierarchical Clustering Algorithm")
print("-----------------------------------------------")
print("PARAMETERS INTERFACE ")
while True:
    nclusters= int(input("Enter desired number of clusters: "))
    if (is_numerical(nclusters)):
        break
    else:
        print("Error: It has to be a number")

while True:
    print("1-single, 2-complete, 3-average, 4-ward")
    link = int(input("Enter desired linkage method: "))
    if (is_numerical(nclusters) and (link==1 or link==2 or link==3 or link==4)):
        break
    else:
        print("Error: It has to be 1, 2, 3 or 4")

# Usage
print("------------------ALGORITHM -------------------")
file_path = "C:/Users/arnau/OneDrive/Escriptori/WUT/EDAMI/Project/adult.data"

# Define column names
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
           'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Read CSV file, excluding rows with missing values
data = pd.read_csv(file_path, header=None, names=columns, skipinitialspace=True).dropna()

# Identify nominal columns based on data type (object, usually strings)
nominal_columns = [column for column in data.columns if data[column].dtype == 'O']

# Main function call 
result, merge_history  = agglomerative_hierarchical_clustering(data, linkage_mapping[link], nominal_columns, nclusters)

# Print Merge History
print("Merge History:")
for i, merge in enumerate(merge_history):
    if len(merge) == 2:
        print(f"Iteration {i + 1}: Merge clusters {merge[0]} and {merge[1]}")
    elif len(merge) == 3:
        print(f"Iteration {i + 1}: Merge clusters {merge[0]} and {merge[1]} with distance {merge[2]}")
    else:
        print(f"Iteration {i + 1}: Unexpected merge format")


# Plot dendrogram
def plot_dendrogram(merge_history):
    dendrogram_info = np.array([[i, j, distance, 2] for i, j, distance in merge_history])
    linkage_matrix = linkage(dendrogram_info[:, :3])
    dendrogram(linkage_matrix)
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()

# Plot dendrogram
plot_dendrogram(merge_history)