# EDAMI-AHC
Agglomerative hierarchical clustering for clustering objects described by nominal and numerical attributes


Input Data:
    CSV File:
      ● The input data is read from a CSV file, specifically from the file path defined in file_path. By default data from the dataset “adult.data”.
    Column Names:
      ● The column names are specified in the columns list. In default scenario, the columns include:
      ● 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
      ● 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week','native-country', 'income'
    Data Processing:
      ● The code utilizes Pandas (pd) for data processing. It reads the CSV file, drops any rows with missing values, and identifies nominal columns based on data types.
    User Input:
      ● The user is prompted to input two parameters:
      ● nclusters: The desired number of clusters.
      ● link: The desired linkage method represented by numeric values (1 for 'single', 2 for 'complete', 3 for 'average', 4 for 'ward').
Output Data:
    Encoded Data:
      ● Nominal attributes in the input data are encoded using simple label encoding via the encode_nominal function.
    Hierarchical Clusters:
      ● The main output is the result of hierarchical clustering, stored in the result variable. It represents a list of clusters, where each cluster is an array of data points.
    Merge History:
      ● The merge history is recorded in the merge_history variable. It contains information about the merging of clusters at each iteration, including the indices of clusters being merged and the distance between them.
    Dendrogram Visualization:
      ● The dendrogram plot is generated using Matplotlib based on the merge history. It visually represents the hierarchical structure of the clustering.
    Printed Output:
      ● The code prints information about the merge history, indicating which clusters were merged at each iteration.

Design and implementation issues
  The code follows a procedural design paradigm. Functions like agglomerative_hierarchical_clustering, calculate_distance, and others represent procedural steps in the algorithm.
  The code is organized into modular functions, such as encode_nominal,agglomerative_hierarchical_clustering, and others. Each function has a specific responsibility, promoting code readability and maintainability.
  
Functions in the code include:
  calculate_distance(cluster1, cluster2, link)
  This function calculates the distance between two clusters based on the chosen linkage method.
  ● Input:
    ● cluster1: Data points in the first cluster.
    ● cluster2: Data points in the second cluster.
    ● link: Chosen linkage method ('single', 'complete', 'average', 'ward').
  ● Output:
    ● Distance between the two clusters based on the specified linkage method.
    
  encode_nominal(data, nominal_columns)
  This function encodes nominal attributes in the input data using simple label encoding.
  ● Input:
    ● data: Pandas DataFrame containing the input data.
    ● nominal_columns: List of column names with nominal attributes.
  ● Output:
    ● A new DataFrame with nominal attributes encoded.
    
  agglomerative_hierarchical_clustering(data, link, nominal_columns, nclusters)
  This is the main function implementing agglomerative hierarchical clustering.
  ● Input:
    ● data: Pandas DataFrame containing the input data.
    ● link: Chosen linkage method ('single', 'complete', 'average', 'ward').
    ● nominal_columns: List of column names with nominal attributes.
    ● nclusters: Desired number of clusters.
  ● Output:
    ● A list of clusters, where each cluster is represented as an array of data points.
    ● The merge history, containing information about the merging of clusters at each iteration.
    
  is_numerical(input_str)
  This function checks if a given input is numerical.
  ● Input:
    ● input_str: Input value to be checked.
  ● Output:
    ● Boolean value indicating whether the input is numerical (True) or not (False).
    
  plot_dendrogram(merge_history)
  This function generates a dendrogram visualization based on the merge history.
  ● Input:
    ● merge_history: List containing information about the merging of clusters at each iteration.
  ● Output:
    ● Matplotlib dendrogram plot visualizing the hierarchical clustering.
    
  MainScript
  The main script handles user input, reads the data from a CSV file, initiates the clustering algorithm, and prints the erge history and the dendrogram plot.
    ● User prompts to input the desired number of clusters (nclusters) and the chosen linkage method (link).
    ● Reading the data from a CSV file.
    ● Calling the agglomerative_hierarchical_clustering function.
    ● Printing the merge history.
    ● Generating and displaying the dendrogram plot.
  
  The use of linkage_mapping to map numeric values to linkage methods enhances code readability and makes it more user-friendly. Users can input numeric values, and the code translates them to the corresponding linkage method.
  The code provides a console-based user interface for inputting parameters(nclusters and link). The use of user prompts ensures a straightforward and interactive user experience.
  The merge_history variable records information about the merging of clusters at each iteration. This provides insights into the hierarchical clustering process.
  The plot_dendrogram function leverages Matplotlib to visualize the hierarchical clustering results. This visual representation enhances the interpretability of the clustering structure.
