import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters
        
        self.centers = None
        self.cluster_center_label = None

    #Helper function to kmeans
    def __compute_distance(self, data):
        """
        Compute the euclidean distance between each datapoint and each center.
    
        Arguments:    
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        N = data.shape[0]
        distances = np.zeros((N,self.K))
       
        for k in range(self.K):
            center = self.centers[k]
            distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))
        
        
        return distances
    
    def __find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.
    
        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
   
        cluster_assignments = np.argmin(distances, axis=1)
    
        return cluster_assignments

    def __compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments: 
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
    
        centers = np.zeros((self.K,data.shape[1]))
        for i in range(self.K):
            centers[i] = np.mean(data[cluster_assignments == i], axis=0)
    
        return centers


    def k_means(self, data, max_iter):
        """
        Main function that combines all the former functions together to build the K-means algorithm.
    
        Arguments: 
            data: array of shape (N, D) where N is the number of data samples, D is number of features.
            K: int, the number of clusters.
            max_iter: int, the maximum number of iterations
        Returns:
            centers: array of shape (K, D), the final cluster centers.
            cluster_assignments: array of shape (N,) final cluster assignment for each data point.
        """
        # Initialize the centers
        random_idx = np.random.permutation(data.shape[0])[:self.K]
        self.centers = data[random_idx[:self.K]]

        # Loop over the iterations
        for i in range(max_iter):
            old_centers = self.centers.copy()  # keep in memory the centers of the previous iteration

        
            
            self.centers = self.__compute_centers(data, self.__find_closest_cluster(self.__compute_distance(data)))

            # End of the algorithm if the centers have not moved (hint: use old_centers and look into np.all)
            if np.array_equal(old_centers, self.centers):  ### WRITE YOUR CODE HERE
                print(f"K-Means has converged after {i+1} iterations!")
                break
            
        cluster_assignments = self.__find_closest_cluster(self.__compute_distance(data))
        # Compute the final cluster assignments
        return self.centers, cluster_assignments

    
    

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        self.centers, cluster_assignments = self.k_means(training_data, self.max_iters)
        self.cluster_center_label = np.zeros((self.centers.shape[0]))
        for i in range(self.centers.shape[0]):
            self.cluster_center_label[i] = np.argmax(np.bincount(training_labels[cluster_assignments == i]))

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        distances = self.__compute_distance(test_data)
        cluster_assignments = self.__find_closest_cluster(distances)
    
        pred_labels = self.cluster_center_label[cluster_assignments]

        return pred_labels