from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pandas as pd


class KMeansModel():
    """
        Class for K-Means based model.

    """
    def __init__(self):
        """
           The constructor for kmeans_model class.

           Arguments:
            None.

           Returns:
            None.

           Tips:
            None.

        """
        pass


    def fit(self, image_vectors_df, num_clusters=3):
        """
           Fit the K-Means model, creates an estimator and index of cluster labels.

           Arguments:
            image_vectors_df -- Pandas dataframe containing vector representations of training images.
            num_clusters -- scalar, number of clusters for K-Means.

           Returns:
            None.

           Tips:
            - Input image_vectors_df can be generated from images_to_vectors_df() method in etl module.
            - num_clusters default has been determined via inertial modeling.

        """
        self.reference_vectors = image_vectors_df
        self.estimator = KMeans(n_clusters=num_clusters, random_state=0).fit(self.reference_vectors)
        self.cluster_idx = self.estimator.labels_


    def predict(self, candidate_vector, k_candidates=12, dist_metric='correlation'):
        """
           Predict cluster membership for candidate_vector and calculate pairwise distances to same-cluster neighbors.

           Arguments:
            candidate_vector -- numpy array from flattened candidate image.
            k_candidates -- scalar, number of nearest neighbors to return.
            dist_metric -- string, any valid metric from scipy.spatial.distance.pdist.

           Returns:
            dist_df -- Pandas dataframe with K-nearest neighbors.

           Tips:
            - Input candidate_vector can be generated from read_image() and flatten_image() methods in etl module.
            - Default dist_metric value set based on small batch testing.

        """
        vector = candidate_vector.reshape((1, candidate_vector.shape[0]))
        self.prediction = self.estimator.predict(vector)
        ref_vectors_subset = self.reference_vectors[self.cluster_idx == self.prediction]
        distances = pairwise_distances(ref_vectors_subset.values, Y=vector, metric=dist_metric)
        dist_df = pd.DataFrame(data={dist_metric: distances[: ,0]}, index=ref_vectors_subset.index)
        dist_df.sort_values(by=[dist_metric], ascending=True, inplace=True)

        return dist_df[0:k_candidates]


    def model_inertias(self, clusters):
        """
           Generate estimator inertia for a set of cluster values to determine the optimal number of clusters
           based on the training data.

           Arguments:
            clusters -- list, of cluster values.

           Returns:
            inertias -- list, of estimator inertias.

           Tips:
            - clusters, inertias can be passed to elbow_plot() method in visualize module.

        """
        inertias = []
        for k in clusters:
            self.fit(self.reference_vectors, num_clusters=k)
            inertias.append(self.estimator.inertia_)

        return inertias