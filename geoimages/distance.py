from sklearn.metrics import pairwise_distances
import pandas as pd


class PairwiseModel():
    """
        Class for Pairwise distance based model.

    """
    def __init__(self):
        """
            The constructor for distance_model class.

           Arguments:
            None.

           Returns:
            None.

           Tips:
            None.

        """
        pass


    def fit(self, image_vectors_df):
        """
           Seed data for Pairwise distance calculations.

           Arguments:
            image_vectors_df -- Pandas dataframe containing vector representations of training images.

           Returns:
            None.

           Tips:
            - Input image_vectors_df can be generated from images_to_vectors_df() method in etl module.

        """
        self.reference_vectors = image_vectors_df


    def predict(self, candidate_vector, k_candidates=12, dist_metric='correlation'):
        """
           Calculate pairwise distance from reference_vectors to candidate_vector.

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
        distances = pairwise_distances(self.reference_vectors.values, Y=vector, metric=dist_metric)
        dist_df = pd.DataFrame(data={dist_metric: distances[: ,0]}, index=self.reference_vectors.index)
        dist_df.sort_values(by=[dist_metric], ascending=True, inplace=True)

        return dist_df[0:k_candidates]