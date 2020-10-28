import imageio
import matplotlib.pyplot as plt
import os
import math

class Plots():
    """
        Class for generating visualizations.

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


    def read_image(self, file_path, to_grey_scale=0):
        """
           Read and image file.

           Arguments:
            file_path -- string, path to file.
            to_grey_scale -- scalar, 0 or 1.  Convert image to grey scale.

           Returns:
            im -- imageio.core.util.Array

           Tips:
            None.

        """
        im = imageio.imread(file_path)
        if to_grey_scale == 1:
            im = im.mean(axis=2)

        return im


    def plot_matches(self, matches_df, meta_data):
        """
           Plot images returned from kmeans_model.predict or distance_model.predict.

           Arguments:
            matches_df -- Pandas dataframe, K-nearest matches from model.predict.
            meta_data -- dictionary, metadata for training images.

           Returns:
            None.

           Tips:
            - Input meta_data can be generated from generate_image_metadata() method in etl module.
            - Presupposes %matplotlib inline

        """
        match_paths = []
        for ky in matches_df.index:
            match_paths.append(meta_data['images'][ky]['file_path'])

        k = len(matches_df.index)
        nrows = math.ceil(k / 3)
        figsize=(15, 15)
        fig,a =  plt.subplots(nrows,3,figsize=(int(k * 1.5), int(k * 1.5)))
        fig.tight_layout(h_pad=2)
        im_count = 0
        for i in range(nrows):
            for j in range(3):
                disp = self.read_image(match_paths[im_count], to_grey_scale=0)
                a[i][j].imshow(disp)
                a[i][j].set_title(match_paths[im_count].split(os.sep)[-2] + ':' + match_paths[im_count].split(os.sep)[-1])
                im_count += 1
                if im_count == k:
                    break


    def elbow_plot(self, clusters, inertias):
        """
           Generate an Elbow Plot of estimator inertias to optimize K-Means cluster number.

           Arguments:
            clusters -- list, of cluster values.
            inertias -- list, of estimator inertias.

           Returns:
            None.

           Tips:
            - Inputs can be generated from model_inertias() method in cluster module.
            - Presupposes %matplotlib inline

        """
        plt.plot(clusters, inertias, 'ro-')
        plt.ylabel('Inertia')
        plt.xlabel('num_clusters')
        plt.title('K-Means Elbow Plot')