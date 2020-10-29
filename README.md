# geoimages
Package developed for machine learning proofs of concept for a geotechnical imaging firm.

## Development and Use
### Create Conda environment:
> $ conda env create -f environment.yml 

### For active development:  
> $ conda activate geoimages  
> $ pip install -e . --no-deps

## Summary  
**Machine Learning Accelerates Identification of Priority Geological Images**
    
Two models were developed to aid and speed identification of distinguishing features and patterns in a set of geological imagery. Using an unsupervised machine learning approach and, separately, a vector-distance/ similarity approach, the models successfully identified target images such that targets were most similar to a single input candidate image. Both models yielded similar results, with some overlap in the returned target-image sets. Both the K-Means unsupervised machine learning approach and the vector-distance/ similarity approach had similar elapsed times for prediction. Fitting each model took less than a minute, with a training-set of approximately one-hundred thousand images. However, elapsed time for training the K-Means model was significantly higher than the vector-distance/ similarity approach and may present limitations as the training data scales. A/B testing with the customer is recommended to determine which model returns more valuable results. Addition recommendations include discussion with the customer focussed on the necessity and feasibility to label features within the images to enable development of a supervised machine learning approach that may yield more relevant or diagnostic returns.

Need: A geotechnical consulting company needed a tool to aid and speed identification of interesting features and patterns in their imagery data.

Requirements:    
* Identify target images that are similar to a single candidate image.
* Score target images such that targets are most similar to the candidate image.
* Return n-target images based on defined input.

Approach:

    Because the images lacked wholesale labels or labels for specific features within the images, an unsupervised machine learning approach was appropriate.
    In addition to an unsupervised machine learning apparoch, the requirements could also be met via vector-distance/ similarity methods. This approach was also tested.

Methodology:

Setup

        Create root directory {choose_a_name}
        Create sub-directory named notebooks
        Create sub-directory named datasets
        Create sub-directory named images
        Within images directory, create sub-directory dev

Extract, Transform, Load

        Unzip geological_similarity.zip in images directory using ETL Class unzip_images() method.
        Randomly sample images for development and testing purposes using ETL Class sample_images() method.
        Augment remaining training data by creating additional training examples via image rotation using ETL Class rotate_images() method.
        Generate image metadata for all images using ETL Class generate_image_metadata() method.
        Convert images to vectors for model training and store in m by n dimensional array using ETL Class images_to_vectors_df() method.

Optimize K-Means Model

        Optimize model hyperparameter to the training set by calculating estimator inertia for a range of cluster numbers using KMmeansModel Class model_inertias() method.
        Visualize inertias using the Plots Class elbow_plot() method.
        Set the optimal number of clusters as the default based on the Elbow plot.

Fit K-Means Model

        Fit the K-Means model using the training data and the KMeansModel Class fit() method.

Fit Vector-distance/ similarity Model

        Fit the vector-distance/ similarity model using the training data and the PairwiseModel Class fit() method.

Load Candidate Image

        Load candidate image using the ETL Class read_image() method.
        Visualize candidate image.
        Vectorize candidate image for prediction using the ETL Class flatten_image() method.

Predict Target Images with K-Means Model

        Identify similar target images to candiate image using the KMeansModel Class predict() method.
        Visualize target images using the Plots Class plot_matches() method.

Predict Target Images with Vector-distance/ similarity Model

        Identify similar target images to candidate image using the PairwiseModel Class predict() method.
        Visualize target images using the Plots Class plot_matches() method.
