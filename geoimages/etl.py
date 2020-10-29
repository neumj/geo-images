import os
import json
import numpy as np
import pandas as pd
import random
import shutil
from scipy import ndimage
import imageio
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


class Images():
    """
        Class for ETL of geological_similarity.zip image data.

    """
    def __init__(self, images_path='../images/geological_similarity',
                 data_path='../datasets', dev_path='../images/dev',
                 sub_dirs={'andesite': {'label': 1}, 'gneiss': {'label': 2}, 'marble': {'label': 3}, 'quartzite': {'label': 4},
                           'rhyolite': {'label': 5}, 'schist': {'label':6}}):
        """
           The constructor for etl class.

           Arguments:
            images_path -- string, path to root directory containing images.
            data_path -- string, path to save output to.
            dev_path -- string, path to dev directory.
            sub_dirs -- dictionary, of directories holding images at root images directory and class labels.

           Returns:
            None.

           Tips:
            None.

        """
        self.images_path = images_path
        self.data_path = data_path
        self.dev_path = dev_path
        self.sub_dirs = sub_dirs


    def unzip_images(self, file_path):
        """
           Unzip .zip file.

           Arguments:
            file_path -- string, path to file.

           Returns:
            None.

           Tips:
            None.

        """
        with ZipFile(file_path, 'r') as zipObj:
            zipObj.extractall('../images')


    def rename_images(self):
        """
           Rename images to include containing directory name, which may be an appropriate label.

           Arguments:
            None.

           Returns:
            None.

           Tips:
            - Uses paths defined in class initiation.

        """
        for sd in self.sub_dirs.keys():
            files = os.listdir(self.images_path + os.sep + sd)
            for f in files:
                source_file = self.images_path + os.sep + sd + os.sep + f
                renamed_file = self.images_path + os.sep + sd + os.sep + sd + '.' + f
                shutil.move(source_file, renamed_file)


    def sample_images(self, sample_percent=0.1):
        """
           Randomly sample from each image directory and move images for training/ development purposes.

           Arguments:
            sample_percent -- float, percentage of images to sample.

           Returns:
            None.

           Tips:
            - Uses paths defined in class initiation.

        """
        for sd in self.sub_dirs.keys():
            files = os.listdir(self.images_path + os.sep + sd)
            sample = random.sample(files, int(np.floor(len(files) * sample_percent)))
            for f in sample:
                source_file = self.images_path + os.sep + sd + os.sep + f
                dev_file = self.dev_path + os.sep + f
                shutil.move(source_file, dev_file)


    def rotate_images(self):
        """
           Rotate and output images in image root directory.

           Arguments:
            None.

           Returns:
            None.

           Tips:
            - Uses paths defined in class initiation.
            - Can be used to augment training data set.

        """
        rots = [90, 180, 270]
        for sd in self.sub_dirs.keys():
            files = os.listdir(self.images_path + os.sep + sd)
            for f in files:
                source_file = self.images_path + os.sep + sd + os.sep + f
                im = self.read_image(source_file)
                for r in rots:
                    rot_file = self.images_path + os.sep + sd + os.sep \
                               + f.split('.')[0] + '.' + f.split('.')[1] + '.' + str(r) + '.jpg'
                    im_rot = ndimage.rotate(im, r)
                    imageio.imwrite(rot_file, im_rot)


    def read_json(self, file_path):
        """
           Read and json file.

           Arguments:
            file_path -- string, path to file.

           Returns:
            d -- dictionary, with json contents.

           Tips:
            None.

        """
        with open(file_path) as json_data:
            d = json.load(json_data)

        return d


    def write_json(self, data_dict, file_path):
        """
           Write dictionary to json.

           Arguments:
            data_dict -- dictionary.
            file_path -- string, path to file.

           Returns:
            None.

           Tips:
            None.

        """
        with open(file_path, 'w') as fp:
            json.dump(data_dict, fp, indent=4)


    def generate_image_metadata(self, exp_size=(28, 28, 3)):
        """
           Generate metadata for images in root image directory.

           Arguments:
            exp_size -- tuple, expected image height, width, channels.

           Returns:
            images_meta_data -- dictionary, of image metadata.

           Tips:
            None.

        """
        self.images_meta_data = {'root': self.images_path,
                                 'dirs': [],
                                 'images': {},
                                 'errata': []}
        for (root, dirs, files) in os.walk(self.images_path, topdown=False):
            if len(files) > 1:
                for f in files:
                    im = imageio.imread(root + os.sep + f)
                    fsegs = f.split('.')
                    if len(fsegs) == 3:
                        img_key = f.split('.')[0] + '.' + f.split('.')[1]
                    else:
                        img_key = f.split('.')[0] + '.' + f.split('.')[1] + '.' + f.split('.')[2]
                    self.images_meta_data['images'].update({img_key: {'file_name': f, 'root': root,
                                                                      'file_path': root + os.sep + f,
                                                                      'size': im.shape}})

            else:
                self.images_meta_data['dirs'].append(dirs)

        return self.images_meta_data


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


    def flatten_image(self, image_array, normalize=1):
        """
           Flatten an image array.

           Arguments:
            image_array -- array, > 2-dimensional array.
            normalize -- scalar, 0 or 1.  Normalize by 255.

           Returns:
            flat -- array, 1-dimensional array.

           Tips:
            None.

        """
        flat = image_array.flatten()
        if normalize == 1:
            flat = flat / 255.

        return flat


    def images_to_vectors_df(self, image_size=[28 ,28 ,3], to_grey_scale=0):
        """
           From image metadata, read images, flatten, and load into an array for training.

           Arguments:
            image_size -- list, input image height, width, channels.
            to_grey_scale -- scalar, 0 or 1.  Convert image to grey scale.

           Returns:
            vectors_df -- Pandas dataframe, of image vectors for model training or other analysis.

           Tips:
            None.

        """
        df_idx = []
        if to_grey_scale == 0:
            im_array = np.zeros((len(self.images_meta_data['images']) ,(image_size[0] * image_size[1] * image_size[2])))
        elif to_grey_scale == 1:
            im_array = np.zeros((len(self.images_meta_data['images']) ,(image_size[0] * image_size[1] * 1)))

        im_count = 0
        for ky in self.images_meta_data['images'].keys():
            im = self.read_image(self.images_meta_data['images'][ky]['file_path'], to_grey_scale=to_grey_scale)
            im = self.flatten_image(im)
            im_array[im_count ,:] = im
            df_idx.append(ky)
            im_count += 1

        self.vectors_df = pd.DataFrame(data=im_array, index=df_idx)

        return self.vectors_df


    def images_to_x_y(self, image_size=[28 ,28 ,3]):
        """
           From image metadata, read images, flatten, and load into an array for training.

           Arguments:
            image_size -- list, input image height, width, channels.
            to_grey_scale -- scalar, 0 or 1.  Convert image to grey scale.

           Returns:
            vectors_df -- Pandas dataframe, of image vectors for model training or other analysis.

           Tips:
            None.

        """
        df_idx = []
        x_array = np.zeros((len(self.images_meta_data['images']), image_size[0], image_size[1], image_size[2]))

        y_array = []

        im_count = 0
        for ky in self.images_meta_data['images'].keys():
            im = self.read_image(self.images_meta_data['images'][ky]['file_path'])
            x_array[im_count, :, :, :] = im
            y_array.append(ky.split('.')[0])
            df_idx.append(ky)
            im_count += 1

        self.X = x_array
        self.Y = np.array(y_array)

        return self.X, self.Y


class Labels():
    """
        Class for ETL to generate labels.

    """
    def __init__(self):
        """
           The constructor for Labels class.

           Arguments:
            None.

           Returns:
            None.

           Tips:
            None.

        """
        pass


    def categorical_to_onehot(self, Y_Labels):
        encoder = LabelEncoder()
        encoder.fit(Y_Labels)
        classes = encoder.classes_
        Y_Enc = encoder.transform(Y_Labels)
        Y_OH = np_utils.to_categorical(Y_Enc)

        return Y_OH, classes