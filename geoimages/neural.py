from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')


class Convolutional():
    """
        Class for Convolutional models.

    """
    def __init__(self):
        """
           The constructor for Convolutional models class.

           Arguments:
            None.

           Returns:
            None.

           Tips:
            None.

        """
        pass

    def Conv1L(input_shape):
        """
        input_shape: The height, width and channels as a tuple.
            Note that this does not include the 'batch' as a dimension.
            If you have a batch like 'X_train',
            then you can provide the input_shape using
            X_train.shape[1:]
        """

        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
        X_input = Input(input_shape)

        # Zero-Padding: pads the border of X_input with zeroes
        X = ZeroPadding2D((2, 2))(X_input)

        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0')(X)
        X = BatchNormalization(axis=3, name='bn0')(X)
        X = Activation('relu')(X)

        # MAXPOOL
        X = MaxPooling2D((2, 2), name='max_pool')(X)

        # FLATTEN X (means convert it to a vector)
        X = Flatten()(X)

        # DROPOUT
        X = Dropout(0.30)(X)

        # FULLY CONNECTED
        X = Dense(6, activation='softmax', name='fc')(X)

        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs=X_input, outputs=X, name='Conv1L')

        return model


    def Conv2L(input_shape):
        """
        input_shape: The height, width and channels as a tuple.
            Note that this does not include the 'batch' as a dimension.
            If you have a batch like 'X_train',
            then you can provide the input_shape using
            X_train.shape[1:]
        """

        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
        X_input = Input(input_shape)

        # Zero-Padding: pads the border of X_input with zeroes
        X = ZeroPadding2D((2, 2))(X_input)

        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0')(X)
        X = BatchNormalization(axis=3, name='bn0')(X)
        X = Activation('relu')(X)

        # MAXPOOL
        X = MaxPooling2D((2, 2), name='max_pool0')(X)

        # DROPOUT
        X = Dropout(0.30)(X)

        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(64, (3, 3), strides=(1, 1), name='conv1')(X)
        X = BatchNormalization(axis=3, name='bn1')(X)
        X = Activation('relu')(X)

        # MAXPOOL
        X = MaxPooling2D((2, 2), name='max_pool1')(X)

        # DROPOUT
        X = Dropout(0.30)(X)

        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        X = Dense(6, activation='softmax', name='fc')(X)

        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs=X_input, outputs=X, name='Conv2L')

        return model


    def smallVGGNet(input_shape):
        """
        input_shape: The height, width and channels as a tuple.
            Note that this does not include the 'batch' as a dimension.
            If you have a batch like 'X_train',
            then you can provide the input_shape using
            X_train.shape[1:]
        """

        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
        X_input = Input(input_shape)

        # CONV => RELU => POOL
        X = Conv2D(32, (3, 3), padding="same", strides=(1, 1), name='conv0')(X_input)
        X = Activation("relu")(X)
        X = BatchNormalization(axis=3, name='bn0')(X)
        X = MaxPooling2D(pool_size=(3, 3), name='max_pool0')(X)
        X = Dropout(0.25)(X)

        # (CONV => RELU) * 2 => POOL
        X = Conv2D(64, (3, 3), padding="same", name='conv1')(X)
        X = Activation("relu")(X)
        X = BatchNormalization(axis=3, name='bn1')(X)
        X = Conv2D(64, (3, 3), padding="same", name='conv2')(X)
        X = Activation("relu")(X)
        X = BatchNormalization(axis=3, name='bn2')(X)
        X = MaxPooling2D(pool_size=(2, 2), name='max_pool1')(X)
        X = Dropout(0.25)(X)

        # (CONV => RELU) * 2 => POOL
        X = Conv2D(128, (3, 3), padding="same", name='conv3')(X)
        X = Activation("relu")(X)
        X = BatchNormalization(axis=3, name='bn3')(X)
        X = Conv2D(128, (3, 3), padding="same", name='conv4')(X)
        X = Activation("relu")(X)
        X = BatchNormalization(axis=3, name='bn4')(X)
        X = MaxPooling2D(pool_size=(2, 2), name='max_pool2')(X)
        X = Dropout(0.25)(X)

        # first (and only) set of FC => RELU layers
        X = Flatten()(X)
        X = Dense(1024, name='fc1')(X)
        X = Activation("relu")(X)
        X = BatchNormalization(name='bn5')(X)
        X = Dropout(0.5)(X)

        # softmax classifier
        X = Dense(6, name='fc2')(X)
        X = Activation("softmax")(X)

        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs=X_input, outputs=X, name='smallVGGNet')

        return model