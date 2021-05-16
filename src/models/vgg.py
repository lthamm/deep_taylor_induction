""" Pretrained VGG model intialized with weights learned from the imagenet
    dataset. Prepared for fine tuning via removal of the last set of fully
    connected layers (head / top / output) and freezing of the previous layers
    to keep the information previously learned
"""

import keras

class VGGFinetune():

    def __init__(self, force_new=False):

        self.model = self.__create_model()

        if not force_new:
            self.__load_weights()

        self.compile()

    def __load_weights(self):
        """Load the weights for finetuning
            Keras 2.2.4, which has to be used for this project because of
            dependencies, has a bug when loading models with a predefined input
            layer, the model needs to be built every time so only the weights
            can be loaded, which is working fine.
        """
        try:
            self.model.load_weights('output/models/vgg.model')
            print('[+] Succesfully loaded weights')
        except Exception as e:
            print(e)
            print('[!] No weights loaded, train a new model')


    def __create_model(self):
        """Construct the vgg16 model with new trainable head"""

        # Initialize VGG16 without the fully connected output layers
        #   which is done via include_top=False
        base_model = keras.applications.VGG16(weights='imagenet',
                        include_top=False,
                        input_tensor=keras.Input(shape=(224, 224, 3))
                                              )

        # Freeze the base model
        base_model.trainable = False

        # Create a new sequential model
        model = keras.models.Sequential()

        # Add the base model layers one by one, so the model
        #   is not added as a model object
        for layer in base_model.layers:
            model.add(layer)

        # New output layers for finetuning
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        return model

    def compile(self, learning_rate=0.001):
        print("[*] Compiling model")
        self.model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                           loss="binary_crossentropy",
                           metrics=["binary_accuracy"])

    def info(self):
        for l in self.model.layers:
            print(l.name, l.trainable)

    def set_trainable(self, n_base_layers):
        """Freeze or unfreeze the the vgg convolutional layers
        
        Parameters
        ----------
        n_nase_layers : int
            Number of base layers to train
        """

        NEW_LAYERS = 3   # number of newly created layers ontop of the vgg16

        # The last vgg16 layer is a maxpooling layer, so for one more
        # trainable layer, we need to choose the convolutional layer before
        stop = n_base_layers + 1 + NEW_LAYERS

        # Set all layers except the last n_base_layers to be non trainable
        for layer in self.model.layers[:-stop]:
            layer.trainable = False

        # Unfreeze the n_trainable last layers
        for layer in self.model.layers[-stop:]:
            layer.trainable = True

        # Set a lower learning rate if training the old body also
        learning_rate = 0.0001 if n_base_layers != 0  else 0.001
        print(f'[*] Set learning rate to {learning_rate}')
        self.compile(learning_rate)

        print(self.model.summary())
        print(f'[*] Set {n_base_layers} vgg16 base layers to be trainable')
