import os
from datetime import datetime

from keras.callbacks import ModelCheckpoint, TensorBoard


class VGGTrainer():

    def __init__(self, vgg, training, validation, info, config):
        """ Inialize a trainer for finetuning of the vgg model """

        self.vgg = vgg
        self.training = training
        self.validation = validation
        self.config = config
        self.info = info
        self.callbacks = []
        self.histories = []
        self.init_callbacks()

    def init_callbacks(self):

        # Callback to save the model at a specific frequency
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_path, 'vgg_cp.ckpt'),
                monitor='val_loss',
                mode='auto',
                save_best_only=False,
                save_weights_only=False,
                verbose=0,
            )
        )

        # Add TensorBoard callback
        log_dir = os.path.sep.join(['logs', 'fit'])
        log_dir = log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callbacks.append(
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
                )
        )

    def train(self, epochs, n_base_layers=0):
        """ Train the model on the dataset given with the constructor

        Parameters
        -----------
        n_base_layers : int
            Number of base model layers to unfreeze.
        epochs : int
             Number of epochs to train the model for.
        """

        self.vgg.set_trainable(n_base_layers)

        print('[*] Started training')

        history = self.vgg.model.fit_generator(
            self.training,
            validation_data = self.validation,
            epochs=epochs,
            steps_per_epoch= self.info['n_train'] // self.config.batch_size,
            validation_steps = self.info['n_validation'] // self.config.batch_size,
            callbacks=self.callbacks,
        )

        self.histories.append(history)

        # Save the new model
        self.vgg.model.save(self.config.vgg_path)

        # Reset the generators, so they can be used for the next training iteration
        self.training.reset()
        self.validation.reset()

        return history
