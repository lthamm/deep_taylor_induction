import argparse

import numpy as np
from sklearn.metrics import classification_report

import config
from models.vgg import VGGFinetune
from trainers.vgg_trainer import VGGTrainer
from data_loaders.picasso_loader import PicassoLoader


parser = argparse.ArgumentParser(description='Finetune a pretrained model to \
                                            classify the picasso dataset')

parser.add_argument('--epochs', default=1, type=int,
                    help='Number of epochs to train the model for finetuning')

parser.add_argument('--base_layers', default=0, type=int,
                    help='Number of layers of the pretrained network to unfreeze')



if __name__ == "__main__":

    args = parser.parse_args()

    # Tensorflow flow_from_directory for the picasso dataset with generators
    training, validation, test, info = PicassoLoader().get_data_flows()

    # Warm up" the newly created head of the network
    print('[*] Training: Warming up head')


    vgg = VGGFinetune()
    trainer = VGGTrainer(vgg=vgg,
                        config=config,
                        validation=validation,
                        training=training,
                        info=info)

    # Start the actual finetuning
    histroy = trainer.train(epochs=args.epochs, n_base_layers=args.base_layers)
