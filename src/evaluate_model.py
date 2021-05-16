"""Evaluate the finetuned VGG16 model on the picasso dataset"""

import numpy as np
from sklearn.metrics import classification_report

from data_loaders.picasso_loader import PicassoLoader
import config


def eval_model(model, test, steps):
    """Generate a sklearn classification report with the keras model
    
    Parameters
    ----------
    model : keras.Model
         Picasso Dataset pretrained keras model
    
    test : generator
        Generator for test data

    steps : int
        Number of steps to predict on the generator
    """
    
    test.reset()
    pred_idxs = model.predict_generator(test, steps=steps)
    pred_idxs = np.round(pred_idxs, 0)
    test.reset()
    report = classification_report(test.classes, pred_idxs)
    return report


if __name__ == "__main__":
    from models.vgg import VGGFinetune
    vgg = VGGFinetune().model
    training, validation, test, info = PicassoLoader().get_data_flows()
    steps = (info['n_test'] // config.batch_size) + 1
    report = eval_model(vgg, test, steps)
    print(report)
