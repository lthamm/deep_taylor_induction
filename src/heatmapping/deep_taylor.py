"""Create a deep taylor analyser with the iNNvestigate toolkit"""

import os

import innvestigate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import helpers.img


class DeepTaylor():

    def __init__(self, model, inputs):
        self.analyzer = self.create_analyzer(model)
        self.outputs = None
        self.inputs = inputs

    def create_analyzer(self, model):
        analyzer = innvestigate.create_analyzer("deep_taylor", model)
        return analyzer

    def analyze(self, steps):
        """Create the heatmap with a given model"""

        analysis = np.empty(shape=(0, 224, 224))
        print('[*] Heatmapping with deep taylor')

        for i in tqdm(range(steps)):
            batch = self.inputs.next()
            # index 0 to get image
            tmp_analysis = self.analyzer.analyze(batch[0])
            outputs = self.postprocess_outputs(tmp_analysis)
            analysis = np.append(analysis, outputs, axis=0)

        self.outputs = analysis

        return self.outputs

    def postprocess_outputs(self, outputs):
        """Transform outputs"""
        transformed = []
        for i, a in tqdm(enumerate(outputs)):
            a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
            a /= np.max(np.abs(a))
            transformed.append(a)

        return np.array(transformed)

    def save_images(self, path, cmap='seismic', dpi=271):

        if self.outputs is None:
            print('[!] No images to save')
        else:
            print('[*] Saving heatmaps')
            os.makedirs(path, exist_ok=True)
            self.inputs.reset()
            filenames = self.inputs.filenames
            print(f'Filenames: {len(filenames)}')
            print(f'Outputs: {len(self.outputs)}')
            files = len(filenames)

            # the analyzer uses generator batches to create the outputs
            #   e.g. with 100 files and a step size of 32, we will need
            #   4 steps, but this will generate 128 outputs, so we limit
            #   the output loop to the number of input files we have
            #   in order to avoid duplicate outputs and an IndexError
            for index, img in enumerate(tqdm(self.outputs[:files])):

                filename = helpers.img.path_to_name(filenames[index])
                savepath = os.path.sep.join([path, filename])
                fig = plt.figure(None,
                                 figsize=(2.24+0.3, 2.24+0.3),
                                 dpi=100,
                                 tight_layout=True,
                                 frameon=False)
                plt.axis('off')
                plt.imshow(img, clim=(-1, 1), cmap=cmap)
                plt.savefig(savepath, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close()
