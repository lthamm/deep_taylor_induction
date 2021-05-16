"""Parse the heatmap annotations, create Sample objects and store them
as pickle files"""

import sys
import os
import pathlib
import pandas as pd
import xml.etree.ElementTree as ET
import pickle

from tqdm import tqdm

import config
from ilp.structures.feature_type import FeatureType
from ilp.structures.feature import Feature
from ilp.structures.sample import Sample
from helpers.img import name_to_path


def parse_file(filepath, heatmap, orig_file, meta, show):
    """Parse a single image"""
    tree = ET.parse(filepath)
    root = tree.getroot()

    filename = root.find('filename').text
    features = []

    for obj in root.iter('object'):

        # type of the object
        obj_type = obj.find('name').text

        # Coordinates defining the object
        #   this is the bounding box for e.g. a nose
        coordinates = {
            'xmin': int(obj.find('bndbox/xmin').text),
            'xmax': int(obj.find('bndbox/xmax').text),
            'ymin': int(obj.find('bndbox/ymin').text),
            'ymax': int(obj.find('bndbox/ymax').text)
        }

        feature = Feature(coordinates, FeatureType[obj_type])
        features.append(feature)

    sample = Sample(features=features,
                    ground_truth=meta['ground_truth'].iloc[0].item(),
                    prediction=meta['prediction'].iloc[0].item(),
                    predicted_class=meta['predicted_class'].iloc[0].item(),
                    heatmap=heatmap,
                    orig_file=orig_file
                    )
    if show:
        sample.show()

    return sample


def parse(show=False):

    df_path = os.path.sep.join([config.ilp_path, 'pickles', 'df.p'])
    df = pd.read_pickle(df_path)
    print(f'[+] Got {len(df)} files')

    samples = []

    try:
        heatmaps = os.listdir(config.heatmap_path)
    except FileNotFoundError:
        print('[!] No heatmaps, generate them first.', file=sys.stderr)
        raise

    try:
        annotations = os.listdir(config.annotation_path)
        print(f'[+] Got {len(annotations)} annotations')
    except FileNotFoundError:
        print('[!] No annotations found. They need to be created manually'
              'and then stored as VOC xml files in the corresponding'
              ' directory.', file=sys.stderr)
        raise

    print('[*] Parsing annotations')
    for hmap in tqdm(heatmaps):

        # Construct full file paths
        hmap_path = os.path.sep.join([config.heatmap_path, hmap])
        orig_file = name_to_path(hmap, config.test)

        # Get the heatmap base filename
        hmap_base = pathlib.Path(hmap_path).stem

        # Add only if annotations for heatmap exists
        if hmap_base+'.xml' in annotations:

            # Get the according annotation
            annotation = os.path.sep.join([config.annotation_path,
                                           hmap_base+'.xml'])

            # filename is stored relative inside the dataframe
            p = pathlib.Path(orig_file)
            rel_orig_file = os.path.join(*p.parts[-2:])

            # Part of the df relevant for this annotation
            hmap_df = df[df['filename'] == rel_orig_file]

            samples.append(parse_file(filepath=annotation,
                                      heatmap=hmap_path,
                                      orig_file=orig_file,
                                      meta=hmap_df,
                                      show=show))

            output_path = os.path.sep.join([config.pickle_path, 'samples.p'])
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)
