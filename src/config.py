import os

# Root path of the project
base_path = '.'

############################################################
# Keras settings (train)
############################################################

# Batch size
batch_size = 32
# Annotation for the class labels
class_labels = ['ACTUAL FACE', 'MIXED UP FACE']

############################################################
# Output paths
############################################################
checkpoint_path = 'checkpoints'
output_path = 'output'
ilp_path = os.path.sep.join([base_path, output_path, 'ilp'])
ilp_img_path = os.path.sep.join([ilp_path, 'images'])
heatmap_path = os.path.sep.join([ilp_path, 'deep_taylor_images'])
annotation_path = os.path.sep.join([ilp_path, 'annotations'])
pickle_path = os.path.sep.join([ilp_path, 'pickles'])

# set the path to the serialized model after training
vgg_path = os.path.sep.join(['output', 'models', 'vgg.model'])

# data path
train = os.path.sep.join([base_path, 'datasets', 'train'])
test = os.path.sep.join([base_path, 'datasets', 'test'])
val = os.path.sep.join([base_path, 'datasets', 'val'])

# training data path for valiating the generated hypothesis
predicate_picasso = os.path.sep.join([base_path, 'datasets', 'predicate_picasso'])

############################################################
# ILP Knowedlge Generation
############################################################

# the number of samples per ilp class
n_ilp_samples = 50

# percentage of images that should be correctly classified
# as negative in evaluate_hypothesis
validation_threshold = 95

############################################################
# Aleph settings
############################################################
""" Settings for the aleph interpreter that are written to
the background knowledge files in generate_aleph

Documentation:
    https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html#SEC45
"""

aleph_settings = [':- use_module(library(lists)).',
                  # Upper bound of layers for new variables
                  ':- set(i, 5).',
                  # Maximum length for the precondition
                  ':- set(clauselength, 30).',
                  # Minimum number of examples covered by an accepted clause
                  ':- set(minpos, 2).',
                  # Lower bound for score of an accepted clause
                  '%:- set(minscore, 0).',
                  '%:- set(verbosity, 0).',
                  ':- set(noise, 0).',
                  ':- set(nodes, 50000).'
                  ]
