""" Write the Aleph background knowledge files

Aleph needs 3 files:
    - filestem.b    -   Background Knowledge
    - filestem.f    -   Positive Examples
    - filestem.n    -   Negative Examples



Reference:
    https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html
"""
import os
import pickle

from tqdm import tqdm

import config
from ilp.structures import predicates


def writeln(file, text):
    if text:
        file.write(text + '\n')


if __name__ == "__main__":

    print("[*] Generating aleph files")

    aleph_path = os.path.sep.join([config.ilp_path, 'aleph'])
    os.makedirs(aleph_path, exist_ok=True)
    bg_path = os.path.sep.join([aleph_path, 'picasso.b'])
    pos_path = os.path.sep.join([aleph_path, 'picasso.f'])
    neg_path = os.path.sep.join([aleph_path, 'picasso.n'])

    with open(bg_path, 'w') as bg_knowledge,\
         open(pos_path, 'w') as pos_examples,\
         open(neg_path, 'w') as neg_examples:

        # Init list with indexes matching the class of the precition
        files = [neg_examples, pos_examples]

        # List of all available predicates
        preds = [predicates.Face,
                 predicates.HasA,
                 predicates.IsA,
                 predicates.Contains,
                 predicates.Intersects,
                 predicates.Disjoint,
                 predicates.Overlaps,
                 predicates.LeftOf,
                 predicates.TopOf
                 ]

        # All predicates with two arguments
        preds_binary = list(filter(lambda x: x.type == 'binary', preds))

        # write setting to background knowledge file
        writeln(bg_knowledge, config.aleph_settings[0])

        # write background knowledge mode
        [writeln(bg_knowledge, pred.mode()) for pred in preds]

        # write background knowledge determinations
        [writeln(bg_knowledge, pred.determination()) for pred in preds]

        # write setting to background knowledge file
        [writeln(bg_knowledge, ln) for ln in config.aleph_settings[1:]]

        # Load the pickled samples containing sample objects
        samples_path = os.path.sep.join([config.pickle_path, 'samples.p'])
        with open(samples_path, 'rb') as f:

            samples = pickle.load(f)

            print('[*] Generating postive and negative examples')
            for sample in tqdm(samples):
                # pick example file based on the precited class
                #   0: negative prediction for negative file
                #   1: positive prediction for positive file
                class_file = files[sample.predicted_class]

                # register example as pos or negative
                identifier = os.path.basename(sample.heatmap)
                identifier = os.path.splitext(identifier)[0]
                writeln(class_file, predicates.Face.calc(identifier))
 
                # Background Knowledge
                for feature in sample.features:
                    ########
                    # Has_a:
                    ########
                    # write the features the image contains
                    #   e.g. has_a(pos_01, pos_01nose)
                    writeln(bg_knowledge, predicates.HasA.calc(identifier,
                                                       feature))
                    
                    ########
                    # Is_a:
                    ########
                    # each feature of each image describe with a has_a predicate
                    #   now gets a type with is_a predicate
                    #   e.g. has_a(pos_01, pos_01nose) will get a 
                    #       is_a(pos_01nose, nose)
                    writeln(bg_knowledge, predicates.IsA.calc(identifier, feature))

                # Calculate all binary predicates between all features of
                #   a given example
                for pred in preds_binary:
                    for feature_a in sample.features:
                        for feature_b in sample.features:

                            # don't calculate relations with self
                            if feature_a == feature_b:
                                continue

                            pred_str = pred.calc(feature_a,
                                                 feature_b,
                                                 identifier
                                                 )

                            writeln(bg_knowledge, pred_str)

    print(f"Succesfully stored files in {aleph_path}")
