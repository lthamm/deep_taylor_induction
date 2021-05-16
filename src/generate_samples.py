"""Parse the manual annotations, create python object and dump it as a pickle"""

import argparse

import ilp.parse_annotations

parser = argparse.ArgumentParser(description='Parse annotations and create pickles')
parser.add_argument('--show', default=False, type=bool,
                    help=('Plot each sample after it has been created. '
                          'Requires user to close the plot in order to proceed.'
                          'Can be used to verify that everything has been '
                          'created correctly.'))


if __name__ == "__main__":
    print('[*] Generating samples.')
    args = parser.parse_args()
    ilp.parse_annotations.parse(args.show)
    print('[+] Samples created. You can now generate aleph.')
