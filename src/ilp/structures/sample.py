import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ilp.structures.feature_type import FeatureType


class Sample():

    def __init__(self,
                 features,
                 ground_truth,
                 prediction,
                 predicted_class,
                 heatmap,
                 orig_file):

        self.features = features
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.predicted_class = predicted_class
        self.heatmap = heatmap
        self.orig_file = orig_file
        self.color_map = {
                FeatureType.left_eye: '#264653',
                FeatureType.right_eye: '#2a9d8f',
                FeatureType.nose: '#e9c46a',
                FeatureType.mouth: '#e76f51',
                FeatureType.face_frame: '#f4a261'
        }

    def show(self):
        # Create figure and axes
        fig, ax = plt.subplots(1)

        heatmap_img = mpimg.imread(self.heatmap)
        orig_img = mpimg.imread(self.orig_file)

        # Display the image
        ax.imshow(orig_img)
        ax.imshow(heatmap_img, alpha=0.8)

        # Show the features
        for feature in self.features:
            x, y = feature.polygon.exterior.xy
            ax.plot(x, y, color=self.color_map[feature.kind])

            t_x = feature.coordinates['xmin']
            t_y = feature.coordinates['ymin'] - 5

            ax.text(t_x, t_y,
                    feature.kind.name,
                    color=self.color_map[feature.kind],
                    fontsize=8)

        plt.show()
