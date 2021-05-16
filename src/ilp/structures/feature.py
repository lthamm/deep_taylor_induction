from shapely.geometry import Polygon


class Feature():

    def __init__(self, coordinates, kind):
        """Create new feature

        Parameters
        ----------
        coordinates : dict
            Dict containing coordinates of the feature in the parent
            image with keys xmin, xmax, ymin, ymax

        kind : FeatureType
            The objects type, e.g. face, nose
        """

        self.coordinates = coordinates
        self.kind = kind
        self.polygon = Feature.construct_polygon(coordinates)

    def construct_polygon(coordinates):
        """Construct a shapely polygon given coordinates"""
        # Shapely expects ordered sequence of (x, y[, z]) point tuples
        #   so x and y need to be in one tuple, our input has x and y
        #   separated

        p1 = (coordinates['xmin'], coordinates['ymin'])
        p2 = (coordinates['xmin'], coordinates['ymax'])
        p3 = (coordinates['xmax'], coordinates['ymax'])
        p4 = (coordinates['xmax'], coordinates['ymin'])

        return Polygon([p1, p2, p3, p4, p1]) # start is end for poylgon # TODO: correct?
