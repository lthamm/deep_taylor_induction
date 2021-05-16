""" Predicates for automatic relation extraction
Terminology:
    example: A Sample (one image)
    organ: Object class, e.g. nose
    part: Organ in a specific example

"""


class Predicate():

    @staticmethod
    def calc():
        raise NotImplementedError

    @staticmethod
    def get():
        """Get the calculcation as an aleph string"""
        raise NotImplementedError

    @staticmethod
    def mode():
        """Returns the aleph mode defintion of the relation"""
        raise NotImplementedError

    @staticmethod
    def determination():
        """Return the aleph determination of the relation"""
        raise NotImplementedError

# Binary Predicates

class Contains(Predicate):

    type = 'binary'
    name = 'contains'

    def calc(feature_a, feature_b, identifier):
        """See if a contains b or b contains a

        Parameters
        ----------
        identifier : string
            Filename of the identifier containing the sample

        """

        if feature_a.polygon.contains(feature_b.polygon):
            return f'contains({identifier+str(feature_a.kind)}, {identifier+str(feature_b.kind)}).'

    def mode():
        return f':- modeb(*, {Contains.name}(+part, +part)).'

    def determination():
        return f':- determination(face/1, {Contains.name}/2).'


class Overlaps(Predicate):

    type = 'binary'
    name = 'overlaps'

    def calc(feature_a, feature_b, identifier):
        """Calculate if a overlaps b or b overlaps a"""

        if feature_a.polygon.overlaps(feature_b.polygon):
            return f'overlaps({identifier+str(feature_a.kind)}, {identifier+str(feature_b.kind)}).'

    def mode():
        return f':- modeb(*, {Overlaps.name}(+part, +part)).'

    def determination():
        return f':- determination(face/1, {Overlaps.name}/2).'


class Intersects(Predicate):

    type = 'binary'
    name = 'intersects'

    def calc(feature_a, feature_b, identifier):
        """Calculate if a intersects b"""

        if feature_a.polygon.intersects(feature_b.polygon):
            return f'intersects({identifier+str(feature_a.kind)}, {identifier+str(feature_b.kind)}).'

    def mode():
        return f':- modeb(*, {Intersects.name}(+part, +part)).'

    def determination():
        return f':- determination(face/1, {Intersects.name}/2).'


class Disjoint(Predicate):

    type = 'binary'
    name = 'disjoint'

    def calc(feature_a, feature_b, identifier):
        """Calculate if a overlaps b or b contains a"""

        if feature_a.polygon.disjoint(feature_b.polygon):
            return f'disjoint({identifier+str(feature_a.kind)}, {identifier+str(feature_b.kind)}).'

    def mode():
        return f':- modeb(*, {Disjoint.name}(+part, +part)).'

    def determination():
        return f':- determination(face/1, {Disjoint.name}/2).'


class LeftOf(Predicate):

    name = 'left_of'
    type = 'binary'
    tolerance = 5  # offest tolreance in px

    @staticmethod
    def left_of(p1, p2):
        if p1.x < p2.x - LeftOf.tolerance:
            return True
        else:
            return False

    def calc(feature_a, feature_b, identifier):
        """Calculate the centroid Point for each feature polygon and check
        if it is positioned left of the other one"""

        # TODO: check if we should pass this relation for the outlines of a face
        # beacuse it might not be to interesting
        # but it also can be interesting as well, because the face outlines
        # are a reference system (where from the center of the face)
        # is the object located

        if LeftOf.left_of(feature_a.polygon.centroid,
                          feature_b.polygon.centroid):
            return f'left_of({identifier+str(feature_a.kind)}, {identifier+str(feature_b.kind)}).'

    def mode():
        return f':- modeb(*, {LeftOf.name}(+part, +part)).'

    def determination():
        return f':- determination(face/1, {LeftOf.name}/2).'


class TopOf(Predicate):

    name = 'top_of'
    type = 'binary'
    tolerance = 5  # offest tolreance in px

    @staticmethod
    def top_of(p1, p2):
        if p1.y < p2.y - TopOf.tolerance:
            return True
        else:
            return False

    def calc(feature_a, feature_b, identifier):
        """Calculate the centroid Point for each feature polygon and check
        if it is positioned left of the other one"""

        # TODO: check if we should pass this relation for the outlines of a face
        # beacuse it might not be to interesting
        # but it also can be interesting as well, because the face outlines
        # are a reference system (where from the center of the face)
        # is the object located

        if TopOf.top_of(feature_a.polygon.centroid,
                         feature_b.polygon.centroid):
            return f'{TopOf.name}({identifier+str(feature_a.kind)}, {identifier+str(feature_b.kind)}).'

    def mode():
        return f':- modeb(*, {TopOf.name}(+part, +part)).'

    def determination():
        return f':- determination(face/1, {TopOf.name}/2).'


# Unariy Predicates
class IsA(Predicate):
    """Classify the part in an example as some organ"""

    type = 'meta'
    name = 'is_a'

    def calc(identifier, feature):
        return f'{IsA.name}({identifier+str(feature.kind)}, {feature.kind}).'

    def mode():
        return f':- modeb(*, {IsA.name}(+part, #organ)).'

    def determination():
        return f':- determination(face/1, {IsA.name}/2).'


class HasA(Predicate):
    """State that an example has some part"""

    type = 'meta'
    name = 'has_a'

    def calc(identifier, feature):
        return f'{HasA.name}({identifier}, {identifier+str(feature.kind)}).'

    def mode():
        return f':- modeb(*, {HasA.name}(+example, -part)).'

    def determination():
        return f':- determination(face/1, {HasA.name}/2).'


class Face(Predicate):

    type = 'unariy'
    name = 'face'

    def calc(identifier):
        return f'face({identifier}).'

    def mode():
        return ':- modeh(1, face(+example)).'

    def determination():
        return None
