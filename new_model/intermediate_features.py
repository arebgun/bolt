import sys
sys.path.insert(1,"..")
from myrandom import random
choice = random.choice

# from myrandom import nprandom as random
from numpy import logical_and as land, logical_not as lnot, ndarray
# from scipy.stats import norm
from planar import Vec2, Affine
from planar.line import LineSegment, Ray
from semantics.landmark import Landmark
from semantics.representation import SurfaceRepresentation

from numpy.testing import assert_allclose


class Feature(object):
    def __init__(self):
        pass

class FeatureSet(object):
    def __init__(self):
        pass


class DistanceFeature(Feature):
    def __init__(self):
        super(DistanceFeature, self).__init__()

    @staticmethod
    def measure(perspective, landmark, trajector):
        if isinstance(trajector, Landmark):
            return landmark.distance_to(trajector.representation)
        elif isinstance(trajector, Vec2):
            return landmark.distance_to_point(trajector)
        elif isinstance(trajector, ndarray) and trajector.shape[1]==2:
            return landmark.distance_to_points(trajector)
        else:
            raise TypeError("trajector must be a Landmark, Vec2, or Nx2 numpy array")

    def __hash__(self):
        return hash(self.__class__.__name__)

    def __cmp__(self, other):
        return cmp(self.__hash__(), other.__hash__())


class SurfaceFeature(Feature):
    def __init__(self):
        super(SurfaceFeature, self).__init__()

    @staticmethod
    def measure(perspective, landmark, trajector):
        return isinstance(landmark.representation,SurfaceRepresentation)

    def __hash__(self):
        return hash(self.__class__.__name__)

    def __cmp__(self, other):
        return cmp(self.__hash__(), other.__hash__())


class ContainmentFeature(Feature):
    def __init__(self):
        super(ContainmentFeature, self).__init__()

    @staticmethod
    def measure(perspective, landmark, trajector):
        if isinstance(trajector, Landmark):
            return trajector.representation.overlap_fraction( 
                        landmark.representation )
        elif isinstance(trajector, Vec2):
            return landmark.contains_point(trajector)
        elif isinstance(trajector, ndarray) and trajector.shape[1]==2:
            return landmark.contains_points(trajector)
        else:
            raise TypeError(
                "trajector must be a Landmark, Vec2, or Nx2 numpy array")


class OrientationFeature(Feature):

    def __init__(self, orientation):
        super(OrientationFeature, self).__init__()

    def measure(self, perspective, landmark, trajector):
        p_ray = self.get_perspective_ray(perspective, landmark)

        if isinstance(trajector, Landmark):
            projected = None
            if landmark.parent is not None:
                projected = landmark.parent.project_point(
                                               trajector.representation.middle)
            else:
                projected = trajector.representation.middle

            return p_ray.angle_to(projected-p_ray.anchor)

        elif isinstance(trajector, Vec2):

            return p_ray.angle_to(trajector-p_ray.anchor)

        elif isinstance(trajector, ndarray) and trajector.shape[1]==2:
            # projecteds = ori_ray.line.project_points(trajector)
            return p_ray.angle_to_points(trajector-p_ray.anchor)
        else:
            raise TypeError(
                "trajector must be a Landmark, Vec2, or Nx2 numpy array")

    @staticmethod
    def get_perspective_ray(perspective, landmark):
        top_primary_axes = landmark.get_top_parent().get_primary_axes()

        our_axis = None
        for axis in top_primary_axes:
            if axis.contains_point(perspective):
                our_axis = axis
        assert( our_axis != None )

        new_axis = our_axis.parallel(landmark.representation.middle)
        new_perspective = new_axis.project(perspective)

        p_ray=Ray.from_points([landmark.representation.middle,
                               new_perspective])

        return p_ray

featureList =[DistanceFeature,
              SurfaceFeature,
              ContainmentFeature,
              OrientationFeature]

featureDict = {'distance':   DistanceFeature,
               'surface':    SurfaceFeature,
               'contained':  ContainmentFeature,
               'angle':      OrientationFeature
               }
