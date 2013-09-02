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

class ContainmentFeature(Feature):
	def __init__(self):
		super(ContainmentFeature, self).__init__()

	@staticmethod
	def measure(perspective, landmark, trajector):
		if isinstance(trajector, Landmark):
			return float(landmark.representation.contains( 
													trajector.representation ))
		elif isinstance(trajector, Vec2):
			return float(landmark.contains_point(trajector))
		elif isinstance(trajector, ndarray) and trajector.shape[1]==2:
			return landmark.contains_points(trajector)
		else:
			raise TypeError(
				"trajector must be a Landmark, Vec2, or Nx2 numpy array")


class OrientationFeature(Feature):

	def __init__(self, orientation):
		super(OrientationFeature, self).__init__()
		self.orientation = orientation

	def measure(self, perspective, landmark, trajector):
		ori_ray = self.get_orientation_ray(perspective, landmark)

		if isinstance(trajector, Landmark):
			projected = None
			if landmark.parent is not None:
				projected = landmark.parent.project_point(
											   trajector.representation.middle)
			else:
				projected = trajector.representation.middle
			projected = ori_ray.line.project(projected)
			return ori_ray.contains_point(projected) and not \
				   landmark.representation.contains_point(projected)

		elif isinstance(trajector, Vec2):
			projected = ori_ray.line.project(trajector)
			return ori_ray.contains_point(projected) and not \
				   landmark.representation.contains_point(projected)

		elif isinstance(trajector, ndarray) and trajector.shape[1]==2:
			projecteds = ori_ray.line.project_points(trajector)
			return land(ori_ray.contains_points(projecteds),
				   lnot(landmark.representation.contains_point(projected)))
		else:
			raise TypeError(
				"trajector must be a Landmark, Vec2, or Nx2 numpy array")

	def get_orientation_ray(self, perspective, landmark):

		standard_direction = Vec2(0,1)

		top_primary_axes = landmark.get_top_parent().get_primary_axes()

		our_axis = None
		for axis in top_primary_axes:
			if axis.contains_point(perspective):
				our_axis = axis
		assert( our_axis != None )

		new_axis = our_axis.parallel(landmark.representation.middle)
		new_perspective = new_axis.project(perspective)

		p_segment = LineSegment.from_points( [new_perspective, landmark.representation.middle] )

		angle = standard_direction.angle_to(p_segment.vector)
		rotation = Affine.rotation(angle)
		o = [self.orientation]
		rotation.itransform(o)
		direction = o[0]
		ori_ray = Ray(p_segment.end, direction)

		return ori_ray

featureList =[DistanceFeature(), 
			  ContainmentFeature(),
			  OrientationFeature(Vec2(0,-1)), # in front
			  OrientationFeature(Vec2(0,1)),  # behind
			  OrientationFeature(Vec2(-1,0)), # left
			  OrientationFeature(Vec2(1,0))]  # right
