#############################################################################
# Copyright (c) 2010 by Casey Duncan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name(s) of the copyright holders nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#############################################################################

from __future__ import division

from numpy import (array, zeros, argmin, arange, hypot, abs as npabs,
                   logical_and as land, logical_not as lnot)
import planar
from planar import Vec2
from planar.util import cached_property


class BoundingBox(object):
    """An axis-aligned immutable rectangular shape described
    by two points that define the minimum and maximum
    corners.

    :param points: Iterable containing one or more :class:`~planar.Vec2` 
        objects.
    """

    def __init__(self, points):
        self._init_min_max(points)
    
    def _init_min_max(self, points):
        # points = iter(points)
        # try:
        #     min_x, min_y = max_x, max_y = points.next()
        # except StopIteration:
        #     raise ValueError, "BoundingBox() requires at least one point"
        # for x, y in points:
        #     if x < min_x:
        #         min_x = x * 1.0
        #     elif x > max_x:
        #         max_x = x * 1.0
        #     if y < min_y:
        #         min_y = y * 1.0
        #     elif y > max_y:
        #         max_y = y * 1.0

        #NOTE: numpy way should be faster on large polygons
        xs,ys = array(points).T
        min_x = xs.min()*1.0
        min_y = ys.min()*1.0
        max_x = xs.max()*1.0
        max_y = ys.max()*1.0
        self._min = planar.Vec2(min_x, min_y)
        self._max = planar.Vec2(max_x, max_y)
        self._edge_segments = None
    
    @property
    def bounding_box(self):
        """The bounding box for this shape. For a BoundingBox instance,
        this is always itself.
        """
        return self
    
    @property
    def min_point(self):
        """The minimum corner point for the shape. This is the corner
        with the smallest x and y value.
        """
        return self._min
    
    @property
    def max_point(self):
        """The maximum corner point for the shape. This is the corner
        with the largest x and y value.
        """
        return self._max

    @property
    def width(self):
        """The width of the box."""
        return self._max.x - self._min.x
    
    @property
    def height(self):
        """The height of the box."""
        return self._max.y - self._min.y

    @property
    def area(self):
        return self.width*self.height

    @property
    def edge_segments(self):
        """The edges of the bounding box as LineSegments"""
        if self._edge_segments is None:
            self._edge_segments = []
            verts = [self._min, (self._min.x, self._max.y), 
                     self._max, (self._max.x, self._min.y), self._min]
            for i in range(4):
                self._edge_segments.append( 
                    planar.LineSegment(verts[i],verts[i+1]-verts[i]))
        return self._edge_segments
    
    @cached_property
    def center(self):
        """The center point of the box."""
        return (self._min + self._max) / 2.0
    
    @cached_property
    def is_empty(self):
        """True if the box has zero area."""
        width, height = self._max - self._min
        return not width or not height

    @classmethod
    def from_points(cls, points):
        """Create a bounding box that encloses all of the specified points.
        """
        box = object.__new__(cls)
        box._init_min_max(points)
        return box

    @classmethod
    def from_shapes(cls, shapes):
        """Creating a bounding box that completely encloses all of the
        shapes provided.
        """
        shapes = iter(shapes)
        try:
            shape = shapes.next()
        except StopIteration:
            raise ValueError, (
                "BoundingBox.from_shapes(): requires at least one shape")
        min_x, min_y = shape.bounding_box.min_point
        max_x, max_y = shape.bounding_box.max_point

        for shape in shapes:
            x, y = shape.bounding_box.min_point
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            x, y = shape.bounding_box.max_point
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        box = object.__new__(cls)
        box._min = planar.Vec2(min_x, min_y)
        box._max = planar.Vec2(max_x, max_y)
        return box
    
    @classmethod
    def from_center(cls, center, width, height):
        """Create a bounding box centered at a particular point.

        :param center: Center point
        :type center: :class:`~planar.Vec2`
        :param width: Box width.
        :type width: float
        :param height: Box height.
        :type height: float
        """
        cx, cy = center
        half_w = width * 0.5
        half_h = height * 0.5
        return cls.from_points([
            (cx - half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            ])
    
    def inflate(self, amount):
        """Return a new box resized from this one. The new
        box has its size changed by the specified amount,
        but remains centered on the same point.

        :param amount: The quantity to add to the width and
            height of the box. A scalar value changes
            both the width and height equally. A vector
            will change the width and height independently.
            Negative values reduce the size accordingly.
        :type amount: float or :class:`~planar.Vec2`
        """
        try:
            dx, dy = amount
        except (TypeError, ValueError):
            dx = dy = amount * 1.0
        dv = planar.Vec2(dx, dy) / 2.0
        return self.from_points((self._min - dv, self._max + dv))
    
    def contains_point(self, point):
        """Return True if the box contains the specified point.

        :param other: A point vector
        :type other: :class:`~planar.Vec2`
        :rtype: bool
        """
        x, y = point
        return (self._min.x <= x < self._max.x 
            and self._min.y < y <= self._max.y)

    def contains_points(self, points):
        """Like contains_point but takes a list or array of points."""
        xs, ys = array(points).T
        return land( land(self._min.x <= xs, xs < self._max.x),
                     land(self._min.y < ys, ys <= self._max.y) )

    def distance_to(self, point):
        """Return the distance between the given point and this box."""
        x, y = point
        lt_min_x = x < self._min.x
        le_max_x = x <= self._max.x
        lt_min_y = y < self._min.y
        le_max_y = y <= self._max.y
        if lt_min_x:
            if lt_min_y:
                return self._min.distance_to(point)
            elif le_max_y:
                return self._min.x - point.x
            else:
                return Vec2(self._min.x,self._max.y).distance_to(point)
        elif le_max_x:
            if lt_min_y:
                return self._min.y - point.y
            elif le_max_y:
                return 0
            else:
                return point.y - self._max.y
        else:
            if lt_min_y:
                return Vec2(self._max.x,self._min.y).distance_to(point)
            elif le_max_y:
                return point.x - self._max.x
            else:
                return self._max.distance_to(point)

    def distance_to_points(self, points):
        xs, ys = array(points).T
        xds = array([self._min.x - xs, xs - self._max.x])
        yds = array([self._min.y - ys, ys - self._max.y])
        r = arange(len(xs))
        xds = xds[argmin(npabs(xds),axis=0),r]
        yds = yds[argmin(npabs(yds),axis=0),r]
        xds[xds<0] = 0
        yds[yds<0] = 0
        return hypot(xds,yds)

    def signed_distance_to_points(self, points):
        """Returns negative distance if point is inside"""
        #TODO implement signed_distance_to_points?
        raise NotImplementedError



    def project(self,point):
        #TODO implement box specific version
        return self.to_polygon().project(point)

    def project_points(self,points):
        #TODO implement box specific version
        return self.to_polygon().project_points(points)

    def _distance_to_line_ray_or_segment(self, lros):
        min_dist = float('inf')
        for edge in self.edge_segments():
            dist = lros.distance_to_segment(edge)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def distance_to_line(self, line):
        return self._distance_to_line_ray_or_segment(line)

    def distance_to_ray(self, ray):
        return self._distance_to_line_ray_or_segment(ray)

    def distance_to_segment(self, segment):
        return self._distance_to_line_ray_or_segment(segment)

    def distance_to_box(self, box):
        #TODO: optimize distance_to_box?
        xdelta = ydelta = 0
        if self._min.x > box._max.x:
            xdelta = self._min.x - box._max.x
        elif self._max.x < box._min.x:
            xdelta = box._min.x - self._max.x
        if self._min.y > box._max.y:
            ydelta = self._min.y - box._max.y
        elif self._max.y < box._min.y:
            ydelta = box._min.y - self._max.y
        return hypot(xdelta,ydelta)

    def distance_to_polygon(self, poly):
        return poly.distance_to_box(self)
    
    def fit(self, shape):
        """Create a new shape by translating and scaling shape so that
        it fits in this bounding box. The shape is scaled evenly so that
        it retains the same aspect ratio.

        :param shape: A transformable shape with a bounding box.
        """
        if isinstance(shape, BoundingBox):
            scale = min(self.width / shape.width, self.height / shape.height)
            return shape.from_center(
                self.center, shape.width * scale, shape.height * scale)
        else:
            shape_bbox = shape.bounding_box
            offset = planar.Affine.translation(self.center - shape_bbox.center)
            scale = planar.Affine.scale(min(self.width / shape_bbox.width,
                self.height / shape_bbox.height))
            return shape * (offset * scale)

    def overlap_with_point(self, point):
        return 0

    def overlap_with_line(self, line):
        return 0

    def overlap_with_box(self, box):
        xdelta = min(self._max.x, box._max.x) - max(self._min.x, box._min.x)
        ydelta = min(self._max.y, box._max.y) - max(self._min.y, box._min.y)
        if xdelta <= 0 or ydelta <= 0:
            return 0
        else:
            return xdelta * ydelta

    def overlap_with_circle(self, circle):
        raise NotImplementedError

    def overlap_with_polygon(self, poly):
        raise NotImplementedError

    def to_polygon(self):
        """Return a rectangular :class:`~planar.Polygon` object with the same
        vertices as the bounding box.

        :rtype: :class:`~planar.Polygon`
        """
        return planar.Polygon([
            self._min, (self._min.x, self._max.y), 
            self._max, (self._max.x, self._min.y)],
            is_convex=True)

    def __eq__(self, other):
        return (self.__class__ is other.__class__
            and self.min_point == other.min_point
            and self.max_point == other.max_point)

    def __ne__(self, other):
        return not self.__eq__(other)

    def almost_equals(self, other):
        """Return True if this bounding box is approximately equal to another
        box, within precision limits.
        """
        return (self.__class__ is other.__class__
            and self.min_point.almost_equals(other.min_point)
            and self.max_point.almost_equals(other.max_point))

    def __repr__(self):
        """Precise string representation."""
        return "BoundingBox([(%r, %r), (%r, %r)])" % (
            self.min_point.x, self.min_point.y, 
            self.max_point.x, self.max_point.y)

    __str__ = __repr__

    def __mul__(self, other):
        try:
            rectilinear = other.is_rectilinear
        except AttributeError:
            return NotImplemented
        if rectilinear:
            return self.from_points(
                [self._min * other, self._max * other])
        else:
            p = self.to_polygon()
            p *= other
            return p

    __rmul__ = __mul__


# vim: ai ts=4 sts=4 et sw=4 tw=78

