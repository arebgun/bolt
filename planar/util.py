#############################################################################
#
# Copyright (c) 2010 by Casey Duncan and contributors
# All Rights Reserved.
#
# This software is subject to the provisions of the BSD License
# A copy of the license should accompany this distribution.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#
#############################################################################


from numpy import array, dot, arccos, where, minimum
import math

# Define assert_unorderable() depending on the language 
# implicit ordering rules. This keeps things consistent
# across major Python versions
try:
    3 > ""
except TypeError: # pragma: no cover
    # No implicit ordering (newer Python)
    def assert_unorderable(a, b):
        """Assert that a and b are unorderable"""
        return NotImplemented
else: # pragma: no cover
    # Implicit ordering by default (older Python)
    # We must raise an exception ourselves
    # To prevent nonsensical ordering
    def assert_unorderable(a, b):
        """Assert that a and b are unorderable"""
        raise TypeError("unorderable types: %s and %s"
            % (type(a).__name__, type(b).__name__))

def cached_property(func):
    """Special property decorator that caches the computed 
    property value in the object's instance dict the first 
    time it is accessed.
    """

    def getter(self, name=func.func_name):
        try:
            return self.__dict__[name]
        except KeyError:
            self.__dict__[name] = value = func(self)
            return value
    
    getter.func_name = func.func_name
    return property(getter, doc=func.func_doc)

def cos_sin_deg(deg):
    """Return the cosine and sin for the given angle
    in degrees, with special-case handling of multiples
    of 90 for perfect right angles
    """
    deg = deg % 360.0
    if deg == 90.0:
        return 0.0, 1.0
    elif deg == 180.0:
        return -1.0, 0
    elif deg == 270.0:
        return 0, -1.0
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)


# vim: ai ts=4 sts=4 et sw=4 tw=78

def signed_angle_diff(x,y): 
    return (array(x) - array(y) + 180)%360 - 180

def counterclockwise(A,B,C):
    ccw = (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
    return 1 if ccw > 0.0 else -1 if ccw < 0.0 else 0

def intersects(A,B,C,D):
    return counterclockwise(A,C,D) != counterclockwise(B,C,D) and \
           counterclockwise(A,B,C) != counterclockwise(A,B,D)

def segments_intersect(seg1,seg2):
    A,B = seg1.points
    C,D = seg2.points
    return intersects(A,B,C,D)


def norm(x,axis):
    return sum(abs(x)**2,axis=axis)**0.5

def angle_between(a,b):
    """Return the angle in radians between two vectors of equal
    cardinality. 
    """
    a = array(a)
    if len(a.shape) == 1: a = a.reshape((1,len(a)))
    b = array(b).T
    if len(b.shape) == 1: b = b.reshape((len(b),1))
    arccosInput = dot(a,b)[0]/norm(a,axis=-1)/norm(b,axis=-2)
    arccosInput[where(arccosInput > 1.0)] = 1.0
    arccosInput[where(arccosInput < -1.0)] = -1.0
    return arccos(arccosInput)