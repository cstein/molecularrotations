#!/usr/bin/env python
import math
import numpy

class QuaternionException(BaseException):
  pass

class QuaternionIsNotUnitException(QuaternionException):
  pass

class Quaternion(object):
  def __init__(self,w,x,y,z):
    self._w = w
    self._v = numpy.array([x,y,z])

  def W(self):
    return self._w

  def X(self):
    return self._v[0]

  def Y(self):
    return self._v[1]

  def Z(self):
    return self._v[2]

  def V(self):
    return self._v

  def as4Vector(self):
    return numpy.array([self.W(),self.X(),self.Y(),self.Z()])

  @classmethod
  def fromWAndV(cls,w,v):
    if not isinstance(v,numpy.ndarray):
      raise TypeError("argument v must be of %s but user supplied %s" % (type(numpy.array([1])),type(v)))
    return cls(w,v[0],v[1],v[2])

  @classmethod
  def fromAngleAndVector(cls, input_angle, input_vector):
    angle = input_angle*0.5
    vector = cls.fromWAndV(0.0, input_vector)
    vector.Normalize()
    v = numpy.sin(angle)*vector.V()
    w = numpy.cos(angle)
    return cls.fromWAndV(w,v)

  @classmethod
  def from4Vector(cls,v):
    if not isinstance(v, numpy.ndarray):
      raise TypeError("argument v must be of %s but user supplied %s" % (type(numpy.array([1])),type(v)))
    return cls(v[0], v[1], v[2], v[3])

  def SquaredNorm(self):
    return self._w*self._w + numpy.dot(self._v,self._v)

  def Norm(self):
    return math.sqrt(self.SquaredNorm())

  def Normalize(self):
    magnitude = self.Norm()
    self._w /= magnitude
    self._v /= magnitude

  def isUnit(self):
    threshold = 1.0e-6
    return abs(self.Norm() - 1.0) < threshold

  def getConjugate(self):
    return Quaternion(self._w, -self._v[0], -self._v[1], -self._v[2])

  def getInverse(self):
    return self.getConjugate() * (1.0/self.SquaredNorm())

  def rotateVector(self,vector_to_rotate):
    if not self.isUnit():
      raise QuaternionIsNotUnitException
    if not isinstance(vector_to_rotate,numpy.ndarray):
      raise TypeError
    q_vector_to_rotate = Quaternion.fromWAndV(0.0,vector_to_rotate)
    rhs = q_vector_to_rotate * self.getConjugate()
    rotated_quaternion = self*rhs
    return rotated_quaternion.V()

  def as4Matrix(self):
    # return 4x4 real matrix representation
    return numpy.array([[ self._w, self._v[0], self._v[1], self._v[2]],
                        [-self._v[0], self._w, -self._v[2], self._v[1]],
                        [-self._v[1], self._v[2], self._w, -self._v[0]],
                        [-self._v[2], -self._v[1], self._v[0], self._w]])

  ### override __xxx__ to imitate behavior of real math
  def __eq__(self,other):
    threshold = 1.0e-6  # accuracy of decimals
    w_eq = abs(self._w - other.W()) < threshold
    x_eq = abs(self._v[0] - other.X()) < threshold
    y_eq = abs(self._v[1] - other.Y()) < threshold
    z_eq = abs(self._v[2] - other.Z()) < threshold
    return w_eq and x_eq and y_eq and z_eq

  def __ne__(self,other):
    return not self == other

  def __mul__(self,other):
    if isinstance(other, Quaternion):
      w = self._w*other.W() - numpy.dot(self._v,other.V())
      v = self._w*other.V() + other.W()*self._v + numpy.cross(self._v, other.V())
      return Quaternion(w,v[0],v[1],v[2])
    elif isinstance(other,int) or isinstance(other,float) or isinstance(other,numpy.float64):
      w = self._w * other
      v = self._v * other
      return Quaternion(w,v[0],v[1],v[2])
    else:
      raise QuaternionException("Operation now allowed")

  def __add__(self,other):
    if isinstance(other,Quaternion):
      return Quaternion.fromWAndV(self._w+other.W(), self._v + other.V())
    else:
      raise QuaternionException("Operation now allowed")

  def __repr__(self):
    return "%s(%f,%f,%f,%f)" % (self.__class__.__name__,self._w,self._v[0],self._v[1],self._v[2])

  def __str__(self):
    return "%6.2f%6.2f%6.2f%6.2f" % (self._w, self._v[0], self._v[1], self._v[2])

import unittest
class TestQuaternion(unittest.TestCase):
  def setUp(self):
    self.q0 = Quaternion(1.0,0.0,0.0,0.0)
    self.q1 = Quaternion(1.0,1.0,1.0,1.0)
    self.qn = Quaternion(0.5,2.0,-1.3,4.0)

  def test_BasicAccessorsCreationQ0(self):
    self.assertEqual(self.q0.W(), 1.0)
    self.assertEqual(self.q0.X(), 0.0)
    self.assertEqual(self.q0.Y(), 0.0)
    self.assertEqual(self.q0.Z(), 0.0)
    self.assertEqual(repr(self.q0),"Quaternion(1.000000,0.000000,0.000000,0.000000)")
    self.assertEqual(str(self.q0),"  1.00  0.00  0.00  0.00")

  def test_BasicAccesorsCreationQ0V(self):
    test_vector = numpy.array([0.0,0.0,0.0])
    the_test = (test_vector == self.q0.V()).all()
    self.assertTrue(the_test)

  def test_BasicAccessorsCreationQ1(self):
    self.assertEqual(self.q1.W(), 1.0)
    self.assertEqual(self.q1.X(), 1.0)
    self.assertEqual(self.q1.Y(), 1.0)
    self.assertEqual(self.q1.Z(), 1.0)
    self.assertEqual(repr(self.q1),"Quaternion(1.000000,1.000000,1.000000,1.000000)")
    self.assertEqual(str(self.q1),"  1.00  1.00  1.00  1.00")

  def test_BasicAccesorsCreationQ1V(self):
    test_vector = numpy.array([1.0,1.0,1.0])
    the_test = (test_vector == self.q1.V()).all()
    self.assertTrue(the_test)

  def test_CreateQuaternionFromWandVWrongArgument(self):
    self.assertRaises(TypeError, Quaternion.fromWAndV, 1.0, 1)
    self.assertRaises(TypeError, Quaternion.fromWAndV, 1.0, 1.0)
    self.assertRaises(TypeError, Quaternion.fromWAndV, 1.0, True)
    self.assertRaises(TypeError, Quaternion.fromWAndV, 1.0, {})

  def test_CreateQuaternionFromWandV(self):
    test_vector = numpy.array([1.0,1.2,-2.3])
    test_w = 2.0
    test_q = Quaternion(test_w,test_vector[0],test_vector[1],test_vector[2])
    q = Quaternion.fromWAndV(test_w, test_vector)
    self.assertEqual(q.W(), test_w)
    self.assertTrue((q.V() == test_vector).all())
    self.assertEqual(q,test_q)

  def test_CreateRotationQuarternionFromAngleAndVector(self):
    test_vector = numpy.array([2.0,0.0,0.0])
    test_angle = numpy.pi
    test_quaternion = Quaternion(0.0,1.0,0.0,0.0)
    q = Quaternion.fromAngleAndVector(test_angle, test_vector)
    self.assertEqual(q, test_quaternion)

  def test_CreateRotationQuarternionFromAngleAndVector2(self):
    test_vector = numpy.array([2.0,2.0,0.0])
    test_angle = numpy.pi
    test_quaternion = Quaternion(0.0,numpy.sqrt(0.5),numpy.sqrt(0.5),0.0)
    q = Quaternion.fromAngleAndVector(test_angle, test_vector)
    self.assertEqual(q,test_quaternion)
    self.assertEqual(test_quaternion,q)

  def test_CreateRotationQuarternionFromAngleAndVector3(self):
    test_vector = numpy.array([0.0,1.0,1.0])
    test_angle = numpy.pi
    test_quaternion = Quaternion(0.0,0.0,numpy.sqrt(0.5),numpy.sqrt(0.5))
    q = Quaternion.fromAngleAndVector(test_angle, test_vector)
    self.assertEqual(q,test_quaternion)
    self.assertEqual(test_quaternion,q)

  def test_CreateRotationQuarternionFromAngleAndVector4(self):
    test_vector = numpy.ones(3)
    test_angle = 2.0*numpy.pi/3.0
    test_quaternion = Quaternion(0.5,0.5,0.5,0.5)
    q = Quaternion.fromAngleAndVector(test_angle, test_vector)
    self.assertEqual(q,test_quaternion)

  def test_CreateQuaternionFrom4Vector(self):
    v = numpy.arange(0,4, dtype=float)
    q = Quaternion.from4Vector(v)
    self.assertTrue(numpy.allclose(v, q.as4Vector()))

  def test_QuaternionNorm(self):
    self.assertEqual(self.q0.Norm(), 1.0)
    self.assertEqual(self.q1.Norm(), 2.0)

    q = Quaternion(1.0,1.0,0.0,0.0)
    self.assertEqual(q.Norm(),numpy.sqrt(2.0))

  def test_QuaternionNormalizeQ0(self):
    self.q0.Normalize()
    self.assertEqual(self.q0.W(), 1.0)
    self.assertEqual(self.q0.X(), 0.0)
    self.assertEqual(self.q0.Y(), 0.0)
    self.assertEqual(self.q0.Z(), 0.0)
    self.assertEqual(self.q0.Norm(),1.0)

  def test_QuaternionNormalizeQ1(self):
    self.q1.Normalize()
    self.assertEqual(self.q1.W(), 0.5)
    self.assertEqual(self.q1.X(), 0.5)
    self.assertEqual(self.q1.Y(), 0.5)
    self.assertEqual(self.q1.Z(), 0.5)
    self.assertEqual(self.q1.Norm(),1.0)

  def test_QuaternionGetConjugate(self):
    q2 = self.q1.getConjugate()
    self.assertEqual(q2.W(),  self.q1.W())
    self.assertEqual(q2.Y(), -self.q1.Y())
    self.assertEqual(q2.X(), -self.q1.X())
    self.assertEqual(q2.Z(), -self.q1.Z())

  def test_QuaternionGetInverse(self):
    q2 = self.q1.getConjugate()
    norm2 = self.q1.SquaredNorm()
    qi = self.q1.getInverse()
    self.assertEqual(q2.W()/norm2, qi.W())
    self.assertEqual(q2.X()/norm2, qi.X())
    self.assertEqual(q2.Y()/norm2, qi.Y())
    self.assertEqual(q2.Z()/norm2, qi.Z())

  def test_QuationIsUnit(self):
    self.assertTrue(self.q0.isUnit())
    self.assertFalse(self.q1.isUnit())
    self.assertFalse(self.qn.isUnit())
    self.q1.Normalize()
    self.qn.Normalize()
    self.assertTrue(self.q1.isUnit())
    self.assertTrue(self.qn.isUnit())

  def test_QuaternionsEqual(self):
    q1_test = Quaternion(1.0,1.0,1.0,1.0)
    self.assertTrue(self.q1 == q1_test)
    self.assertTrue(q1_test == self.q1)
    self.assertFalse(self.q1 == self.q0)
    self.assertFalse(self.q0 == self.q1)

  def test_QuaternionsNotEqual(self):
    q1_test = Quaternion(1.0,1.0,1.0,1.0)
    self.assertFalse(self.q1 != q1_test)
    self.assertFalse(q1_test != self.q1)
    self.assertTrue(self.q0 != q1_test)
    self.assertTrue(q1_test != self.q0)

  def test_QuaternionAs4Matrix(self):
    qmat = numpy.array([[0.5,2.0,-1.3,4.0],
                        [-2.0,0.5,-4.0,-1.3],
                        [ 1.3,4.0,0.5,-2.0 ],
                        [-4.0,1.3,2.0,0.5]])
    self.assertTrue(numpy.allclose(self.qn.as4Matrix(),qmat))

  def test_QuaternionMultiplyRaisesException(self):
    #self.assertRaises(QuaternionException, self.q1.__mul__, 1)
    #self.assertRaises(QuaternionException, self.q1.__mul__, 1.0)
    self.assertRaises(QuaternionException, self.q1.__mul__, [])
    self.assertRaises(QuaternionException, self.q1.__mul__, ())
    self.assertRaises(QuaternionException, self.q1.__mul__, {})
    self.assertRaises(QuaternionException, self.q1.__mul__, "")

  def test_QuaternionMultiplyLHS(self):
    q2 = Quaternion(2.0,-1.0,0.0,1.0)
    q_answer = Quaternion(2.0,2.0,0.0,4.0)
    self.assertEqual(self.q1*q2,q_answer)

  def test_QuaternionMultiplyRHS(self):
    q2 = Quaternion(2.0,-1.0,0.0,1.0)
    q_answer = Quaternion(2.0,0.0,4.0,2.0)
    self.assertEqual(q2*self.q1,q_answer)

  def test_QuaternionAddFail(self):
    self.assertRaises(QuaternionException, self.q0.__add__, 1)
    self.assertRaises(QuaternionException, self.q0.__add__, 1.0)
    self.assertRaises(QuaternionException, self.q0.__add__, "")
    self.assertRaises(QuaternionException, self.q0.__add__, True)
    self.assertRaises(QuaternionException, self.q0.__add__, [])
    self.assertRaises(QuaternionException, self.q0.__add__, ())
    self.assertRaises(QuaternionException, self.q0.__add__, [])

  def test_QuaternionAddQuaternion(self):
    q1 = Quaternion(1.0,0.0,1.0,0.0)
    q2 = Quaternion(1.0,2.0,3.0,-3.0)
    qresult = Quaternion(2.0,2.0,4.0,-3.0)
    q3 = q1 + q2
    self.assertEqual(q3,qresult)
    q3 = q2 + q1
    self.assertEqual(q3,qresult)

  def test_QuaternionrotateVectorExceptionRaise(self):
    self.assertRaises(TypeError, self.q0.rotateVector, 1)
    self.assertRaises(TypeError, self.q0.rotateVector, 1.0)
    self.assertRaises(TypeError, self.q0.rotateVector, "")
    self.assertRaises(TypeError, self.q0.rotateVector, True)
    self.assertRaises(TypeError, self.q0.rotateVector, [])
    self.assertRaises(TypeError, self.q0.rotateVector, ())
    self.assertRaises(TypeError, self.q0.rotateVector, [])

    # make sure that it stops if quaternion is not a unit_vector
    self.assertRaises(QuaternionIsNotUnitException, self.q1.rotateVector, numpy.array([0.0,0.0,0.0]))

  def test_QuaternionrotateVector(self):
    test_angle = 2.0*numpy.pi/3.0
    test_rotateabout = numpy.ones(3)
    vector_to_be_rotated = numpy.array([1.0,2.0,3.0])
    rotation_q = Quaternion.fromAngleAndVector(test_angle,test_rotateabout)

    rotated_vector = rotation_q.rotateVector(vector_to_be_rotated)
    self.assertTrue(numpy.allclose(rotated_vector,numpy.array([3.0,1.0,2.0])))
    rotated_vector = rotation_q.rotateVector(rotated_vector)
    self.assertTrue(numpy.allclose(rotated_vector,numpy.array([2.0,3.0,1.0])))
    rotated_vector = rotation_q.rotateVector(rotated_vector)
    self.assertTrue(numpy.allclose(rotated_vector,numpy.array([1.0,2.0,3.0])))

# test cases built right in
if __name__ == '__main__':
  unittest.main()
