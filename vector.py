import numpy

class VectorException(BaseException):
  pass

class Vector(object):
  def __init__(self,x,y,z):
    self._x = x
    self._y = y
    self._z = z

  def X(self):
    return self._x

  def Y(self):
    return self._y

  def Z(self):
    return self._z

  def Magnitude(self):
    return numpy.sqrt(self*self)

  def isUnit(self):
    threshold = 1.0e-6
    return (self.Magnitude() - 1.0) < threshold

  def Normalize(self):
    magnitude = self.Magnitude()
    self._x /= magnitude
    self._y /= magnitude
    self._z /= magnitude

  ### __XXX__ overrides
  def __repr__(self):
    return "Vector(%f,%f,%f)" % (self._x, self._y, self._z)

  def __str__(self):
    return "%6.2f%6.2f%6.2f" % (self._x, self._y, self._z)

  def __eq__(self,other):
    threshold = 1.0e-6
    eq_x = (self._x - other.X()) < threshold
    eq_y = (self._y - other.Y()) < threshold
    eq_z = (self._z - other.Z()) < threshold
    return eq_x and eq_y and eq_z

  def __ne__(self,other):
    return not self == other

  def __add__(self,other):
    if isinstance(other,Vector):
      return Vector(self._x + other.Z(), self._y + other.Y(), self._z + other.Z())
    else:
      raise VectorException("Operation not allowed")

  def __sub__(self,other):
    if isinstance(other,Vector):
      return Vector(self._x - other.X(), self._y - other.Y(), self._z - other.Z())
    else:
      raise VectorException("Operation not allowed")

  def __mul__(self,other):
    if isinstance(other,Vector):
      return self._x*other.X() + self._y*other.Y() + self._z*other.Z()
    if isinstance(other,int) or isinstance(other,float):
      return Vector(self._x*other, self._y*other, self._z*other)
    raise VectorException("Operation not allowed")

  def __rmul__(self,other):
    return self.__mul__(other)

  def __pow__(self,other):
    if isinstance(other,Vector):
      x = self._y*other.Z() - other.Y()*self._z
      y = self._z*other.X() - other.Z()*self._x
      z = self._x*other.Y() - other.X()*self._y
      return Vector(x,y,z)
    else:
      raise VectorException("Operation now allowed")

import unittest
class TestVector(unittest.TestCase):
  def setUp(self):
    self.v0 = Vector(0.0,0.0,0.0)
    self.v1 = Vector(1.0,1.0,1.0)

  def test_VectorCreationBasicV0(self):
    self.assertEqual(self.v0.X(), 0.0)
    self.assertEqual(self.v0.Y(), 0.0)
    self.assertEqual(self.v0.Z(), 0.0)

  def test_VectorCreationBasicV1(self):
    self.assertEqual(self.v1.X(), 1.0)
    self.assertEqual(self.v1.Y(), 1.0)
    self.assertEqual(self.v1.Z(), 1.0)

  def test_VectorMagnitude(self):
    self.assertEqual(self.v0.Magnitude(),0.0)
    self.assertEqual(self.v1.Magnitude(),numpy.sqrt(3))

  def test_VectorIsUnit(self):
    v1 = Vector(1.0,0.0,0.0)
    v2 = Vector(1.0,1.0,0.0)
    self.assertTrue(v1.isUnit())
    self.assertFalse(v2.isUnit())

  def test_VectorNormalize(self):
    v2 = Vector(1.0,1.0,0.0)
    v2.Normalize()
    self.assertTrue(v2.isUnit())

  def test_VectorReprAndStr(self):
    self.assertEqual(repr(self.v0),"Vector(0.000000,0.000000,0.000000)")
    self.assertEqual(repr(self.v1),"Vector(1.000000,1.000000,1.000000)")
    self.assertEqual(str(self.v0), "  0.00  0.00  0.00")
    self.assertEqual(str(self.v1), "  1.00  1.00  1.00")

  def test_VectorEqual(self):
    v0 = Vector(0.0,0.0,0.0)
    self.assertTrue(self.v0 == v0)
    self.assertFalse(self.v1 == v0)

  def test_VectorNotEqual(self):
    v0 = Vector(0.0,0.0,0.0)
    self.assertFalse(self.v0 != v0)
    self.assertTrue(self.v1 != v0)

  def test_VectorAdd(self):
    v2 = Vector(2.0,0.0,0.0)
    vresult = Vector(3.0,1.0,1.0)
    self.assertEqual(self.v1 - v2,vresult)

  def test_VectorSubtract(self):
    v2 = Vector(2.0,0.0,0.0)
    vresult = Vector(-1.0,1.0,1.0)
    self.assertEqual(self.v1 - v2,vresult)

  def test_VectorMultiplyVectors(self):
    self.assertEqual(self.v0*self.v0,0.0)
    self.assertEqual(self.v1*self.v1,3.0)
    self.assertEqual(self.v0*self.v1,0.0)
    v2 = Vector(2.0,-3.0,1.0)
    v3 = Vector(4.0,1.0,-4.0)
    self.assertEqual(v2*v3,1.0)

  def test_VectorPowerUnitVectors(self):
    v1 = Vector(1.0,0.0,0.0)
    v2 = Vector(0.0,1.0,0.0)
    self.assertEqual(v1**v2, Vector(0.0,0.0,1.0))

  def test_VectorPowerRealVectors(self):
    v1 = Vector(1.0,2.0,4.0)
    v2 = Vector(-1.0,3.0,2.0)
    self.assertEqual(v1**v2, Vector(8.0,-6.0,5.0))

  def test_VectorMultplyNumbers(self):
    v2 = Vector(2.0,3.0,1.0)
    self.assertEqual(self.v0*1,Vector(0.0,0.0,0.0))
    self.assertEqual(self.v1*0,Vector(0.0,0.0,0.0))
    self.assertEqual(self.v1*2.0,Vector(2.0,2.0,2.0))
    self.assertEqual(v2*-3,Vector(-6.0,-9.0,-3.0))

  def test_VectorAddWrongOperation(self):
    self.assertRaises(VectorException, self.v0.__add__, 1)
    self.assertRaises(VectorException, self.v0.__add__, 1.0)
    self.assertRaises(VectorException, self.v0.__add__, "")
    self.assertRaises(VectorException, self.v0.__add__, True)
    self.assertRaises(VectorException, self.v0.__add__, [])
    self.assertRaises(VectorException, self.v0.__add__, ())
    self.assertRaises(VectorException, self.v0.__add__, {})

  def test_VectorSubtractWrongOperation(self):
    self.assertRaises(VectorException, self.v0.__sub__, 1)
    self.assertRaises(VectorException, self.v0.__sub__, 1.0)
    self.assertRaises(VectorException, self.v0.__sub__, "")
    self.assertRaises(VectorException, self.v0.__sub__, True)
    self.assertRaises(VectorException, self.v0.__sub__, [])
    self.assertRaises(VectorException, self.v0.__sub__, ())
    self.assertRaises(VectorException, self.v0.__sub__, {})

  def test_VectorMultiplyOperation(self):
    self.assertRaises(VectorException, self.v0.__mul__, "")
    #self.assertRaises(VectorException, self.v0.__mul__, True)
    self.assertRaises(VectorException, self.v0.__mul__, [])
    self.assertRaises(VectorException, self.v0.__mul__, ())
    self.assertRaises(VectorException, self.v0.__mul__, {})

  def test_VectorPowerWrongOperation(self):
    self.assertRaises(VectorException, self.v0.__pow__, 1)
    self.assertRaises(VectorException, self.v0.__pow__, 1.0)
    self.assertRaises(VectorException, self.v0.__pow__, "")
    self.assertRaises(VectorException, self.v0.__pow__, True)
    self.assertRaises(VectorException, self.v0.__pow__, [])
    self.assertRaises(VectorException, self.v0.__pow__, ())
    self.assertRaises(VectorException, self.v0.__pow__, {})

if __name__ == '__main__':
  unittest.main()
