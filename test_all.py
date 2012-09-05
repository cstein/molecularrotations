#!/usr/bin/env python
import unittest

import vector
import quaternion

if __name__ == '__main__':
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(vector.TestVector))
  suite.addTest(unittest.makeSuite(quaternion.TestQuaternion))
  unittest.TextTestRunner().run(suite)
