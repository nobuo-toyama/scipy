# numby
# ==== numpy quickstart

import numpy as np
a = np.arange(15).reshape(3, 5)
a
a.shape
a.ndim
a.dtype.name
a.itemsize
type(a)
b = np.array([6, 7, 8])
b
type(b)

# ---- Array Creation
import numpy as np
a = np.array([2, 3, 4])
a
a.dtype
b = np.array([1.5, 2.2, 3.1])
b.dtype

b = np.array([(1.5, 2, 3), (4, 5, 6)])
b

# The type of the array can also be explicitly specified at creation time:
c = np.array([[1, 2], [3, 4]], dtype=complex)
c

np.zeros((3, 4))
np.ones((2, 3, 4), dtype=np.int32)
np.empty((2, 3))

# To create sequences of numbers, NumPy provides the arange function
# which is analogous to the Python built-in range, but returns an array.
np.arange(10, 30, 5)
np.arange(0, 2, 0.3)
from numpy import pi
np.linspace(0, 2, 9)
x = np.linspace(0, 2*pi, 100)
f = np.sin(x)

# ---- Printing array
a = np.arange(6)
print(a)
b = np.arange(12).reshape(4, 3)
print(b)
c = np.arange(24).reshape(2, 3, 4)
print(c)
print(np.arange(10000))
print(np.arange(10000).reshape(100, 100))

# ---- Basic Operations
a = np.array([20, 30, 40, 50])
b = np.arange(4)
b
c = a - b
c
b**2
10*np.sin(a)
a<35

A = np.array([[1, 1],
             [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
A * B
A @ B
A.dot(B)

rg = np.random.default_rng(1)     # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3
a
b += a
b