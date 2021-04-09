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

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3)
b.dtype.name
c = a + b
c
c.dtype.name
d = np.exp(c*1j)
d
d.dtype.name

b = np.arange(12).reshape(3, 4)
b
b.sum(axis=0)    # sum of each column
b.min(axis=1)    # min of each row
b.cumsum(axis=1)    # cumulative sum along each row

# ---- Universal Functions
B = np.arange(3)
B
np.exp(B)
np.sqrt(B)
C = np.array([2., -1., 4.])
np.add(B, C)

# ---- Indexing, Slicing and Iterating
a = np.arange(10)**3
a
a[2]
a[2:5]
a[:6:2] = 1000
a
a[ : : -1]

for i in a:
    print(i**(1/3.))

def f(x, y):
    return 10*x + y


b = np.fromfunction(f, (5, 4), dtype=int)
b
b[2, 3]
b[0:5, 1]
b[ : , 1]
b[1:3, : ]

c = np.array([[[0, 1, 2],
               [10, 11, 12]],
              [[100, 101, 102],
              [110, 112, 120]]])
c.shape
c[1, ...]
c[..., 2]

for row in b:
    print(row)

for element in b.flat:
    print(element)

# ==== Shape Manipulation
# ---- Changing the shape of an array
a = np.floor(10 * rg.random((3, 4)))
a
a.shape

a.ravel()    # returns the array, flattened
a.reshape(6, 2)    # returns the array with a modified shape
a.T    # returns the array, transposed
a.T.shape

a
a.resize(2, 6)
a

# If a dimension is given as -1 in a reshaping operation,
# the other dimensions are automatically calculated:
a.reshape(3, -1)

# ---- Stacking together different arrays
a = np.floor(10 * rg.random((2, 2)))
a
b = np.floor(10 * rg.random((2, 2)))
b
np.vstack((a, b))
np.hstack((a, b))

from numpy import newaxis
np.column_stack((a, b))
a = np.array([4., 2.])
b = np.array([3., 8.])
np.column_stack((a, b))
np.hstack((a, b))
a[:, newaxis]
np.column_stack((a[:, newaxis], b[:, newaxis]))
np.hstack((a[:, newaxis], b[:, newaxis]))

np.column_stack is np.hstack
np.row_stack is np.vstack

# ---- Splitting one array into several smaller ones
a = np.floor(10*rg.random((2, 12)))
a
# Split a into 3
np.hsplit(a, 3)
# Split a after the third and the fourth column
np.hsplit(a, (3, 4))

# ==== Copies and Views
# Simple assignments make no copy of objects or their data.
a = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])
b = a            # no new object is created
b is a           # a and b are two names for the same ndarray object
# Python passes mutable objects as references, so function calls make no copy.


def f(x):
    print(id(x))


id(a)                           # id is a unique identifier of an object
f(a)

# ---- View or Shallow Copy
c = a.view()
c is a
c.base is a
c.flags.owndata
c = c.reshape((2, 6))
a.shape
c[0, 4] = 1234
a
# Slicing an array returns a view of it:
s = a[:, 1:3]
s[:] = 10    # s[:] is a view of s. Note the difference between s = 10 and s[:] = 10
a

# ---- Deep Copy
# The copy method makes a complete copy of the array and its data.
d = a.copy()
d is a
d.base is a    # d doesn't share anything with a
d[0, 0] = 9999
a

a = np.arange(int(1e8))
b = a[:100].copy()
del a    # the memory of ``a`` can be released.

# ---- Functions and Methods Overview

# ============================================
#    Less Basic
# ============================================

# ==== Broadcasting rules

# ============================================
#    Advanced indexing and index tricks
# ============================================

# ==== Indexing with Arrays of Indices
a = np.arange(12)**2    # the first 12 square numbers
i = np.array([1, 1, 3, 8, 5])    # an array of indices
a[i]    # the elements of a at the positions i

j = np.array([[3, 4], [9, 7]])    # a bidimensional array of indices
a[j]    # the same shape as j

pallete = np.array([[0, 0, 0],    # black
                    [255, 0, 0],    # red
                    [0, 255, 0],    # green
                    [0, 0, 255],    # blue
                    [255, 255, 255]])    # white
image = np.array([[0, 1, 2, 0],
                 [0, 3, 4, 0]])    # each value corresponds to a color in the palette
pallete[image]

# We can also give indexes for more than one dimension.
a = np.arange(12).reshape(3, 4)
a
i = np.array([[0, 1],
              [1, 2]])    # indices for the first dim of a
j = np.array([[2, 1],
              [3, 3]])    # indices for the second dim
a[i, j]    # i and j must have equal shape
a[i, 2]
a[:, j]

l = (i, j)
a[l]

# Another common use of indexing with arrays is
# the search of the maximum value of time-dependent series:
time = np.linspace(20, 145, 5)    # time scale
data = np.sin(np.arange(20)).reshape(5, 4)    # 4 time-dependent series
time
data

# index of the maxima for each series
ind = data.argmax(axis=0)
ind

# times corresponding to the maxima
time_max = time[ind]
data_max = data[ind, range(data.shape[1])]
time_max
data_max
np.all(data_max == data.max(axis=0))

# You can also use indexing with arrays as a target to assign to:
a = np.arange(5)
a
a[[1, 3, 4]] = 0
a

# ==== Indexing with Boolean Arrays
