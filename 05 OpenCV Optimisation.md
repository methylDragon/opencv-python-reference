# OpenCV Python Optimisation Cheatsheet

Author: methylDragon  
Contains a syntax reference and code snippets for OpenCV for Python!  
Note that this document is more or less based on the tutorials on https://docs.opencv.org    
With some personal notes from me!    

------

## Pre-Requisites

### Required

- Python knowledge, this isn't a tutorial!
- OpenCV installed



## 1. Introduction

Not much of an introduction here. OpenCV is just really great!

Since this is a work in progress, it's not going to be very well organised.

```python
# These will have been assumed to have been run

import cv2 as cv2, cv
import numpy as np
```

If you need additional help or need a refresher on the parameters, feel free to use:

```python
help(cv.FUNCTION_YOU_NEED_HELP_WITH)
```



## 2. Optimisation

### Enabling Optimisation

```python
cv.useOptimized() # Returns True if optimisation is enabled
cv.setUseOptimized(True) # Set it to True
```



### Measuring Performance

Timing your code is important!

**Tick comparison**

```python
cv.getTickCount() # Current clock cycles
cv.getTickFrequency() # Number of clock cycles per second

# Example use
e1 = cv.getTickCount()
# your code execution
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()
```

**Using Timeit**

**Example**

```python
import timeit

def performSearch(array):
    array.sort()

arrayTest = ["X"]*1000

if __name__ == "__main__":
    print(timeit.timeit("performSearch(arrayTest)",
                        "from __main__ import performSearch, arrayTest",
                        repeat=3,
                        number=10))
```



### Cython

- Cython code compiles to C, making it way faster than just pure Python code!
- Install Cython with `pip install cython` or `conda install cython`

**Note: Using Cython to optimise your OpenCV script will only work generally for the for loops, since the OpenCV Python API is actually a Python wrapper for already fairly optimised C++ code.**

Ok!

So.... You're going to need to check the Cython reference I made, or know how to use Cython. But generally...

You need a

**setup.py**

```cython
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize('script_file.pyx'
)
```

**script_file.pyx**

```cython
# Source: https://www.pyimagesearch.com/2017/08/28/fast-optimized-for-pixel-loops-with-opencv-and-python/

import cython
 
@cython.boundscheck(False)
cpdef unsigned char[:, :] threshold_fast(int T, unsigned char [:, :] image):
    # set the variable extension types
    cdef int x, y, w, h
    
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]
    
    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = 255 if image[y, x] >= T else 0
    
    # return the thresholded image
    return image
```

And then compile it with

```shell
$ python3 setup.py build_ext --inplace
```

Then import it!

```python
from script_file import threshold_fast

# Now you can use it!
```



```
                            .     .
                         .  |\-^-/|  .    
                        /| } O.=.O { |\
```



------

 [![Yeah! Buy the DRAGON a COFFEE!](assets/COFFEE%20BUTTON%20%E3%83%BE(%C2%B0%E2%88%87%C2%B0%5E).png)](https://www.buymeacoffee.com/methylDragon)

