
# Nabla

_Note: this is an unmaintained personal project, use at your own risk._

Automatic, machine-precision forward differentiation in python using dual numbers.

```python
from nabla import grad

def f(x):
    return 3*x*x

print(grad(f)(5))
```

    Dual(75,  [30.])

[Example use for logistic regression](examples/mnist/logistic.ipynb)

## Support for multiple variables


```python
def f(x, y, param, z):
    return 2*x*y + z**4

x, y, z, param = 1, 2, 3, "this is a non-numeric parameter"

print(f(x, y, param, z))
# Get the gradient w.r.t. x,y,z
# The non-numeric parameter is automatically ignored
print(grad(f)(x, y, param, z))
```

    85
    Dual(85,  [  4.   2. 108.])


## Specify variables explicitly by position


```python
# Find gradient w.r.t y,x
print(grad([1,0])(f)(x, y, param, z))
```

    Dual(85,  [2. 4.])


## Use decorators; interop with numpy


```python
from numpy import sin, cos

@grad
def f(x, y):
    return sin(x)*cos(y)

print(f(1,2))

# nabla can automatically differentiate w.r.t. a combination of numpy array entries and other function arguments: 
print(f(np.array([1,2,3]), 2))
```

    Dual(-0.35017548837401463,  [-0.2248451 -0.7651474])
    [Dual(-0.35017548837401463,  [-0.2248451 -0.        -0.        -0.7651474])
     Dual(-0.37840124765396416,  [ 0.          0.17317819  0.         -0.82682181])
     Dual(-0.05872664492762098,  [ 0.          0.          0.41198225 -0.12832006])]


## Gradient descent without any extra code


```python
from nabla import minimise

def f(x, y, z):
        return sin(x+1) + 2*cos(y-1) + (z-1)**2

x0, fval, gradient = minimise(f, [0, 0, 0])
print("Minimum found at x0 = {}".format(x0))
print("Function, gradient at minimum = {}, {}\n".format(fval, gradient))

# Can also minimise w.r.t. a subset of variables
# Here we minimise w.r.t. x and y while holding z=0 fixed
x0, fval, gradient = minimise(f, [0, 0, 0], variables=[0,1])
print("Minimum found at x0 = {}".format(x0))
print("Function, gradient at minimum = {}, {}\n".format(fval, gradient))
```

    Minimum found at x0 = [-2.57079546 -2.14159265  1.        ]
    Function, gradient at minimum = -2.9999999999996243, [ 8.66727231e-07  1.62321409e-14 -3.77475828e-15]
    
    Minimum found at x0 = [-2.57079546 -2.14159265  0.        ]
    Function, gradient at minimum = -1.9999999999996243, [8.66727231e-07 1.62321409e-14]
    


## Comparison with finite-difference


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def f(x):
    return np.sqrt(3*x + np.sin(x)**2)/(4*x**3 + 2*x + 1)

def analytical_derivative(x):
    A = (2*np.sin(x)*np.cos(x)+3)/(2*(4*x**3 + 2*x + 1)*np.sqrt(3*x + np.sin(x)**2))
    B = (12*x**2 + 2)*np.sqrt(3*x + np.sin(x)**2) / (4*x**3 + 2*x + 1)**2
    return A - B

x = 1
dfdx_nabla = grad(f)(x).dual
dfdx_analytic = analytical_derivative(x)

eps = np.logspace(-15, -3)
dfdx_fd = np.zeros(eps.shape)
for i,e in enumerate(eps):
    dfdx_fd[i] = (f(x+e) - f(x))/e

err_nabla = np.abs(dfdx_nabla - dfdx_analytic) * np.ones(eps.shape)
err_fd = np.abs(dfdx_fd - dfdx_analytic)
    
# Plot error
plt.loglog(eps, err_fd, label='Finite-difference')
plt.loglog(eps, err_nabla, label='Nabla (dual numbers)')
plt.xlabel('Finite-difference step')
plt.ylabel('Error in gradient')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fce4983bfd0>




![png](README_files/README_11_1.png)


Compare time taken:


```python
%timeit -n10000 grad(f)(x)
%timeit -n10000 (f(x+1e-8) - f(x))/1e-8
```

    124 µs ± 6.93 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    8.27 µs ± 366 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

