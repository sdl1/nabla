#!/usr/bin/env python3

import numpy as np

import nabla
from nabla import grad, Dual, minimise

def close(x, y, eps=1e-12):
    return abs(x-y)<eps

def dualclose(x, y, eps=1e-12):
    isclose = close(x.real, y.real, eps)
    for i in range(x.nvars):
        isclose = isclose and close(x.dual[i], y.dual[i], eps)
    return isclose

def test_dual():
    x = Dual(2,3)
    y = Dual(4,5)
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual[0]==8
    z = x - y
    assert z.real==-2 and z.dual[0]==-2
    z = x * y
    assert z.real==8 and z.dual[0]==22
    z = x / y
    assert z.real==0.5 and z.dual[0]==(3*4 - 2*5)/4**2

    x = Dual(2,3)
    y = 4
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual[0]==3
    z = x - y
    assert z.real==-2 and z.dual[0]==3
    z = x * y
    assert z.real==8 and z.dual[0]==12
    z = x / y
    assert z.real==0.5 and z.dual[0]==(3*4 - 2*0)/4**2

    x = 2
    y = Dual(4,5)
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual[0]==5
    z = x - y
    assert z.real==-2 and z.dual[0]==-5
    z = x * y
    assert z.real==8 and z.dual[0]==10
    z = x / y
    assert z.real==0.5 and z.dual[0]==(0*4 - 2*5)/4**2

    sqrty = np.sqrt(y)
    ytohalf = y ** 0.5
    assert close(sqrty.real, ytohalf.real) and close(sqrty.dual[0], ytohalf.dual[0])

    z = 2**y
    zalt = Dual(2)**y
    assert close(z.real, zalt.real) and close(z.dual[0], zalt.dual[0])

    x = Dual(2,3)
    y = Dual(4,5)
    w = x*y
    z = nabla.dot([x], [y])
    assert dualclose(z, w)
    z = nabla.dot(np.array([x]), np.array([y]))
    assert dualclose(z, w)
    z = nabla.dot(np.array([x, y]), np.array([x, x]))
    assert dualclose(z, x*x + y*x)
    z = nabla.dot(np.array([x, 3]), np.array([x, x]))
    assert dualclose(z, x*x + 3*x)

def test_dual_multivar():
    x = Dual(2, [3, 1])
    y = Dual(4, [5, 2])
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual[0]==8 and z.dual[1]==3
    z = x - y
    assert z.real==-2 and z.dual[0]==-2 and z.dual[1]==-1
    z = x * y
    assert z.real==8 and z.dual[0]==22 and z.dual[1]==8
    z = x / y
    assert z.real==0.5 and z.dual[0]==(3*4 - 2*5)/4**2 and z.dual[1]==(1*4 - 2*2)/4**2

    x = Dual(2, [3, 1])
    y = 4
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual[0]==3 and z.dual[1]==1
    z = x - y
    assert z.real==-2 and z.dual[0]==3 and z.dual[1]==1
    z = x * y
    assert z.real==8 and z.dual[0]==12 and z.dual[1]==4
    z = x / y
    assert z.real==0.5 and z.dual[0]==(3*4 - 2*0)/4**2 and z.dual[1]==(1*4 - 2*0)/4**2

    x = 2
    y = Dual(4, [5, 2])
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual[0]==5 and z.dual[1]==2
    z = x - y
    assert z.real==-2 and z.dual[0]==-5 and z.dual[1]==-2
    z = x * y
    assert z.real==8 and z.dual[0]==10 and z.dual[1]==4
    z = x / y
    assert z.real==0.5 and z.dual[0]==(0*4 - 2*5)/4**2 and z.dual[1]==(0*4 - 2*2)/4**2

def test_gradsimple():
    @grad
    def sq(x):
        return x*x
    @grad
    def cupow(x):
        return x**3
    @grad
    def plusthree(x):
        return x + 3

    z = sq(3)
    assert z.real==9 and z.dual[0]==6
    z = cupow(4)
    assert close(z.real, 64) and close(z.dual[0], 48)
    z = plusthree(3)
    assert z.real==6 and z.dual[0]==1

    def f(x):
        return x*x

    w = grad(0)(f)(np.array([1,2,3]))
    assert dualclose(w[0], Dual(1, [2, 0, 0]))
    assert dualclose(w[1], Dual(4, [0, 4, 0]))
    assert dualclose(w[2], Dual(9, [0, 0, 6]))

    w = grad(f)(np.array([1,2,3]))
    assert dualclose(w[0], Dual(1, [2, 0, 0]))
    assert dualclose(w[1], Dual(4, [0, 4, 0]))
    assert dualclose(w[2], Dual(9, [0, 0, 6]))

    # Element-wide mult of np arrays
    A = np.array([[1,2], [3,4]])
    w = grad(f)(A)
    assert dualclose(w[0,0], Dual(1, [2, 0, 0, 0]))
    assert dualclose(w[0,1], Dual(4, [0, 4, 0, 0]))
    assert dualclose(w[1,0], Dual(9, [0, 0, 6, 0]))
    assert dualclose(w[1,1], Dual(16, [0, 0, 0, 8]))

    # Reduction
    def f(x):
        return np.sum(x*x)
    w = grad(f)(np.array([1,2,3]))
    assert dualclose(w, Dual(14, [2, 4, 6]))
    w = grad(0)(f)(np.array([1,2,3]))
    assert dualclose(w, Dual(14, [2, 4, 6]))

    # kwargs
    z = sq(x=3)
    assert z.real==9 and z.dual[0]==6

    # non-numeric args
    def fn(x, y):
        print(y)
        return 2*x**3

    fgrad = nabla.grad()(fn)(2, y="non-numeric")
    assert close(fgrad.real, 16) and fgrad.nvars==1 and close(fgrad.dual[0], 24)

    fgrad = nabla.grad()(fn)(y="non-numeric", x=2)
    assert close(fgrad.real, 16) and fgrad.nvars==1 and close(fgrad.dual[0], 24)

    # Transcendental functions
    x = Dual(5,1)
    z = np.sin(x)
    assert close(z.real, np.sin(5)) and close(z.dual[0], np.cos(5))
    z = np.cos(x)
    assert close(z.real, np.cos(5)) and close(z.dual[0], -np.sin(5))
    z = np.exp(x)
    assert close(z.real, np.exp(5)) and close(z.dual[0], np.exp(5))
    z = np.log(x)
    assert close(z.real, np.log(5)) and close(z.dual[0], 1/5)
    
def test_grad_multivar():
    def func(x, y, z):
        return 2*x*y**2*np.cos(z)

    @grad(1)
    def funcgrad(x, y, z):
        return 2*x*y**2*np.cos(z)

    @grad
    def funcfullgrad(x, y, z):
        return 2*x*y**2*np.cos(z)

    x = 2
    y = 3
    z = 4
    f = 2*x*y**2*np.cos(z)
    dfdx = 2*y**2*np.cos(z)
    dfdy = 4*x*y*np.cos(z)
    dfdz = -2*x*y**2*np.sin(z)

    w = funcgrad(x,y,z)
    assert close(w.real, f) and close(w.dual[0], dfdy)

    fx = nabla.grad(0)(func)(x,y,z)
    fy = nabla.grad(1)(func)(x,y,z)
    fz = nabla.grad(2)(func)(x,y,z)
    
    assert close(fx.real, f) and close(fx.dual[0], dfdx)
    assert close(fy.real, f) and close(fy.dual[0], dfdy)
    assert close(fz.real, f) and close(fz.dual[0], dfdz)

    w = funcfullgrad(x,y,z)
    
    assert close(w.real, f) and close(w.dual[0], dfdx)
    assert close(w.real, f) and close(w.dual[1], dfdy)
    assert close(w.real, f) and close(w.dual[2], dfdz)

    fgrad = nabla.grad()(func)(x,y,z)
    assert close(fgrad.real, f)
    assert close(fgrad.dual[0], dfdx)
    assert close(fgrad.dual[1], dfdy)
    assert close(fgrad.dual[2], dfdz)

    fgrad = nabla.grad([0,2])(func)(x,y,z)
    assert close(fgrad.real, f)
    assert close(fgrad.dual[0], dfdx)
    assert close(fgrad.dual[1], dfdz)

    fgrad = nabla.grad([2,1,0])(func)(x,y,z)
    assert close(fgrad.real, f)
    assert close(fgrad.dual[0], dfdz)
    assert close(fgrad.dual[1], dfdy)
    assert close(fgrad.dual[2], dfdx)

    # Non-numeric and kwargs
    def func(x, y, z, w):
        print(x)
        return y + z + w
    x, y, z, w = "non-numeric", 1, 2, 3
    fgrad = nabla.grad()(func)(x, y, z, w)
    assert fgrad.real==6 and fgrad.dual[0]==1 and fgrad.dual[1]==1 and fgrad.dual[2]==1
    fgrad = nabla.grad()(func)(x=x, y=y, z=z, w=w)
    assert fgrad.real==6 and fgrad.dual[0]==1 and fgrad.dual[1]==1 and fgrad.dual[2]==1

    def f(x, y, param, z):
        return 2*x*y + z**4

    x, y, z, param = 1, 2, 3, "this is a non-numeric parameter"

    w = grad(f)(x, y, param, z)
    assert dualclose(w, Dual(85, [4, 2, 108]))
    w = grad([1,0])(f)(x, y, param, z)
    assert dualclose(w, Dual(85, [2, 4]))

    # Reduction
    def f(x, y):
        return np.sum(x*x) + np.sum(y*y)

    x, y = np.array([1,2,3]), np.array([4,5])
    w = grad(f)(x, y)
    assert dualclose(w, Dual(14+16+25, [2, 4, 6, 8, 10]))
    w = grad(1)(f)(x, y)
    assert dualclose(w, Dual(14+16+25, [8, 10]))

def test_grad_multimulti():

    def func(x, y, z):
        f0 = np.sqrt(x*np.exp(y) + np.cos(z))
        f1 = x*y**2 + np.cos(z) + 5
        f2 = (x + y*z)/np.sqrt(x + y) * 0.1
        f3 = np.sin(np.cos(np.sqrt(x*y + z)))
        return [f0, f1, f2, f3]

    x, y, z = 1, 2, 3

    # Numerical derivatives
    eps = 1e-6
    fpx = func(x+eps, y, z)
    fpy = func(x, y+eps, z)
    fpz = func(x, y, z+eps)
    f = func(x, y, z)
    
    fx = []
    fy = []
    fz = []
    for i in range(4):
        fx.append( (fpx[i] - f[i])/eps )
        fy.append( (fpy[i] - f[i])/eps )
        fz.append( (fpz[i] - f[i])/eps )

    fgrad = nabla.grad()(func)(x, y, z)
    print(fgrad)

    for i in range(4):
        assert close(fx[i], fgrad[i].dual[0], 1e-5)
        assert close(fy[i], fgrad[i].dual[1], 1e-5)
        assert close(fz[i], fgrad[i].dual[2], 1e-5)

def test_minimise():
    def f(x, y, z):
        return np.sin(x+1) + 2*np.cos(y-1) + (z-1)**2

    x0, fval, gradient = minimise(f, [0, 0, 0])
    assert close(x0[0], -2.57078753, 1e-5) and close(x0[1], -2.14159265, 1e-5) and close(x0[2], 1.0, 1e-5)
    assert close(fval, -3.0, 1e-5)
    assert close(gradient[0], 0.0, 1e-5)
    assert close(gradient[1], 0.0, 1e-5)
    assert close(gradient[2], 0.0, 1e-5)

    x0, fval, gradient = minimise(f, [0, 0, 0], variables=[0,1])
    assert close(x0[0], -2.57078753, 1e-5) and close(x0[1], -2.14159265, 1e-5) and close(x0[2], 0.0, 1e-5)
    assert close(fval, -2.0, 1e-5)
    assert close(gradient[0], 0.0, 1e-5)
    assert close(gradient[1], 0.0, 1e-5)




if __name__=="__main__":
    test_gradsimple()

