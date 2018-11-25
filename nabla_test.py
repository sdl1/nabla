#!/usr/bin/env python3

import math

import nabla
from nabla import grad, Dual

def close(x, y, eps=1e-12):
    return abs(x-y)<eps

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
    z = nabla.sin(x)
    assert close(z.real, math.sin(5)) and close(z.dual[0], math.cos(5))
    z = nabla.cos(x)
    assert close(z.real, math.cos(5)) and close(z.dual[0], -math.sin(5))
    z = nabla.exp(x)
    assert close(z.real, math.exp(5)) and close(z.dual[0], math.exp(5))
    z = nabla.log(x)
    assert close(z.real, math.log(5)) and close(z.dual[0], 1/5)
    
def test_grad_multivar():
    def func(x, y, z):
        return 2*x*y**2*nabla.cos(z)

    @grad(1)
    def funcgrad(x, y, z):
        return 2*x*y**2*nabla.cos(z)

    @grad
    def funcfullgrad(x, y, z):
        return 2*x*y**2*nabla.cos(z)

    x = 2
    y = 3
    z = 4
    f = 2*x*y**2*math.cos(z)
    dfdx = 2*y**2*math.cos(z)
    dfdy = 4*x*y*math.cos(z)
    dfdz = -2*x*y**2*math.sin(z)

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

def test_grad_multimulti():

    def func(x, y, z):
        f0 = nabla.sqrt(x*nabla.exp(y) + nabla.cos(z))
        f1 = x*y**2 + nabla.cos(z) + 5
        f2 = (x + y*z)/nabla.sqrt(x + y) * 0.1
        f3 = nabla.sin(nabla.cos(nabla.sqrt(x*y + z)))
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




if __name__=="__main__":
    test_grad_multimulti()

