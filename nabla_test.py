#!/usr/bin/env python3

import math

import nabla
from nabla import grad, Dual

def close(x, y):
    return abs(x-y)<1e-12

def test_dual():
    x = Dual(2,3)
    y = Dual(4,5)
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual==8
    z = x - y
    assert z.real==-2 and z.dual==-2
    z = x * y
    assert z.real==8 and z.dual==22
    z = x / y
    assert z.real==0.5 and z.dual==(3*4 - 2*5)/4**2

    x = Dual(2,3)
    y = 4
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual==3
    z = x - y
    assert z.real==-2 and z.dual==3
    z = x * y
    assert z.real==8 and z.dual==12
    z = x / y
    assert z.real==0.5 and z.dual==(3*4 - 2*0)/4**2

    x = 2
    y = Dual(4,5)
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.real==6 and z.dual==5
    z = x - y
    assert z.real==-2 and z.dual==-5
    z = x * y
    assert z.real==8 and z.dual==10
    z = x / y
    assert z.real==0.5 and z.dual==(0*4 - 2*5)/4**2

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
    assert z.real==9 and z.dual==6
    z = cupow(4)
    assert close(z.real, 64) and close(z.dual, 48)
    z = plusthree(3)
    assert z.real==6 and z.dual==1

    # kwargs
    z = sq(x=3)
    assert z.real==9 and z.dual==6

    # Transcendental functions
    x = Dual(5,1)
    z = nabla.sin(x)
    assert close(z.real, math.sin(5)) and close(z.dual, math.cos(5))
    z = nabla.cos(x)
    assert close(z.real, math.cos(5)) and close(z.dual, -math.sin(5))
    z = nabla.exp(x)
    assert close(z.real, math.exp(5)) and close(z.dual, math.exp(5))
    z = nabla.log(x)
    assert close(z.real, math.log(5)) and close(z.dual, 1/5)
    
def test_grad_multivar():
    def func(x, y, z):
        return 2*x*y**2*nabla.cos(z)

    @grad(1)
    def funcgrad(x, y, z):
        return 2*x*y**2*nabla.cos(z)

    x = 2
    y = 3
    z = 4
    f = 2*x*y**2*math.cos(z)
    dfdx = 2*y**2*math.cos(z)
    dfdy = 4*x*y*math.cos(z)
    dfdz = -2*x*y**2*math.sin(z)

    w = funcgrad(x,y,z)
    assert close(w.real, f) and close(w.dual, dfdy)

    fx = nabla.grad(0)(func)(x,y,z)
    fy = nabla.grad(1)(func)(x,y,z)
    fz = nabla.grad(2)(func)(x,y,z)
    
    assert close(fx.real, f) and close(fx.dual, dfdx)
    assert close(fy.real, f) and close(fy.dual, dfdy)
    assert close(fz.real, f) and close(fz.dual, dfdz)

    # Get full gradient
    # TODO make this work
    fgrad = nabla.grad()(func)(x,y,z)
    print(fgrad)

def test_grad_multimulti():
    pass


if __name__=="__main__":
    test_grad_multivar()

