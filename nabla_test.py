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
    assert z.a==6 and z.b==8
    z = x - y
    assert z.a==-2 and z.b==-2
    z = x * y
    assert z.a==8 and z.b==22
    z = x / y
    assert z.a==0.5 and z.b==(3*4 - 2*5)/4**2

    x = Dual(2,3)
    y = 4
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.a==6 and z.b==3
    z = x - y
    assert z.a==-2 and z.b==3
    z = x * y
    assert z.a==8 and z.b==12
    z = x / y
    assert z.a==0.5 and z.b==(3*4 - 2*0)/4**2

    x = 2
    y = Dual(4,5)
    assert x<y and y>x and x<=y and y>=x
    z = x + y
    assert z.a==6 and z.b==5
    z = x - y
    assert z.a==-2 and z.b==-5
    z = x * y
    assert z.a==8 and z.b==10
    z = x / y
    assert z.a==0.5 and z.b==(0*4 - 2*5)/4**2

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
    assert z.a==9 and z.b==6
    z = cupow(4)
    assert close(z.a, 64) and close(z.b, 48)
    z = plusthree(3)
    assert z.a==6 and z.b==1

    # kwargs
    z = sq(x=3)
    assert z.a==9 and z.b==6

    # Transcendental functions
    x = Dual(5,1)
    z = nabla.sin(x)
    assert close(z.a, math.sin(5)) and close(z.b, math.cos(5))
    z = nabla.cos(x)
    assert close(z.a, math.cos(5)) and close(z.b, -math.sin(5))
    z = nabla.exp(x)
    assert close(z.a, math.exp(5)) and close(z.b, math.exp(5))
    z = nabla.log(x)
    assert close(z.a, math.log(5)) and close(z.b, 1/5)
    
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
    assert close(w.a, f) and close(w.b, dfdy)

    fx = nabla.grad(0)(func)(x,y,z)
    fy = nabla.grad(1)(func)(x,y,z)
    fz = nabla.grad(2)(func)(x,y,z)
    
    assert close(fx.a, f) and close(fx.b, dfdx)
    assert close(fy.a, f) and close(fy.b, dfdy)
    assert close(fz.a, f) and close(fz.b, dfdz)

    # Get full gradient
    # TODO make this work
    fgrad = nabla.grad()(func)(x,y,z)
    print(fgrad)

def test_grad_multimulti():
    pass


if __name__=="__main__":
    test_grad_multivar()

