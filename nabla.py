import functools
import numbers
import math

def argstodual(e):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = [Dual(arg,e) if isinstance(arg, numbers.Number) else arg for arg in args]
            for key in kwargs:
                if isinstance(kwargs[key], numbers.Number):
                    kwargs[key] = Dual(kwargs[key], e)
            return func(*args, **kwargs)
        return wrapper
    return decorator

#def grad(func):
#    @functools.wraps(func)
#    @argstodual(1)
#    def wrapper(*args, **kwargs):
#        return func(*args, **kwargs)
#    return wrapper

def grad(*args):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Full gradient case
            if varpos==None:
                args = [Dual(arg,1) if isinstance(arg, numbers.Number) else arg for arg in args]
                for key in kwargs:
                    if isinstance(kwargs[key], numbers.Number):
                        kwargs[key] = Dual(kwargs[key], 1)
            else:
                if kwargs:
                    raise Exception("Keyword arguments not supported.")
                args = list(args)
                for i in varpos:
                    args[i] = Dual(args[i], 1)
            print(args)
            return func(*args, **kwargs)
        return wrapper
    if len(args)==1 and callable(args[0]):
        # @grad with no arguments - take full gradient
        varpos = None
        return decorator(args[0])
    elif len(args)==0:
        # grad() with no arguments
        varpos = None
    else:
        varpos = args
    return decorator

# f(a + be) = f(a) + b fprime(a) e

@argstodual(0)
def exp(x):
    expa = math.exp(x.real)
    return Dual(expa, x.dual*expa)

@argstodual(0)
def log(x):
    return Dual(math.log(x.real), x.dual / x.real)

@argstodual(0)
def sin(x):
    return Dual(math.sin(x.real), x.dual * math.cos(x.real))

@argstodual(0)
def cos(x):
    return Dual(math.cos(x.real), -x.dual * math.sin(x.real))

class Dual:
    # TODO multiple variables
    def __init__(self, real=0, dual=0, nvars=1):
        #assert len(dual)==nvars
        self.nvars = nvars
        self.real = real
        self.dual = dual
    @argstodual(0)
    def __add__(self, other):
        return Dual(self.real + other.real, self.dual + other.dual)
    @argstodual(0)
    def __sub__(self, other):
        return Dual(self.real - other.real, self.dual - other.dual)
    @argstodual(0)
    def __mul__(self, other):
        return Dual(self.real*other.real, self.real*other.dual + self.dual*other.real)
    @argstodual(0)
    def __truediv__(self, other):
        return Dual(self.real/other.real, (self.dual*other.real - self.real*other.dual)/(other.real*other.real))
    # object.__floordiv__(self, other)
    # object.__mod__(self, other)
    # object.__divmod__(self, other)
    @argstodual(0)
    def __pow__(self, other, *modulo):
        if modulo:
            return NotImplemented
        else:
            return exp(other*log(self))
    # object.__lshift__(self, other)
    # object.__rshift__(self, other)
    # object.__and__(self, other)
    # object.__xor__(self, other)
    # object.__or__(self, other)

    @argstodual(0)
    def __radd__(self, other):
        return other.__add__(self)
    @argstodual(0)
    def __rsub__(self, other):
        return other.__sub__(self)
    @argstodual(0)
    def __rmul__(self, other):
        return other.__mul__(self)
    @argstodual(0)
    def __rtruediv__(self, other):
        return other.__truediv__(self)
    # object.__rfloordiv__(self, other)
    # object.__rmod__(self, other)
    # object.__rdivmod__(self, other)
    @argstodual(0)
    def __rpow__(self, other, *modulo):
        if modulo:
            return NotImplemented
        else:
            return exp(self*log(other))
    # object.__rlshift__(self, other)
    # object.__rrshift__(self, other)
    # object.__rand__(self, other)
    # object.__rxor__(self, other)
    # object.__ror__(self, other)

    # object.__iadd__(self, other)
    # object.__isub__(self, other)
    # object.__imul__(self, other)
    # object.__itruediv__(self, other)
    # object.__ifloordiv__(self, other)
    # object.__imod__(self, other)
    # object.__ipow__(self, other[, modulo])
    # object.__ilshift__(self, other)
    # object.__irshift__(self, other)
    # object.__iand__(self, other)
    # object.__ixor__(self, other)
    # object.__ior__(self, other)

    # object.__neg__(self)¶
    # object.__pos__(self)
    # object.__abs__(self)
    # object.__invert__(self)

    @argstodual(0)
    def __lt__(self, other):
        return self.real < other.real
    @argstodual(0)
    def __le__(self, other):
        return self.real <= other.real
    @argstodual(0)
    def __gt__(self, other):
        return self.real > other.real
    @argstodual(0)
    def __ge__(self, other):
        return self.real >= other.real

    def __str__(self):
        return "Dual({},  {})".format(self.real, self.dual)
    def __repr__(self):
        return "Dual({},  {})".format(self.real, self.dual)

