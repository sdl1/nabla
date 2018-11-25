import functools
import numbers
import math

#def argstodual(e):
#    def decorator(func):
#        @functools.wraps(func)
#        def wrapper(*args, **kwargs):
#            args = [Dual(arg,e) if isinstance(arg, numbers.Number) else arg for arg in args]
#            for key in kwargs:
#                if isinstance(kwargs[key], numbers.Number):
#                    kwargs[key] = Dual(kwargs[key], e)
#            return func(*args, **kwargs)
#        return wrapper
#    return decorator

def othertodual(e):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, other):
            if isinstance(other, numbers.Number):
                other = Dual(real=other, nvars=self.nvars)
            return func(self, other)
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
            if varpos==None:
                # Full gradient case
                # Count numerical arguments
                nvars = len([a for a in args if isinstance(a, numbers.Number)])
                nvars += len([key for key in kwargs if isinstance(kwargs[key], numbers.Number)])
                #dual = [1]*nvars
                args = [Dual(arg, nvars=nvars) if isinstance(arg, numbers.Number) else arg for arg in args]
                for key in kwargs:
                    if isinstance(kwargs[key], numbers.Number):
                        kwargs[key] = Dual(kwargs[key], nvars=nvars)
                # Now set dual[i]=1 in argument[i]
                i=0
                for arg in args:
                    if isinstance(arg, Dual):
                        arg.dual[i] = 1
                        i += 1
                for key in kwargs:
                    if isinstance(kwargs[key], Dual):
                        kwargs[key].dual[i] = 1
                        i += 1
            else:
                nvars = len(varpos)
                if kwargs:
                    raise Exception("Keyword arguments only supported for full gradient.")
                newargs = [Dual(arg, nvars=nvars) if isinstance(arg, numbers.Number) else arg for arg in args]
                # Replace the chosen vars in varpos with dual[i]=1
                for i in range(nvars):
                    newargs[varpos[i]].dual[i] = 1
                args = newargs
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
        varpos = args[0]
        if isinstance(varpos, numbers.Number):
            varpos = [varpos]
    return decorator

# f(a + be) = f(a) + b fprime(a) e

def exp(x):
    if isinstance(x, Dual):
        expa = math.exp(x.real)
        ret = Dual(expa, nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = x.dual[i]*expa
        return ret
    else:
        return math.exp(x)

def log(x):
    if isinstance(x, Dual):
        ret = Dual(math.log(x.real), nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = x.dual[i] / x.real
        return ret
    else:
        return math.log(x)

def sin(x):
    if isinstance(x, Dual):
        ret = Dual(math.sin(x.real), nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = x.dual[i] * math.cos(x.real)
        return ret
    else:
        return math.sin(x)

def cos(x):
    if isinstance(x, Dual):
        ret = Dual(math.cos(x.real), nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = -x.dual[i] * math.sin(x.real)
        return ret
    else:
        return math.cos(x)

def sqrt(x):
    if isinstance(x, Dual):
        sqrta = math.sqrt(x.real)
        ret = Dual(sqrta, nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = x.dual[i] * 0.5/sqrta
        return ret
    else:
        return math.sqrt(x)

class Dual:
    def __init__(self, real=0, dual=None, nvars=None):
        self.real = real
        if isinstance(dual, numbers.Number):
            dual = [dual]
        if dual==None and nvars==None:
            self.dual = [0]
            self.nvars = 1
        elif nvars==None:
            self.dual = dual
            self.nvars = len(dual)
        elif dual==None:
            self.dual = [0]*nvars
            self.nvars = nvars
        else:
            self.dual = dual
            self.nvars = nvars
    @othertodual(0)
    def __add__(self, other):
        ret = Dual(self. real + other.real, nvars=self.nvars)
        for i in range(self.nvars):
            ret.dual[i] = self.dual[i] + other.dual[i]
        return ret
    @othertodual(0)
    def __sub__(self, other):
        ret = Dual(self. real - other.real, nvars=self.nvars)
        for i in range(self.nvars):
            ret.dual[i] = self.dual[i] - other.dual[i]
        return ret
    @othertodual(0)
    def __mul__(self, other):
        ret = Dual(self. real * other.real, nvars=self.nvars)
        for i in range(self.nvars):
            ret.dual[i] = self.real*other.dual[i] + self.dual[i]*other.real
        return ret
    @othertodual(0)
    def __truediv__(self, other):
        ret = Dual(self. real / other.real, nvars=self.nvars)
        for i in range(self.nvars):
            ret.dual[i] = (self.dual[i]*other.real - self.real*other.dual[i])/(other.real*other.real)
        return ret
    # object.__floordiv__(self, other)
    # object.__mod__(self, other)
    # object.__divmod__(self, other)
    @othertodual(0)
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

    @othertodual(0)
    def __radd__(self, other):
        return other.__add__(self)
    @othertodual(0)
    def __rsub__(self, other):
        return other.__sub__(self)
    @othertodual(0)
    def __rmul__(self, other):
        return other.__mul__(self)
    @othertodual(0)
    def __rtruediv__(self, other):
        return other.__truediv__(self)
    # object.__rfloordiv__(self, other)
    # object.__rmod__(self, other)
    # object.__rdivmod__(self, other)
    @othertodual(0)
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

    @othertodual(0)
    def __lt__(self, other):
        return self.real < other.real
    @othertodual(0)
    def __le__(self, other):
        return self.real <= other.real
    @othertodual(0)
    def __gt__(self, other):
        return self.real > other.real
    @othertodual(0)
    def __ge__(self, other):
        return self.real >= other.real

    def __str__(self):
        return "Dual({},  {})".format(self.real, self.dual)
    def __repr__(self):
        return self.__str__()

