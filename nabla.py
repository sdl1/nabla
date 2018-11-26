import functools
import numbers
import numpy as np

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
            if varpos is None:
                # Full gradient case
                # Count numerical arguments
                nvars = len([a for a in args if isinstance(a, numbers.Number)])
                nvars += len([key for key in kwargs if isinstance(kwargs[key], numbers.Number)])
                # Count numpy args
                nvars += sum([a.size for a in args if isinstance(a, np.ndarray)])
                nvars += sum([kwargs[key].size for key in kwargs if isinstance(kwargs[key], np.ndarray)])
                newargs = [arg for arg in args]
                i=0
                for k,arg in enumerate(args):
                    if isinstance(arg, numbers.Number):
                        newargs[k] = Dual(arg, nvars=nvars, seedvar=i)
                        i += 1
                    elif isinstance(arg, np.ndarray):
                        # Each element is its own variable
                        numpytodual = np.vectorize(lambda x : Dual(x, nvars=nvars))
                        newargs[k] = numpytodual(arg)
                        flat_iter = newargs[k].flat
                        for elt in range(newargs[k].size):
                            flat_iter[elt].dual[i] = 1
                            i += 1
                for key in kwargs:
                    if isinstance(kwargs[key], numbers.Number):
                        kwargs[key] = Dual(kwargs[key], nvars=nvars, seedvar=i)
                        i += 1
                    elif isinstance(kwargs[key], np.ndarray):
                        numpytodual = np.vectorize(lambda x : Dual(x, nvars=nvars))
                        kwargs[key] = numpytodual(kwargs[key])
                        flat_iter = kwargs[key].flat
                        for elt in range(kwargs[key].size):
                            flat_iter[elt].dual[i] = 1
                            i += 1
                args = newargs
            else:
                #nvars = len(varpos)
                nvars = 0
                for i in varpos:
                    if isinstance(args[i], np.ndarray):
                        nvars += args[i].size
                    else:
                        nvars += 1
                if kwargs:
                    raise Exception("Keyword arguments only supported for full gradient.")
                newargs = [arg for arg in args]
                # Replace the chosen vars in varpos with dual[i]=1
                i = 0
                for k in varpos:
                    if isinstance(args[k], np.ndarray):
                        #numpytodual = np.vectorize(lambda x : Dual(x, nvars=nvars, seedvar=i))
                        #newargs[k] = numpytodual(args[k])
                        # Each element is its own variable
                        numpytodual = np.vectorize(lambda x : Dual(x, nvars=nvars))
                        newargs[k] = numpytodual(args[k])
                        flat_iter = newargs[k].flat
                        for elt in range(newargs[k].size):
                            flat_iter[elt].dual[i] = 1
                            i += 1
                    else:
                        newargs[k] = Dual(args[k], nvars=nvars, seedvar=i)
                        i += 1
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


def dot(x, y):
    ret = Dual(0.0)
    for i in range(len(x)):
        ret += x[i]*y[i]
    return ret

class Dual:
    def __init__(self, real=0, dual=None, nvars=None, seedvar=None):
        self.real = real
        if dual is None and nvars is None:
            self.dual = np.zeros(1)
            self.nvars = 1
        elif nvars is None:
            self.dual = np.array(dual, dtype=np.float64, ndmin=1)
            self.nvars = len(self.dual)
        elif dual is None:
            self.dual = np.zeros(nvars)
            self.nvars = nvars
        else:
            self.dual = np.array(dual, dtype=np.float64, ndmin=1)
            self.nvars = nvars
        if seedvar is not None:
            self.dual[seedvar] = 1
    def __add__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        ret = Dual(self.real + otherreal, self.dual)
        if otherdual is not None:
            for i in range(self.nvars):
                ret.dual[i] += otherdual[i]
        return ret
    def __sub__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        ret = Dual(self. real - otherreal, self.dual)
        if otherdual is not None:
            for i in range(self.nvars):
                ret.dual[i] -= otherdual[i]
        return ret
    def __mul__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        ret = Dual(self.real * otherreal, nvars=self.nvars)
        for i in range(self.nvars):
            ret.dual[i] = self.dual[i]*otherreal
        if otherdual is not None:
            for i in range(self.nvars):
                ret.dual[i] += self.real*otherdual[i]
        return ret
    def __truediv__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        #ret.dual[i] = self.dual[i]/other.real - self.real*other.dual[i])/(other.real*other.real)
        ret = Dual(self. real / otherreal, self.dual / otherreal)
        if otherdual is not None:
            for i in range(self.nvars):
                ret.dual[i] -= self.real*otherdual[i]/(otherreal*otherreal)
        return ret
    # object.__floordiv__(self, other)
    # object.__mod__(self, other)
    # object.__divmod__(self, other)
    def __pow__(self, other, *modulo):
        if modulo:
            return NotImplemented
        if isinstance(other, int):
            m = other
            negative = (m<0)
            m = abs(m)
            ret = Dual(1, nvars=self.nvars)
            for _ in range(m):
                ret *= self
            if negative:
                ret = 1/ret
            return ret
        else:
            return Dual.exp(other*Dual.log(self))
    # object.__lshift__(self, other)
    # object.__rshift__(self, other)
    # object.__and__(self, other)
    # object.__xor__(self, other)
    # object.__or__(self, other)

    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return -self.__sub__(other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __rtruediv__(self, other):
        return Dual(1,nvars=self.nvars)/self.__truediv__(other)
    # object.__rfloordiv__(self, other)
    # object.__rmod__(self, other)
    # object.__rdivmod__(self, other)
    def __rpow__(self, other, *modulo):
        if modulo:
            return NotImplemented
        x = Dual(other, nvars=self.nvars)
        return x.__pow__(self)
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

    def __neg__(self):
        return Dual(-self.real, -self.dual)
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

    # f(a + be) = f(a) + b fprime(a) e

    def exp(x):
        expa = np.exp(x.real)
        ret = Dual(expa, nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = x.dual[i]*expa
        return ret
    
    def log(x):
        ret = Dual(np.log(x.real), nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = x.dual[i] / x.real
        return ret
    
    def sin(x):
        ret = Dual(np.sin(x.real), nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = x.dual[i] * np.cos(x.real)
        return ret
    
    def cos(x):
        ret = Dual(np.cos(x.real), nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = -x.dual[i] * np.sin(x.real)
        return ret
    
    def sqrt(x):
        sqrta = np.sqrt(x.real)
        ret = Dual(sqrta, nvars=x.nvars)
        for i in range(x.nvars):
            ret.dual[i] = x.dual[i] * 0.5/sqrta
        return ret

def minimise(f, x0, alpha = 1e-1, maxits = 1000, tolerance=1e-6, variables=None):
    fgrad = grad(variables)(f)
    x0 = np.array(x0)
    zerostep = np.zeros(x0.shape)
    tolerance_sq = tolerance**2
    converged = False
    for it in range(maxits):
        gradient = fgrad(*x0).dual
        if variables is None:
            step = alpha*gradient
        else:
            # Only step in specified variables
            step = zerostep
            step[variables] = alpha*gradient
        x0 = x0 - step
        if np.dot(step, step)<tolerance_sq:
            converged = True
            break
    if not converged:
        print("Warning: didn't converge in {} iterations".format(maxits))
    result = fgrad(*x0)
    return x0, result.real, result.dual


