"""Answer formatting shared by the companion batches: every answer is a LaTeX string from a computed value."""

from fractions import Fraction

import sympy as sp


def L(x) -> str:
    """LaTeX for a computed value (int, Fraction, or SymPy expression), simplified."""
    if isinstance(x, Fraction):
        return frac(x)
    return sp.latex(sp.nsimplify(sp.simplify(x)))


def frac(f: Fraction) -> str:
    f = Fraction(f)
    if f.denominator == 1:
        return str(f.numerator)
    return ("-" if f < 0 else "") + f"\\frac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def linrec(a0, a1, p, q, n):  # a_i = p a_{i-1} + q a_{i-2}
    a = [a0, a1]
    while len(a) <= n:
        a.append(p * a[-1] + q * a[-2])
    return a[n]
