"""Round 5 of CONFLICTING candidates (2026-11-16 ~10:00), a top-up for the 11 training problems
still short after round 4 (50 conflicting picks): omni-0848, -0912, -1299, -1315, -1735, -1814,
-2215, -3926 (1 pick each) and omni-1391, -1877, -1900 (none). Same-setting lookalikes pitched next
to each problem's in-band item, or between its 12-16/16 and 0-3/16 items. Answers are computed
here. Returns ([], conflicting)."""

from fractions import Fraction
from itertools import combinations, product

import mpmath as mp
import sympy as sp

from batch2 import start_money
from batch4 import digit_iter, sq_digit_sum, sum_sq_digits, tan_recursion, trig_case
from batch6 import rec_seq
from common import L, frac

x = sp.Symbol("x")


def hexagon(side):
    return [(side * sp.cos(sp.pi * k / 3), side * sp.sin(sp.pi * k / 3)) for k in range(6)]  # A..F


def area(pts):
    n = len(pts)
    return sp.nsimplify(sp.simplify(abs(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                                            for i in range(n))) / 2))


def rolls(n, at_least=None, exactly=None):
    """Roll an n-sided die until a value <= the previous one. P(at least m rolls) = P(first m-1 rolls
    strictly increase); P(exactly m) = P(at least m) - P(at least m+1). Enumerated exactly."""
    def inc(m):
        return Fraction(sum(1 for r in product(range(1, n + 1), repeat=m) if all(a < b for a, b in zip(r, r[1:]))), n ** m)
    if at_least:
        return inc(at_least - 1)
    return inc(exactly - 1) - inc(exactly)


def monic(deg, conds, at):
    cs = sp.symbols(f"c0:{deg}")
    P = x ** deg + sum(c * x ** i for i, c in enumerate(cs))
    sol = sp.solve([sp.Eq(P.subs(x, a), v) for a, v in conds], cs, dict=True)
    assert len(sol) == 1
    return P.subs(sol[0]).subs(x, at)


def tangent_height(a, c, r):
    """A=(-a,0), B=(0,0), C=(c,0); tangents from A and C to the circle (B, r) on the upper side meet
    at K. Returns the height of K above line AC."""
    a, c, r = map(sp.Integer, (a, c, r))
    mA, mC = r / sp.sqrt(a * a - r * r), r / sp.sqrt(c * c - r * r)
    kx = sp.solve(sp.Eq(mA * (x + a), -mC * (x - c)), x)[0]
    return sp.nsimplify(sp.simplify(mA * (kx + a)))


def build():
    C = []
    A = lambda h, p, a: C.append((h, p, a))

    h = "omni-0848"  # in band: regular hexagon, area of quadrilateral ABCD (9/16)
    H2, H3 = hexagon(2), hexagon(3)
    A(h, r"A regular hexagon $A B C D E F$ has side length 2. Find the area of triangle $A B D$.", L(area([H2[0], H2[1], H2[3]])))
    A(h, r"A regular hexagon $A B C D E F$ has side length 3. Find the area of quadrilateral $A C D F$.", L(area([H3[0], H3[2], H3[3], H3[5]])))
    A(h, r"A regular hexagon $A B C D E F$ has side length 2. Find the area of pentagon $A B C D E$.", L(area(H2[:5])))

    h = "omni-0912"  # in band: pairs of subsets of {1,2} with neither contained in the other (8/16)
    S3 = [frozenset(s) for k in range(4) for s in combinations([1, 2, 3], k)]
    neither = [(X, Y) for X, Y in product(S3, repeat=2) if not X <= Y and not Y <= X]
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2,3\}$ are there such that neither $X \subseteq Y$ nor $Y \subseteq X$?", L(len(neither)))
    A(h, r"How many unordered pairs $\{X, Y\}$ of subsets of $\{1,2,3\}$ are there such that neither $X \subseteq Y$ nor $Y \subseteq X$?", L(len({frozenset(p) for p in neither})))
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2,3\}$ satisfy $X \cap Y=\varnothing$ and $X \cup Y \neq\{1,2,3\}$?", L(sum(1 for X, Y in product(S3, repeat=2) if not X & Y and X | Y != frozenset([1, 2, 3]))))

    h = "omni-1299"  # in band: boundaries of two convex quadrilaterals (7/16). A segment meets a convex
    # boundary at most twice, so the maximum is 2 * min(sides); a rotated concentric pair attains it.
    M = r"What is the maximum number of points in which the boundary of %s and the boundary of %s can intersect, if no two sides overlap?"
    A(h, M % ("a triangle", "a convex quadrilateral"), L(2 * min(3, 4)))
    A(h, M % ("a convex pentagon", "another convex pentagon"), L(2 * min(5, 5)))
    A(h, M % ("a convex quadrilateral", "a convex pentagon"), L(2 * min(4, 5)))

    h = "omni-1315"  # in band: 3 shops, lottery doubles, spend 1000 (8/16)
    S = r"Stacy has $d$ dollars. She enters a mall with %d shops and a lottery stall. First she plays the lottery and her money is %s, then she goes into a shop and spends %d dollars. She repeats this until she has been to all %d shops. When she comes out of the last shop she has no money left. What is the minimum possible value of $d$?"
    for n, word, mult, spend in ((4, "doubled", 2, 1600), (3, "tripled", 3, 2700), (5, "doubled", 2, 3200)):
        A(h, S % (n, word, spend, n), frac(start_money(n, spend, mult=mult)))

    h = "omni-1735"  # in band: 5-sided die, probability of exactly 2 rolls (8/16)
    R = r"We keep rolling a fair %d-sided die (faces 1 to %d) until we get a value less than or equal to the previous roll. What is the probability that %s?"
    A(h, R % (5, 5, "at least 3 rolls are made"), frac(rolls(5, at_least=3)))
    A(h, R % (6, 6, "at least 3 rolls are made"), frac(rolls(6, at_least=3)))
    A(h, R % (5, 5, "exactly 3 rolls are made"), frac(rolls(5, exactly=3)))

    h = "omni-1814"  # in band: sum of squares of the x-coordinates, x^2+y^2=7 and xy=1 (9/16)
    def xs(r2, p):  # x-coordinates of the circle x^2+y^2=r2 meeting xy=p: x^4 - r2 x^2 + p^2 = 0
        roots = sp.solve(x ** 4 - r2 * x ** 2 + p ** 2, x)
        assert len(roots) == 4 and all(r.is_real for r in roots)
        return roots
    A(h, r"The circle $x^{2}+y^{2}=7$ meets the hyperbola $x y=1$ at four points. Find the sum of the fourth powers of their four $x$-coordinates.", L(sp.simplify(sum(r ** 4 for r in xs(7, 1)))))
    A(h, r"The circle $x^{2}+y^{2}=8$ meets the hyperbola $x y=2$ at four points. Find the largest $x$-coordinate among them.", L(max(xs(8, 2), key=lambda r: float(r))))
    A(h, r"The circle $x^{2}+y^{2}=6$ meets the hyperbola $x y=1$ at four points. Find the sum of the squares of their four $x$-coordinates.", L(sp.simplify(sum(r ** 2 for r in xs(6, 1)))))

    h = "omni-2215"  # in band: monic quartic with P(1..3) = 2, 3, 4 and P(4) = 6; find P(0) (8/16)
    A(h, r"Let $P(x)=x^{4}+a x^{3}+b x^{2}+c x+d$ with $P(1)=2, P(2)=3, P(3)=4$, and $P(4)=12$. Find $P(0)$.", L(monic(4, [(1, 2), (2, 3), (3, 4), (4, 12)], 0)))
    A(h, r"Let $P(x)=x^{4}+a x^{3}+b x^{2}+c x+d$ with $P(2)=3, P(3)=4, P(4)=5$, and $P(5)=7$. Find $P(1)$.", L(monic(4, [(2, 3), (3, 4), (4, 5), (5, 7)], 1)))
    A(h, r"Let $P(x)=x^{3}+a x^{2}+b x+c$ with $P(1)=2, P(2)=3$, and $P(3)=5$. Find $P(4)$.", L(monic(3, [(1, 2), (2, 3), (3, 5)], 4)))

    h = "omni-3926"  # in band: a square-of-digit-sum iteration ending at 81 (5/16); g items 13-16/16
    G = r"For a positive integer $k$ let $g(k)$ be the sum of the squares of the digits of $k$, and let $g_{1}=g$, $g_{n+1}(k)=g\left(g_{n}(k)\right)$. "
    A(h, G + r"Compute $g_{100}(4)$.", L(digit_iter(4, 100, sum_sq_digits)))
    A(h, G + r"Compute $g_{2000}(2)$.", L(digit_iter(2, 2000, sum_sq_digits)))
    A(h, r"For a positive integer $k$ let $f(k)$ be the square of the sum of the digits of $k$. Compute $f\left(f\left(f\left(5^{10}\right)\right)\right)$.", L(digit_iter(5 ** 10, 3, sq_digit_sum)))

    h = "omni-1391"  # nothing in band: cos questions with shift 1 scored 0/16 (sign), sin/cos with no sign issue 13-15/16
    mp.mp.dps = 40
    cases = [(r"Suppose $x$ is a real number such that $\sin \left(1+\cos ^{2} x+\sin ^{4} x\right)=\frac{24}{25}$. Compute $\cos ^{2}\left(1+\sin ^{2} x+\cos ^{4} x\right)$.", 1, mp.sin, lambda u: mp.cos(u) ** 2, mp.mpf(24) / 25, sp.Rational(49, 625)),
             (r"Suppose $x$ is a real number such that $\sin \left(1+\cos ^{2} x+\sin ^{4} x\right)=\frac{19}{20}$. Compute $\tan ^{2}\left(1+\sin ^{2} x+\cos ^{4} x\right)$.", 1, mp.sin, lambda u: mp.tan(u) ** 2, mp.mpf(19) / 20, sp.Rational(361, 39)),
             (r"Suppose $x$ is a real number such that $\cos \left(2+\cos ^{2} x+\sin ^{4} x\right)=-\frac{24}{25}$. Compute $\tan \left(2+\sin ^{2} x+\cos ^{4} x\right)$.", 2, mp.cos, mp.tan, -mp.mpf(24) / 25, sp.Rational(-7, 24))]
    for text, s, fw, tg, val, ans in cases:
        assert abs(trig_case(s, fw, tg, val) - mp.mpf(sp.N(ans, 40))) < 1e-25, text
        A(h, text, L(ans))

    h = "omni-1877"  # nothing in band: the full XY problems scored 0-3/16, one-tangent lengths 16/16
    T = r"Let $A, B, C$ be points in that order along a line with $A B=%d$ and $B C=%d$. Let $\omega$ be the circle of radius $%d$ centered at $B$, and let $\ell_{1}$, $\ell_{2}$ be the tangents to $\omega$ through $A$ and $C$ on the same side of line $A C$, meeting at $K$. Find the distance from $K$ to line $A C$."
    for a, c, r in ((20, 15, 12), (13, 13, 5)):
        A(h, T % (a, c, r), L(tangent_height(a, c, r)))
    ak = sp.sqrt(20 ** 2 + (20 * sp.Rational(12, 16)) ** 2)  # tan(angle KAB) = 12 / sqrt(20^2 - 12^2)
    A(h, r"A circle of radius 12 has center $B$, and point $A$ satisfies $A B=20$. A tangent from $A$ to the circle meets the line through $B$ perpendicular to $A B$ at $K$. Find $A K$.", L(ak))

    h = "omni-1900"  # nothing in band: "cannot be extended" items 0-1/16, explicit terms 12-16/16
    E = r"A sequence of real numbers $a_{0}, a_{1}, \ldots, a_{%d}$ with $a_{0}=0, a_{1}=1$, and $a_{2}>0$ satisfies $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for all $1 \leq n \leq %d$, but cannot be extended to $a_{%d}$: no real $a_{%d}$ satisfies $a_{%d} a_{%d} a_{%d}=a_{%d}+a_{%d}+a_{%d}$. Compute the smallest possible value of $a_{2}$."
    for k in (4,):  # tan_recursion has a solution only for k = 4 and k >= 9 (k = 9, 10 used before)
        v = tan_recursion(k)
        assert v is not None, k
        A(h, E % (k - 1, k - 3, k, k, k, k - 2, k - 3, k, k - 2, k - 3), L(v))
    Rr = r"A sequence satisfies $a_{0}=0, a_{1}=%d, a_{2}=%d$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n \geq 1$. Find $a_{%d}$."
    for a1, a2, k in ((2, 1, 8), (3, 2, 7), (2, 3, 7)):
        v = rec_seq(a1, a2, k)
        assert v.is_finite, (a1, a2, k)
        A(h, Rr % (a1, a2, k), L(v))
    return [], C
