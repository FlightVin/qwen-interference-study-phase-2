"""Round 5 of RELATED candidates (2026-11-16 ~10:00), a top-up for the 8 training problems still
short after round 4 (54 related picks): omni-0528, -1391, -1689, -1814, -1900, -3926 (1 pick each)
and omni-0166, -2584 (none). Each is pitched next to the problem's in-band item, or one step harder
than its 12-16/16 items. Answers are computed here. Returns (related, [])."""

from itertools import product

import sympy as sp

from batch3 import count_pairs
from batch6 import walks
from batch9 import tan_seq
from common import L

x = sp.Symbol("x", positive=True)


def pts_at(center, d2):
    """Points of {-1,0,1}^4 at squared distance d2 from center."""
    return sum(1 for p in product((-1, 0, 1), repeat=4) if sum((a - b) ** 2 for a, b in zip(p, center)) == d2)


def count_x(value):
    """Number of x in [0, 2 pi) with cos^2 x + sin^4 x = value (= 1 - s + s^2 with s = sin^2 x)."""
    s = sp.Symbol("s")
    n = 0
    for r in sp.solve(sp.Eq(1 - s + s ** 2, value), s):
        if r.is_real and 0 <= r <= 1:
            n += 2 if r in (0, 1) else 4
    return n


def right_angle_x(a, b):
    """Positive x with angle AXB = 90 degrees for A=(a,1/a), B=(b,1/b), X=(x,1/x) on xy=1."""
    ma = (1 / x - sp.Rational(1, a)) / (x - a)
    mb = (1 / x - sp.Rational(1, b)) / (x - b)
    sols = [r for r in sp.solve(sp.Eq(sp.simplify(ma * mb), -1), x) if r.is_positive]
    assert len(sols) == 1, sols
    return sols[0]


def smallest_with_digit_sum(s):
    k = 1
    while sum(map(int, str(k))) != s:
        k += 1
    return k


def boxes(small, big):
    """Integer sides s1 <= s2 <= s3 with s1 s2 = small and s2 s3 = big."""
    return [(a, small // a, big // (small // a)) for a in range(1, small + 1)
            if small % a == 0 and small // a >= a and big % (small // a) == 0 and big // (small // a) >= small // a]


def build():
    R = []
    A = lambda h, p, a: R.append((h, p, a))

    h = "omni-0528"  # in band: 4D points at distance 2 from the origin (7/16)
    P4 = r"How many points with all coordinates in $\{-1,0,1\}$ in four-dimensional space are at distance exactly 2 from the point $%s$?"
    A(h, P4 % "(1,0,0,0)", L(pts_at((1, 0, 0, 0), 4)))
    A(h, P4 % "(1,1,0,0)", L(pts_at((1, 1, 0, 0), 4)))
    A(h, r"Let $S$ be the set of $3^{4}$ points in four-dimensional space with each coordinate in $\{-1,0,1\}$. How many sequences $P_{1}, P_{2}, P_{3}$ of points in $S$ satisfy $P_{1}=(0,0,0,0)$ and $P_{1} P_{2}=P_{2} P_{3}=P_{3} P_{1}=2$?", L(walks(4, 4, 3, True)))

    h = "omni-1391"  # in band: minimum of the product of the two shifted expressions (10/16)
    A(h, r"How many real numbers $x$ with $0 \leq x<2 \pi$ satisfy $\cos ^{2} x+\sin ^{4} x=\frac{3}{4}$?", L(count_x(sp.Rational(3, 4))))
    s = sp.Symbol("s")
    f = 1 - s + s ** 2  # cos^2 x + sin^4 x = sin^2 x + cos^4 x with s = sin^2 x
    t = sp.Symbol("t", real=True)
    assert sp.simplify((sp.cos(t) ** 2 + sp.sin(t) ** 4) - (sp.sin(t) ** 2 + sp.cos(t) ** 4)) == 0
    assert sp.simplify(sp.cos(t) ** 2 + sp.sin(t) ** 4 - f.subs(s, sp.sin(t) ** 2)) == 0
    A(h, r"Find the minimum value of $\left(\cos ^{2} x+\sin ^{4} x\right)\left(\sin ^{2} x+\cos ^{4} x\right)$ over all real $x$.", L(sp.minimum(f ** 2, s, sp.Interval(0, 1))))
    A(h, r"For real $x$ let $u=1+\cos ^{2} x+\sin ^{4} x$ and $v=1+\sin ^{2} x+\cos ^{4} x$. Find the maximum value of $|u-v|$ over all real $x$.", L(0))
    A(h, r"Find the minimum value of $\cos ^{2} x+\sin ^{4} x+\sin ^{2} x+\cos ^{4} x$ over all real $x$.", L(sp.minimum(2 * f, s, sp.Interval(0, 1))))

    h = "omni-1689"  # in band: pairs with 13n + 17k = 1000 (9/16)
    T = r"How many pairs $(n, k)$ of nonnegative integers satisfy $%d n+%d k=%d$?"
    for a, b, tot in ((17, 23, 3000), (19, 23, 5000), (23, 29, 6000)):
        A(h, T % (a, b, tot), L(count_pairs(a, b, tot)))
    A(h, r"How many pairs $(n, k)$ of positive integers satisfy $13 n+17 k=2000$?", L(count_pairs(13, 17, 2000, pos=True)))

    h = "omni-1814"  # in band: perpendicular chords through (2, 1/2) on xy = 1 (10/16)
    H = r"The points $A=\left(%d, \frac{1}{%d}\right)$ and $B=\left(-%d,-\frac{1}{%d}\right)$ lie on the hyperbola $x y=1$. The point $X=\left(x, \frac{1}{x}\right)$ with $x>0$ also lies on it, and $\angle A X B=90^{\circ}$. Find $x$."
    for a, b in ((2, 3), (2, 5), (4, 5)):
        A(h, H % (a, a, b, b), L(right_angle_x(a, -b)))

    h = "omni-1900"  # in band: a_2 = 2, find a_7 (9/16)
    Q = r"A sequence satisfies $a_{0}=0, a_{1}=1, a_{2}=%s$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n \geq 1$. Find $a_{%d}$."
    # a_8 = 1 for every a_2 (a_6 = -1/a_2, a_5 = (1 - a_2)/(1 + a_2)), so only one a_8 item
    for a2, tex, k in ((2, "2", 9), (3, "3", 8), (sp.Rational(1, 2), r"\frac{1}{2}", 7)):
        v = tan_seq(a2, k)
        assert v.is_finite, (a2, k)
        A(h, Q % (tex, k), L(v))

    h = "omni-3926"  # in band: smallest k with f(k) = 289 (8/16)
    F = r"For a positive integer $k$ let $f(k)$ be the square of the sum of the digits of $k$. Find the smallest positive integer $k$ with $f(k)=%d$."
    for sq in (361, 400, 529):
        A(h, F % sq, L(smallest_with_digit_sum(int(sq ** 0.5))))

    h = "omni-0166"  # no item in band; all earlier items 12-16/16
    D = r"Points $A, B, C, D$ in space satisfy $A C=A D=B C=B D=1$ and $A B=%s$. Find the largest possible value of $C D$."
    for ab, tex in ((sp.Rational(6, 5), r"\frac{6}{5}"), (sp.Rational(1, 2), r"\frac{1}{2}")):
        A(h, D % tex, L(2 * sp.sqrt(1 - (ab / 2) ** 2)))
    u = sp.Symbol("u", positive=True)
    best = sp.maximum(u + 2 * sp.sqrt(1 - u ** 2 / 4), u, sp.Interval(0, 2))
    A(h, r"Points $A, B, C, D$ in space satisfy $A C=A D=B C=B D=1$. Find the largest possible value of $A B+C D$.", L(best))
    ph = sp.Symbol("phi", real=True)
    Cp = sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 2, 0])
    Dp = sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 2 * sp.cos(ph), sp.sqrt(3) / 2 * sp.sin(ph)])
    vol = sp.simplify(sp.Matrix.hstack(sp.Matrix([1, 0, 0]), Cp, Dp).det() / 6)
    assert sp.simplify(vol - sp.sin(ph) / 8) == 0
    A(h, r"A tetrahedron $A B C D$ has $A B=A C=A D=B C=B D=1$. Find the largest possible volume of $A B C D$.", L(sp.maximum(vol, ph, sp.Interval(0, sp.pi))))

    h = "omni-2584"  # no item in band; explicit-structure items 12-14/16, sums over triples 1-2/16
    bx = boxes(48, 240)
    assert bx == [(4, 12, 20), (6, 8, 30)], bx
    B = r"Consider all rectangular boxes with integer side lengths $s_{1} \leq s_{2} \leq s_{3}$ such that $s_{1} s_{2}=48$ and $s_{2} s_{3}=240$. "
    A(h, B + r"Find the sum of all possible values of $s_{3}$.", L(sum(c for _, _, c in bx)))
    A(h, B + r"Find the largest possible value of $s_{1} s_{3}$.", L(max(a * c for a, _, c in bx)))
    I = r"A rectangular box has integer side lengths. Its largest face has area 240 and its smallest face has area 48. "
    A(h, I + r"How many such boxes are there (up to the order of the side lengths)?", L(len(bx)))
    A(h, I + r"Find the smallest possible area of its third face.", L(min(a * c for a, _, c in bx)))
    return R, []
