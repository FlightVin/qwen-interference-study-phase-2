"""Round 2 of CONFLICTING candidates (2026-11-16), calibrated on round-1 screening.

Round 1 had 21/128 conflicting candidates in band (78 below, 29 above). For each training problem
with fewer than 2 in band this batch adds lookalikes in the same setting, pitched between round 1's
16/16 and 0/16 items: easier where round 1 was hard (smaller cases, fewer steps), harder where it
was easy. They stay genuine: real problems with correct answers, solved by a method (small-case
computation, a changed rule, a shortcut that is exact here) that does not solve the hard problem.
Answers are computed here. Returns ([], conflicting).
"""

from fractions import Fraction
from itertools import product
from math import ceil, floor, gcd

import mpmath as mp
import sympy as sp
from sympy import isprime

from batch2 import king, placements, queen, rook
from batch3 import count_funcs, count_pairs, pisano, prism_third_faces, solve_poly
from batch4 import cafe, digit_iter, is_sq, matchings, max_ap_polygon, sq_digit_sum, sum_sq_digits, trig_case
from common import L, frac

x = sp.Symbol("x")


def phi_k(n, k):
    return sum(1 for m in range(1, n // k + 1) if gcd(m, n) == 1)


def walks(dim, step_sq, length, closed):
    pts = list(product((-1, 0, 1), repeat=dim))
    nbr = {p: [q for q in pts if sum((a - b) ** 2 for a, b in zip(p, q)) == step_sq] for p in pts}
    start = tuple([0] * dim)
    cnt = {start: 1}
    for _ in range(length):
        nxt = {}
        for p, c in cnt.items():
            for q in nbr[p]:
                nxt[q] = nxt.get(q, 0) + c
        cnt = nxt
    return cnt.get(start, 0) if closed else sum(cnt.values())


def tangent_xy(a, c, r):
    a, c, r = map(sp.Integer, (a, c, r))
    mA, mC = r / sp.sqrt(a * a - r * r), r / sp.sqrt(c * c - r * r)
    return sp.simplify((c - r / mC) - (-a + r / mA))


def first_m(b, window):
    pw = {b ** e for e in range(60)}
    m = 1
    while any(t in pw for t in range(m, m + window + 1)):
        m += 1
    return m


def circle_hyperbola_rect_area(r2):
    """x^2+y^2=r2 meets xy=1 in four points forming a rectangle; its area."""
    xs = sorted([s for s in sp.solve(sp.Eq(x ** 2 + 1 / x ** 2, r2), x) if s.is_real], key=lambda s: float(s))
    pts = [(s, 1 / s) for s in xs]
    p0 = max(pts, key=lambda p: float(p[0]))
    # neighbours of p0 on the rectangle: its mirror (1/x, x) and (-1/x, -x)
    q1, q2 = (p0[1], p0[0]), (-p0[1], -p0[0])
    d = lambda u, v: sp.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)
    return sp.simplify(d(p0, q1) * d(p0, q2))


def reg_poly_area(n, R):  # regular n-gon inscribed in a circle of radius R
    return sp.nsimplify(sp.simplify(sp.Rational(n, 2) * R ** 2 * sp.sin(2 * sp.pi / n)))


def rec_seq(a1, a2, k):
    s = [sp.Integer(0), sp.Integer(a1), sp.Integer(a2)]
    for n in range(1, k - 1):
        s.append(sp.simplify((s[n] + s[n - 1]) / (s[n] * s[n - 1] - 1)))
    return s[k]


def build():
    C = []
    A = lambda h, p, a: C.append((h, p, a))
    D0 = r"Define $\varphi^{k}(n)$ as the number of positive integers that are at most $n / k$ and relatively prime to $n$. "
    # T0 omni-0793 (0,15,14,13) — larger n where the proportional count phi(n)/k is still exact
    h = "omni-0793"
    for n, k in ((360, 4), (2940, 7), (1350, 3), (3000, 5)):
        A(h, D0 + rf"Compute $\varphi^{{{k}}}({n})$.", L(phi_k(n, k)))
    # T1 omni-1735 (0,0,11,4) — small-die probabilities
    h = "omni-1735"
    A(h, r"A fair 6-sided die is rolled twice. What is the probability that the second roll is less than or equal to the first?", frac(Fraction(21, 36)))
    A(h, r"We keep rolling a fair 3-sided die (faces 1, 2, 3) until we get a value less than or equal to the previous roll. What is the probability that at least 3 rolls are made?", frac(Fraction(3, 9)))
    A(h, r"We keep rolling a fair 5-sided die (faces 1 to 5) until we get a value less than or equal to the previous roll. What is the probability that exactly 2 rolls are made?", frac(1 - Fraction(10, 25)))
    # T2 omni-1814 (all 16) — harder circle/hyperbola lookalikes
    h = "omni-1814"
    for r2 in (6, 10):
        A(h, rf"The circle $x^{{2}}+y^{{2}}={r2}$ meets the hyperbola $x y=1$ at four points, which form a rectangle. Find the area of this rectangle.", L(circle_hyperbola_rect_area(r2)))
    A(h, r"The circle $x^{2}+y^{2}=7$ meets the hyperbola $x y=1$ at four points. Find the sum of the squares of their four $x$-coordinates.", L(sum(s ** 2 for s in sp.solve(sp.Eq(x ** 2 + 1 / x ** 2, 7), x))))
    # T3 omni-2140 (16,15,5,14) — regular polygons from roots of unity, slightly harder
    h = "omni-2140"
    A(h, r"The eight roots of $z^{8}=1$ form a regular octagon in the complex plane. Find its area.", L(reg_poly_area(8, 1)))
    A(h, r"The four roots of $z^{4}+4=0$ form a square in the complex plane. Find its area.", L(reg_poly_area(4, sp.sqrt(2))))
    A(h, r"The twelve roots of $z^{12}=1$ form a regular 12-gon in the complex plane. Find its area.", L(reg_poly_area(12, 1)))
    # T4 omni-2430 (16,10,1,13)
    h = "omni-2430"
    for b, w in ((3, 50), (6, 30), (2, 40)):
        A(h, rf"Find the smallest positive integer $m$ such that none of $\log _{{{b}}}(m), \log _{{{b}}}(m+1), \ldots, \log _{{{b}}}(m+{w})$ is an integer.", L(first_m(b, w)))
    # T6 omni-0528 (0,0,0,0) — low-dimensional versions of the walk count
    h = "omni-0528"
    A(h, r"Let $S$ be the set of 9 points in the plane with both coordinates in $\{-1,0,1\}$. How many sequences $P_{1}, P_{2}, P_{3}$ of points in $S$ have $P_{1}=(0,0)$ and $P_{1} P_{2}=P_{2} P_{3}=\sqrt{2}$?", L(walks(2, 2, 2, False)))
    A(h, r"Let $S$ be the set of 27 points in space with every coordinate in $\{-1,0,1\}$. How many sequences $P_{1}, P_{2}, P_{3}$ of points in $S$ have $P_{1}=(0,0,0)$ and $P_{1} P_{2}=P_{2} P_{3}=\sqrt{3}$?", L(walks(3, 3, 2, False)))
    A(h, r"Let $S$ be the set of 27 points in space with every coordinate in $\{-1,0,1\}$. How many sequences $P_{1}, P_{2}, P_{3}, P_{4}$ of points in $S$ have $P_{1}=(0,0,0)$, $P_{i} P_{i+1}=\sqrt{3}$ for $1 \leq i \leq 4$, where $P_{5}=P_{1}$?", L(walks(3, 3, 4, True)))
    # T7 omni-0596 (all 16) — harder questions on the 6-8-10 triangle
    h = "omni-0596"
    A(h, r"Triangle $A B C$ has vertices $A=(0,6), B=(0,0), C=(8,0)$. Find the area of the set of points $P$ inside the triangle whose distance to line $A B$ is at most their distance to line $B C$.", frac(Fraction(1, 2) * 6 * Fraction(24, 7)))
    A(h, r"Triangle $A B C$ has $A B=6, B C=8, A C=10$. Find the distance from the centroid of the triangle to side $A C$.", frac(Fraction(24 - 3 * Fraction(8, 3) - 4 * 2, 5)))
    # T8 omni-1877 (5,1,1,1) — symmetric cases with a fixed radius
    h = "omni-1877"
    T8 = r"Let $A, B, C$ be points in that order along a line with $A B={a}$ and $B C={c}$. Let $\omega$ be the circle of radius ${r}$ centered at $B$, and let $\ell_{{1}}$, $\ell_{{2}}$ be the tangents to $\omega$ through $A$ and $C$ on the same side of line $A C$, meeting at $K$. Let $X$ on segment $K A$ and $Y$ on segment $K C$ be such that $X Y \| A C$ and $X Y$ is tangent to $\omega$. Find $X Y$."
    for a, c, r in ((13, 13, 12), (17, 17, 15)):
        A(h, T8.format(a=a, c=c, r=r), L(tangent_xy(a, c, r)))
    # T9 omni-0757 (16,3,0,0)
    h = "omni-0757"
    T9 = r"Let $n$ be the maximum number of {p}s that can be placed on a ${s} \times {s}$ chessboard so that no two attack each other, and let $k$ be the number of ways to place $n$ such {p}s. Find $n+k$."
    for p, s, f in (("queen", 4, queen), ("king", 3, king), ("rook", 7, rook)):
        A(h, T9.format(p=p, s=s), L(sum(placements(s, s, f))))
    # T11 omni-0848 (0,1,0,1) — the regular case, plainly asked
    h = "omni-0848"
    A(h, r"Segments $A D, B E, C F$ share a common midpoint, each pair forms a $60^{\circ}$ angle, and $A D=B E=C F=2$, with $A, B, C, D, E, F$ in order around the midpoint. Find the area of hexagon $A B C D E F$.", L(reg_poly_area(6, 1)))
    A(h, r"A regular hexagon has side length 2. Find its area.", L(reg_poly_area(6, 2)))
    A(h, r"A regular hexagon $A B C D E F$ has side length 1. Find the area of triangle $A C E$.", L(sp.sqrt(3) / 4 * 3))
    # T12 omni-1590 (2,1,4,1) — floor equations over a short range
    h = "omni-1590"
    A(h, r"How many integers $n$ with $0 \leq n \leq 20$ satisfy $\left\lfloor\frac{n}{2}\right\rfloor=\left\lfloor\frac{n}{3}\right\rfloor$?", L(sum(1 for n in range(21) if n // 2 == n // 3)))
    A(h, r"How many integers $n$ with $0 \leq n \leq 30$ satisfy $\left\lfloor\frac{n}{4}\right\rfloor=\left\lfloor\frac{n}{5}\right\rfloor$?", L(sum(1 for n in range(31) if n // 4 == n // 5)))
    A(h, r"How many integers $n$ with $1 \leq n \leq 12$ satisfy $\left\lceil\frac{n}{2}\right\rceil=\left\lceil\frac{n}{3}\right\rceil$?", L(sum(1 for n in range(1, 13) if ceil(Fraction(n, 2)) == ceil(Fraction(n, 3)))))
    # T13 omni-0912 (1,0,0,0) — the one-element version and plain antichain counts
    h = "omni-0912"
    A(h, r"Define $P=\{\mathrm{S}\}$ and let $\mathcal{P}$ be the set of all subsets of $P$. How many ordered pairs $(\mathcal{S}, \mathcal{T})$ of subsets of $\mathcal{P}$ are there such that $\mathcal{S}$ is not a proper subset of $\mathcal{T}$ and $\mathcal{T}$ is not a proper subset of $\mathcal{S}$?", L(sum(1 for a, b in product(range(4), repeat=2) if not ((a & b) == a and a != b) and not ((a & b) == b and a != b))))  # subsets of P as 2-bit masks
    A(h, r"How many subsets of $\{\varnothing,\{1\},\{2\},\{1,2\}\}$ contain no two sets such that one is a proper subset of the other?", L(sum(1 for F in range(16) if not any(((F >> i) & 1) and ((F >> j) & 1) and (i & j) == i and i != j for i in range(4) for j in range(4)))))
    # T17 omni-2584 (16,1,16,1) — smaller face areas
    h = "omni-2584"
    T17 = r"Let $V$ be a rectangular prism with integer side lengths. The largest face has area {b} and the smallest face has area {s}. A third face has area $x$, where $x$ is not equal to {s} or {b}. What is the sum of all possible values of $x$?"
    for b, s in ((30, 6), (24, 4)):
        v = prism_third_faces(b, s)
        if v:
            A(h, T17.format(b=b, s=s), L(sum(v)))
    # T18 omni-2215 (0,10,1,0) — small systems
    h = "omni-2215"
    a, b, c = sp.symbols("a b c")
    A(h, r"Let $P(x)=x^{3}+a x^{2}+b x+c$ with $P(1)=1, P(2)=3$, and $P(3)=6$. Find $P(0)$.", L(solve_poly(x ** 3 + a * x ** 2 + b * x + c, [a, b, c], [(1, 1), (2, 3), (3, 6)], 0)))
    A(h, r"Let $P(x)=x^{2}+b x+c$ with $P(1)=2$ and $P(3)=10$. Find $P(0)$.", L(solve_poly(x ** 2 + b * x + c, [b, c], [(1, 2), (3, 10)], 0)))
    # T19 omni-2546 (2,3,1,1) — direct lengths in the same figure
    h = "omni-2546"
    T19 = r"Let $P$ and $Q$ be points on line $l$ with $P Q=12$. Two circles are tangent to $l$ at $P$ and externally tangent to each other. A line through $Q$ meets the first circle at $A$ and $B$ ($A$ closer to $Q$) with $A B=10$, and another line through $Q$ meets the second circle at $C$ and $D$ ($C$ closer to $Q$) with $C D=7$. "
    A(h, T19 + r"Find $Q B \cdot Q D$.", L(18 * 16))
    A(h, T19 + r"Find $\frac{Q B}{Q C}$.", L(Fraction(18, 9)))
    # T21 omni-2207 (0,0,0,2) — n = 3 versions of the variants
    h = "omni-2207"
    T21 = r"How many functions $f:\{{1,2,3\}} \rightarrow\{{1,2,3\}}$ satisfy {cond} for all integers $i, j$ such that $1 \leq i<j \leq 3$?"
    A(h, T21.format(cond=r"$f(j) \leq f(i)+j-i$"), L(count_funcs(3, lambda fi, fj, i, j: fj <= fi + j - i)))
    A(h, T21.format(cond=r"$|f(j)-f(i)|<j-i$"), L(count_funcs(3, lambda fi, fj, i, j: abs(fj - fi) < j - i)))
    A(h, T21.format(cond=r"$f(j) \leq f(i)+2(j-i)$"), L(count_funcs(3, lambda fi, fj, i, j: fj <= fi + 2 * (j - i))))
    # T22 omni-1689 (2,4,0,0) — small totals
    h = "omni-1689"
    T22 = r"A student was counting his stones by {a}. He messed up $n$ times and instead counted {b}s and wound up at {t}. How many values of $n$ are possible?"
    for a_, b_, t in ((5, 3, 30), (7, 4, 56), (6, 5, 90)):
        A(h, T22.format(a=a_, b=b_, t=t), L(count_pairs(b_, a_, t)))
    # T23 omni-2600 (0,0,0,0) — the regular octagon, plainly asked
    h = "omni-2600"
    A(h, r"A regular octagon $A B C D E F G H$ has side length 1. Find its area.", L(sp.nsimplify(2 * (1 + sp.sqrt(2)))))
    A(h, r"A regular octagon $A B C D E F G H$ has side length 1. Find the area of square $A C E G$.", L(2 + sp.sqrt(2)))
    A(h, r"In a regular octagon $A B C D E F G H$, find $\sin \angle A B C$.", L(sp.sin(3 * sp.pi / 4)))
    # T24 omni-1346 (0,2,0,0) — tiny cafes with the changed rules
    h = "omni-1346"
    C24 = r"A cafe has %s and %s. People enter in groups of size between 1 and 4, and groups never share a table. %s $M$ groups with a total of $N$ people enter and sit down, after which every table and counter seat is occupied. What is the minimum possible value of $M+N$?"
    tf = r"Every group, including a group of 1, sits at a table if one is free, and otherwise in counter seats."
    A(h, C24 % ("1 table", "1 counter seat", tf), L(cafe(1, 1, single_table_first=True)))
    A(h, C24 % ("2 tables", "1 counter seat", tf), L(cafe(2, 1, single_table_first=True)))
    A(h, C24 % ("2 tables", "1 counter seat", r"A group of more than 1 sits at a table if one is free, and otherwise in counter seats; a group of 1 sits at the counter if a seat is free, and otherwise at a table."), L(cafe(2, 1)))
    # T25 omni-1728 (2,5,1,4)
    h = "omni-1728"
    T25 = r"The angles of a convex $n$-sided polygon form an arithmetic progression whose common difference is ${d}^{{\circ}}$. Find the largest possible value of $n$."
    for d in (10, 20):
        A(h, T25.format(d=d), L(max_ap_polygon([d])))
    # T26 omni-1900 (1,0,0,0) — the same recursion from other starts, a few steps
    h = "omni-1900"
    Rr = r"A sequence satisfies $a_{0}=0, a_{1}=%d, a_{2}=%d$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n \geq 1$. Find $a_{%d}$."
    for a1, a2, k in ((2, 1, 4), (3, 2, 4)):
        A(h, Rr % (a1, a2, k), L(rec_seq(a1, a2, k)))
    # T28 omni-1391 (15,13,2,16) — sign questions with new values
    h = "omni-1391"
    mp.mp.dps = 40
    cases = [(r"Suppose $x$ is a real number such that $\sin \left(1+\cos ^{2} x+\sin ^{4} x\right)=\frac{19}{20}$. Compute $\cos \left(1+\sin ^{2} x+\cos ^{4} x\right)$.", 1, mp.sin, mp.cos, mp.mpf(19) / 20, -sp.sqrt(39) / 20),
             (r"Suppose $x$ is a real number such that $\cos \left(\cos ^{2} x+\sin ^{4} x\right)=\frac{3}{5}$. Compute $\sin \left(\sin ^{2} x+\cos ^{4} x\right)$.", 0, mp.cos, mp.sin, mp.mpf(3) / 5, sp.Rational(4, 5))]
    for text, s, fw, tg, val, ans in cases:
        assert abs(trig_case(s, fw, tg, val) - mp.mpf(sp.N(ans, 40))) < 1e-25, text
        A(h, text, L(ans))
    # T31 omni-3926 (2,2,3,1) — short iterations
    h = "omni-3926"
    D31 = r"For a positive integer $k$ let $g(k)$ be the sum of the squares of the digits of $k$. "
    A(h, D31 + r"Compute $g(g(g(4)))$.", L(digit_iter(4, 3, sum_sq_digits)))
    A(h, D31 + r"Compute $g(g(2^{10}))$.", L(digit_iter(2 ** 10, 2, sum_sq_digits)))
    A(h, r"For a positive integer $k$ let $f(k)$ be the square of the sum of the digits of $k$. Compute $f(f(2^{16}))$.", L(digit_iter(2 ** 16, 2, sq_digit_sum)))
    # T15 omni-2082 (2,3,8,13)
    h = "omni-2082"
    T15 = r"Let $A B C$ be a triangle with $A B=13, B C=14$, and $C A=15$. "
    A(h, T15 + r"Find the inradius of triangle $A B C$.", L(Fraction(84, 21)))
    A(h, T15 + r"Let $D$ be the foot of the altitude from $A$, and $I$ the incenter of $A B C$. Find $D I$.", L(sp.sqrt(1 + 16)))
    # T16 omni-2543 (6,3,0,2)
    h = "omni-2543"
    A(h, r"Let $r_{1}, r_{2}, r_{3}$ be the roots of $x^{3}-7$, and $K=\left(r_{1}+r_{2}\right)\left(r_{1}+r_{3}\right)\left(r_{2}+r_{3}\right)$. Find $K$.", L(-7))
    A(h, r"Let $r_{1}, r_{2}, r_{3}$ be the roots of $x^{3}-3 x+1$, and $K=\left(r_{1}+r_{2}\right)\left(r_{1}+r_{3}\right)\left(r_{2}+r_{3}\right)$. Find $K^{2}$.", L(1))
    # T20 omni-1362 (12,4,1,5)
    h = "omni-1362"
    T20 = r"The Fibonacci sequence is defined by $F_{{0}}=0, F_{{1}}=1$, and $F_{{n}}=F_{{n-1}}+F_{{n-2}}$. Find the smallest positive integer $m$ such that $F_{{m}} \equiv 0(\bmod {p})$ and $F_{{m+1}} \equiv 1(\bmod {p})$."
    for p in (19, 31, 5):
        A(h, T20.format(p=p), L(pisano(p)))
    # T27 omni-3709 (9,1,0,0)
    h = "omni-3709"
    T27 = r"How many permutations $(a_{{1}}, \ldots, a_{{{n}}})$ of $\{{1, \ldots, {n}\}}$ are there such that $k+a_{{k}}$ is a perfect square for all $1 \leq k \leq {n}$?"
    for n in (3, 12):
        A(h, T27.format(n=n), L(matchings(n, lambda k, v: is_sq(k + v))))
    return [], C
