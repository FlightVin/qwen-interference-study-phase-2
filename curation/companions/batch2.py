"""Companion candidates for training problems T8-T15 (same rules as batch1.py)."""

from fractions import Fraction
from itertools import chain, combinations, product
from math import ceil, floor

import sympy as sp

from common import L, frac


def tangent_xy(a, c, r):
    """A=(-a,0), B=(0,0), C=(c,0); tangents from A and C to the circle (B, r) on the upper side meet
    at K; XY on KA, KC is the upper tangent y=r. Returns XY (checks K lies above the line y=r)."""
    a, c, r = map(sp.Integer, (a, c, r))
    mA, mC = r / sp.sqrt(a * a - r * r), r / sp.sqrt(c * c - r * r)
    kx = sp.solve(sp.Eq(mA * (sp.Symbol("x") + a), -mC * (sp.Symbol("x") - c)), sp.Symbol("x"))[0]
    assert sp.simplify(mA * (kx + a) - r) > 0
    return sp.simplify((c - r / mC) - (-a + r / mA))


def placements(rows, cols, attacks, squares=None):
    """(maximum size, number of maximum sets) of pairwise non-attacking pieces."""
    sq = squares or [(i, j) for i in range(1, rows + 1) for j in range(1, cols + 1)]
    best = [0, 0]
    def go(k, chosen):
        if k == len(sq):
            n = len(chosen)
            if n > best[0]:
                best[:] = [n, 1]
            elif n == best[0]:
                best[1] += 1
            return
        if len(chosen) + len(sq) - k < best[0]:
            return
        if all(not attacks(sq[k], q) for q in chosen):
            go(k + 1, chosen + [sq[k]])
        go(k + 1, chosen)
    go(0, [])
    return tuple(best)


bishop = lambda p, q: abs(p[0] - q[0]) == abs(p[1] - q[1])
rook = lambda p, q: p[0] == q[0] or p[1] == q[1]
queen = lambda p, q: bishop(p, q) or rook(p, q)
king = lambda p, q: max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1


def six_triangles(x, y, z, which="K"):
    """Segments AD, BE, CF (lengths x, y, z) share midpoint M; consecutive rays at 60 degrees."""
    r = [x, y, z, x, y, z]
    P = [(sp.Rational(r[i], 2) * sp.cos(sp.pi * i / 3), sp.Rational(r[i], 2) * sp.sin(sp.pi * i / 3)) for i in range(6)]
    def tri(i, j, k):
        (x1, y1), (x2, y2), (x3, y3) = P[i], P[j], P[k]
        return sp.Abs(((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2)
    if which == "K":
        return sp.nsimplify(sp.simplify(sum(tri(i, (i + 1) % 6, (i + 2) % 6) for i in range(6))))
    if which == "hex":
        return sp.nsimplify(sp.simplify(sum(sp.Abs(P[i][0] * P[(i + 1) % 6][1] - P[(i + 1) % 6][0] * P[i][1]) / 2 for i in range(6))))
    if which == "ACE":
        return sp.nsimplify(sp.simplify(tri(0, 2, 4)))


def count_int(pred, lo=-20000, hi=20000):
    return sum(1 for n in range(lo, hi + 1) if pred(n))


def subsets(s):
    s = list(s)
    return [frozenset(c) for c in chain.from_iterable(combinations(s, k) for k in range(len(s) + 1))]


def pair_count(base, proper_family=True, cond_a=True, cond_b=True):
    """The T13 count: family = (proper) subsets of base; pairs of proper subsets of the family."""
    fam = [s for s in subsets(base) if (s != frozenset(base) or not proper_family)]
    lt = lambda u, v: u < v
    big = [S for S in subsets(fam) if S != frozenset(fam)]
    n = 0
    for S, T in product(big, repeat=2):
        if cond_a and (S < T or T < S):
            continue
        if cond_b and any(lt(s, t) or lt(t, s) for s in S for t in T):
            continue
        n += 1
    return n


def start_money(shops, spend, mult=2, end=0, spend_first=False):
    x = Fraction(end)
    for _ in range(shops):  # undo one (lottery, shop) step
        x = (x + spend) / mult if not spend_first else x / mult + spend
    return x


def build():
    R, C = [], []
    # ---- T8 omni-1877: excenter, AX = AB, sup XY = 36 not attained ----
    h = "omni-1877"
    R += [(h, r"Let $A B X$ be a triangle with $A B=20$. Through $X$ draw the line parallel to $A B$. The ray $X B$ bisects the angle between ray $X A$ and the half of this parallel line on the same side of line $A X$ as $B$. Find $A X$.", L(20)),
          (h, r"Points $K, A, C$ form a nondegenerate triangle with $K A=20 x$, $K C=18 x$ and $A C=38$, where $x>1$ is real. Find the least number $M$ such that $x<M$ for every such triangle.", L(19)),
          (h, r"Let $x$ be a real number with $1<x<16$. Find the largest integer value taken by $30-\frac{30}{x}$.", L(floor(30 - Fraction(30, 16)))),  # sup 28.125 is not an integer, and 28 is attained at x = 15
          (h, r"In triangle $K A C$, point $X$ lies on segment $K A$ and point $Y$ lies on segment $K C$ with $X Y \parallel A C$. If $K A=50$, $X A=20$ and $A C=38$, find $X Y$.", frac(Fraction(38 * 30, 50)))]
    T8 = (r"Let $A, B, C$ be points in that order along a line, such that $A B={a}$ and $B C={c}$. Let $\omega$ be the circle of radius ${r}$ centered at $B$, and let $\ell_{{1}}$ and $\ell_{{2}}$ be the tangents to $\omega$ through $A$ and $C$, respectively, on the same side of line $A C$. Let $K$ be the intersection of $\ell_{{1}}$ and $\ell_{{2}}$. Let $X$ lie on segment $\overline{{K A}}$ and $Y$ lie on segment $\overline{{K C}}$ such that $X Y \| B C$ and $X Y$ is tangent to $\omega$. Find $X Y$.")
    C += [(h, T8.format(a=a, c=c, r=r), L(tangent_xy(a, c, r))) for a, c, r in ((10, 10, 6), (20, 15, 12), (13, 15, 12), (13, 13, 5))]
    # ---- T9 omni-0757: bishops on 6x6, n + k = 74 ----
    h = "omni-0757"
    black6 = [(i, j) for i in range(1, 7) for j in range(1, 7) if (i + j) % 2 == 1]
    assert sum(placements(6, 6, bishop)) == 74
    R += [(h, r"What is the maximum number of bishops that can be placed on a $6 \times 6$ chessboard so that no two bishops attack each other?", L(placements(6, 6, bishop)[0])),
          (h, r"How many ways are there to place 6 bishops on a $4 \times 4$ chessboard so that no two bishops attack each other?", L(placements(4, 4, bishop)[1])),
          (h, r"Label the squares of a $6 \times 6$ chessboard $(x, y)$ with $1 \leq x, y \leq 6$. What is the maximum number of bishops that can be placed on squares with $x+y$ odd so that no two attack each other?", L(placements(6, 6, bishop, black6)[0])),
          (h, r"Label the squares of a $6 \times 6$ chessboard $(x, y)$ with $1 \leq x, y \leq 6$. In how many ways can 5 bishops be placed on squares with $x+y$ odd so that no two attack each other?", L(placements(6, 6, bishop, black6)[1]))]
    T9 = r"Let $n$ be the maximum number of {p}s that can be placed on the squares of a ${s} \times {s}$ chessboard such that no two {p}s attack each other. Let $k$ be the number of ways to put $n$ {p}s on a ${s} \times {s}$ chessboard such that no two attack each other. Find $n+k$."
    C += [(h, T9.format(p=p, s=s), L(sum(placements(s, s, f)))) for p, s, f in (("rook", 6, rook), ("queen", 6, queen), ("queen", 5, queen), ("king", 4, king))]
    # ---- T10 omni-1299: two E-shapes meet at most 11 times (bounds from the reference solution) ----
    h = "omni-1299"
    Cdef = r"A $C$-shape is a figure consisting of two rays pointing in the same direction together with a line segment perpendicular to both rays whose endpoints are the endpoints of the rays. "
    R += [(h, r"Two rays in the plane do not overlap. What is the maximum number of points in which they can intersect?", L(1)),
          (h, Cdef + r"What is the maximum number of points in which a $C$-shape and a ray can intersect, if they do not overlap?", L(2)),
          (h, Cdef + r"What is the maximum number of points in which two $C$-shapes can intersect, if they do not overlap?", L(6)),
          (h, r"What is the maximum number of points in which the boundaries of two triangles in the plane can intersect, if no two sides overlap?", L(6))]
    C += [(h, r"A $V$-shape is a figure consisting of two rays with a common endpoint that do not lie on one line. What is the maximum number of points in which two $V$-shapes can intersect, if they do not overlap?", L(4)),
          (h, r"A zigzag is a figure made of three line segments $P_{1} P_{2}, P_{2} P_{3}, P_{3} P_{4}$. What is the maximum number of points in which a zigzag and a line can intersect, if the line contains none of the segments?", L(3)),
          (h, r"What is the maximum number of points in which the boundaries of two convex quadrilaterals can intersect, if no two sides overlap?", L(8)),
          (h, r"What is the maximum number of points in which a circle and the boundary of a triangle can intersect?", L(6))]
    # ---- T11 omni-0848: K = sqrt3 (2xy+2yz+2zx)/16 = hexagon area ----
    h = "omni-0848"
    assert sp.simplify(six_triangles(7, 10, 18) * sp.sqrt(3) - 141) == 0
    A, B, Cc = (3, 0), (2, 2 * sp.sqrt(3)), (-sp.Rational(5, 2), 5 * sp.sqrt(3) / 2)
    abc = sp.Abs(((B[0] - A[0]) * (Cc[1] - A[1]) - (Cc[0] - A[0]) * (B[1] - A[1])) / 2)
    R += [(h, r"Find the area of a triangle with two sides of lengths 5 and 9 and an included angle of $120^{\circ}$.", L(sp.Rational(45, 2) * sp.sin(2 * sp.pi / 3))),
          (h, r"Point $M$ satisfies $M A=3$, $M B=4$, $M C=5$, $\angle A M B=\angle B M C=60^{\circ}$ and $\angle A M C=120^{\circ}$. Find the area of triangle $A B C$.", L(abc)),
          (h, r"Segments $A D$ and $B E$ share a common midpoint and form a $60^{\circ}$ angle. If $A D=6$ and $B E=8$, find the area of quadrilateral $A B D E$.", L(12 * sp.sqrt(3))),
          (h, r"Let $A D, B E$, and $C F$ be segments sharing a common midpoint, such that each pair of segments forms a $60^{\circ}$ angle and $A, B, C, D, E, F$ are in order around the midpoint. If $A D=4, B E=6$, and $C F=8$, find the area of hexagon $A B C D E F$.", L(six_triangles(4, 6, 8, "hex")))]
    T11 = r"Let $A D, B E$, and $C F$ be segments sharing a common midpoint, such that each pair of segments forms a $60^{{\circ}}$ angle and $A D=B E=C F={s}$. "
    C += [(h, T11.format(s=8) + r"Let $K$ denote the sum of the areas of the six triangles $\triangle A B C, \triangle B C D, \triangle C D E, \triangle D E F, \triangle E F A$, and $\triangle F A B$. Compute $K \sqrt{3}$.", L(six_triangles(8, 8, 8) * sp.sqrt(3))),
          (h, T11.format(s=12) + r"Let $K$ denote the sum of the areas of the six triangles $\triangle A B C, \triangle B C D, \triangle C D E, \triangle D E F, \triangle E F A$, and $\triangle F A B$. Compute $K \sqrt{3}$.", L(six_triangles(12, 12, 12) * sp.sqrt(3))),
          (h, T11.format(s=4) + r"Find the area of triangle $A C E$.", L(six_triangles(4, 4, 4, "ACE"))),
          (h, T11.format(s=10) + r"Find the area of hexagon $A B C D E F$.", L(six_triangles(10, 10, 10, "hex")))]
    # ---- T12 omni-1590: f(n + 10100) = f(n) - 1, one solution per residue ----
    h = "omni-1590"
    assert count_int(lambda n: 1 + floor(Fraction(100 * n, 101)) == ceil(Fraction(99 * n, 100)), -30000, 30000) == 10100
    R += [(h, r"Let $f(n)=\left\lceil\frac{3 n}{4}\right\rceil-\left\lfloor\frac{4 n}{5}\right\rfloor$. There is a constant $c$ such that $f(n+20)=f(n)+c$ for every integer $n$. Find $c$.", L(next(iter({ceil(Fraction(3 * (n + 20), 4)) - floor(Fraction(4 * (n + 20), 5)) - ceil(Fraction(3 * n, 4)) + floor(Fraction(4 * n, 5)) for n in range(-50, 50)})))),
          (h, r"Find the number of integers $n$ such that $1+\left\lfloor\frac{n}{3}\right\rfloor=\left\lceil\frac{n}{2}\right\rceil$.", L(count_int(lambda n: 1 + floor(Fraction(n, 3)) == ceil(Fraction(n, 2))))),
          (h, r"Find the number of integers $n$ such that $1+\left\lfloor\frac{4 n}{5}\right\rfloor=\left\lceil\frac{3 n}{4}\right\rceil$.", L(count_int(lambda n: 1 + floor(Fraction(4 * n, 5)) == ceil(Fraction(3 * n, 4))))),
          (h, r"Find the number of integers $n$ with $0 \leq n<12$ such that $\left\lceil\frac{n}{3}\right\rceil=\left\lfloor\frac{n}{4}\right\rfloor+1$.", L(count_int(lambda n: ceil(Fraction(n, 3)) == floor(Fraction(n, 4)) + 1, 0, 11)))]
    C += [(h, r"Find the number of integers $n$ such that $\left\lfloor\frac{n}{2}\right\rfloor=\left\lfloor\frac{n}{3}\right\rfloor$.", L(count_int(lambda n: floor(Fraction(n, 2)) == floor(Fraction(n, 3))))),
          (h, r"Find the number of integers $n$ such that $\left\lfloor\frac{n}{3}\right\rfloor=\left\lfloor\frac{n}{4}\right\rfloor$.", L(count_int(lambda n: floor(Fraction(n, 3)) == floor(Fraction(n, 4))))),
          (h, r"Find the number of integers $n$ such that $\left\lceil\frac{n}{2}\right\rceil=\left\lceil\frac{n}{3}\right\rceil$.", L(count_int(lambda n: ceil(Fraction(n, 2)) == ceil(Fraction(n, 3))))),
          (h, r"Find the number of integers $n$ such that $\left\lfloor\frac{n}{5}\right\rfloor=\left\lfloor\frac{n}{7}\right\rfloor$.", L(count_int(lambda n: floor(Fraction(n, 5)) == floor(Fraction(n, 7)))))]
    # ---- T13 omni-0912: pairs of proper subsets of the proper subsets of {S, T} ----
    h = "omni-0912"
    assert pair_count("ST") == 7
    R += [(h, r"Let $\mathcal{P}$ be the set of all proper subsets of $\{\mathrm{S}, \mathrm{T}\}$. How many subsets does $\mathcal{P}$ have?", L(2 ** 3)),
          (h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2,3\}$ are there such that $X$ is not a proper subset of $Y$ and $Y$ is not a proper subset of $X$?", L(sum(1 for X, Y in product(subsets("123"), repeat=2) if not (X < Y or Y < X)))),
          (h, r"How many ordered pairs $(U, V)$, with $U$ and $V$ each one of the sets $\varnothing,\{\mathrm{S}\},\{\mathrm{T}\}$ (possibly equal), are there such that $U$ is not a proper subset of $V$ and $V$ is not a proper subset of $U$?", L(sum(1 for U, V in product(subsets("ST")[:3], repeat=2) if not (U < V or V < U)))),
          (h, r"How many subsets of $\{\varnothing,\{\mathrm{S}\},\{\mathrm{T}\}\}$ contain no two sets such that one is a proper subset of the other?", L(sum(1 for F in subsets(range(3)) if not any(subsets("ST")[i] < subsets("ST")[j] for i in F for j in F))))]
    T13 = (r"Define $P=\{{{P}\}}$ and let $\mathcal{{P}}$ be the set of all {fam} of $P$. How many ordered pairs $(\mathcal{{S}}, \mathcal{{T}})$ of proper subsets of $\mathcal{{P}}$ are there such that {conds}?")
    ca = r"$\mathcal{S}$ is not a proper subset of $\mathcal{T}$ and $\mathcal{T}$ is not a proper subset of $\mathcal{S}$"
    cb = r"for any sets $S \in \mathcal{S}$ and $T \in \mathcal{T}$, $S$ is not a proper subset of $T$ and $T$ is not a proper subset of $S$"
    ST, STU = r"\mathrm{S}, \mathrm{T}", r"\mathrm{S}, \mathrm{T}, \mathrm{U}"
    C += [(h, T13.format(P=ST, fam="subsets", conds=f"(a) {ca}; and (b) {cb}"), L(pair_count("ST", proper_family=False))),
          (h, T13.format(P=ST, fam="proper subsets (a proper subset is a subset that is not the set itself)", conds=cb), L(pair_count("ST", cond_a=False))),
          (h, T13.format(P=ST, fam="proper subsets (a proper subset is a subset that is not the set itself)", conds=ca), L(pair_count("ST", cond_b=False))),
          (h, T13.format(P=STU, fam="proper subsets (a proper subset is a subset that is not the set itself)", conds=f"(a) {ca}; and (b) {cb}"), L(pair_count("STU")))]
    # ---- T14 omni-1315: work backwards, d = 1024 - 2^(10-n) ----
    h = "omni-1315"
    assert start_money(10, 1024) == 1023
    S14 = r"Stacy has $d$ dollars. She enters a mall with {n} shops and a lottery stall. {order} When she comes out of the last {last} she has {end}. What is the minimum possible value of $d$?"
    std = r"First she goes to the lottery and her money is {m}, then she goes into the first shop and spends {s} dollars. After that she alternates playing the lottery and getting her money {m} (Stacy always wins) then going into a new shop and spending {s} dollars."
    R += [(h, r"Stacy plays a lottery that doubles her money and then spends 1024 dollars in a shop. Afterwards she has no money left. How much money did she have before the lottery?", frac(start_money(1, 1024))),
          (h, S14.format(n=2, order=std.format(m="doubled", s=1024), last="shop", end="no money left"), frac(start_money(2, 1024))),
          (h, S14.format(n=3, order=std.format(m="doubled", s=1024), last="shop", end="no money left"), frac(start_money(3, 1024))),
          (h, S14.format(n=6, order=std.format(m="doubled", s=64), last="shop", end="no money left"), frac(start_money(6, 64)))]
    rev = r"First she goes into the first shop and spends 1024 dollars, then she plays the lottery and her money is doubled. After that she alternates going into a new shop and spending 1024 dollars, then playing the lottery and getting her money doubled (Stacy always wins)."
    C += [(h, r"Stacy has $d$ dollars. She enters a mall with 10 shops and a lottery stall. " + rev + r" When she leaves the lottery for the tenth time she has no money left. What is the minimum possible value of $d$?", frac(start_money(10, 1024, spend_first=True))),
          (h, S14.format(n=6, order=std.format(m="tripled", s=729), last="shop", end="no money left"), frac(start_money(6, 729, mult=3))),
          (h, S14.format(n=3, order=std.format(m="doubled", s=1000), last="shop", end="no money left"), frac(start_money(3, 1000))),
          (h, S14.format(n=10, order=std.format(m="doubled", s=1024), last="shop", end="exactly 1024 dollars"), frac(start_money(10, 1024, end=1024)))]
    # ---- T15 omni-2082: 13-14-15 triangle, incircles of ABD and ACD ----
    h = "omni-2082"
    T15 = r"Let $A B C$ be a triangle with $A B=13, B C=14$, and $C A=15$. "
    R += [(h, T15 + r"Let $D$ be the foot of the altitude from $A$ to $B C$. Find $A D$.", L(12)),
          (h, T15 + r"Let $D$ be the foot of the altitude from $A$ to $B C$. Find $B D$.", L(5)),
          (h, r"Find the inradius of a right triangle with legs 9 and 12.", L(Fraction(9 + 12 - 15, 2))),
          (h, r"An isosceles right triangle has hypotenuse 6. Find its area.", L(9))]
    C += [(h, T15 + r"The incircle of $A B C$ touches $B C$ at $X$, and the excircle opposite $A$ touches $B C$ at $Y$. Find $X Y$.", L(abs(15 - 13))),
          (h, T15 + r"The incircle of $A B C$ touches $B C$ at $X$, and the excircle opposite $A$ touches $B C$ at $Y$. Find the area of triangle $A X Y$.", L(Fraction(1, 2) * 2 * 12)),
          (h, T15 + r"Let $D$ be the foot of the altitude from $A$ to $B C$. Find the distance between the incenters of triangles $A B D$ and $A C D$.", L(sp.sqrt((3 - (-2)) ** 2 + (3 - 2) ** 2))),
          (h, T15 + r"Let $D$ be the foot of the altitude from $A$ to $B C$. Find the radius of the circle through $A, B$, and $D$.", frac(Fraction(13, 2)))]
    return R, C
