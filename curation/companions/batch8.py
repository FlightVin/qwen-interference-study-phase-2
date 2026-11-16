"""Round 3 of CONFLICTING candidates (2026-11-16 ~01:15), for the 20 training problems still short
after round-2 screening (40/198 in band). Same-setting lookalikes pitched between the round-1/2
items that scored 16/16 and 0-2/16. Answers are computed here. Returns ([], conflicting)."""

from fractions import Fraction
from itertools import product

import mpmath as mp
import sympy as sp

from batch2 import bishop, placements, rook
from batch3 import count_pairs
from batch4 import cafe, digit_iter, is_sq, matchings, sum_sq_digits, trig_case
from batch6 import rec_seq, tangent_xy, walks
from common import L, frac

x = sp.Symbol("x")


def p_contains(n, pat):
    return Fraction(sum(pat in "".join(w) for w in product("HT", repeat=n)), 2 ** n)


def build():
    C = []
    A = lambda h, p, a: C.append((h, p, a))
    h = "omni-1814"
    A(h, r"The circle $x^{2}+y^{2}=5$ meets the hyperbola $x y=2$ at four points. Find the product of their four $x$-coordinates.", L(sp.prod(sp.solve(x ** 4 - 5 * x ** 2 + 4, x))))
    A(h, r"The circle $x^{2}+y^{2}=10$ meets the hyperbola $x y=3$ at four points. Find the sum of the squares of their four $x$-coordinates.", L(sum(r ** 2 for r in sp.solve(x ** 4 - 10 * x ** 2 + 9, x))))
    h = "omni-0962"
    A(h, r"A fair coin is flipped 5 times. What is the probability that two heads in a row appear at least once?", frac(p_contains(5, "HH")))
    A(h, r"A fair coin is flipped 4 times (H for heads, T for tails). What is the probability that the pattern HT appears as consecutive flips?", frac(p_contains(4, "HT")))
    h = "omni-0528"
    A(h, r"How many points with all coordinates in $\{-1,0,1\}$ in three-dimensional space are at distance $\sqrt{2}$ from the origin?", L(sum(1 for p in product((-1, 0, 1), repeat=3) if sum(c * c for c in p) == 2)))
    A(h, r"How many points with all coordinates in $\{-1,0,1\}$ in four-dimensional space are at distance $\sqrt{2}$ from the origin?", L(sum(1 for p in product((-1, 0, 1), repeat=4) if sum(c * c for c in p) == 2)))
    A(h, r"Let $S$ be the set of 9 points in the plane with both coordinates in $\{-1,0,1\}$. How many sequences $P_{1}, P_{2}, P_{3}$ of points in $S$ have $P_{1}=(0,0)$ and $P_{1} P_{2}=P_{2} P_{3}=1$?", L(walks(2, 1, 2, False)))
    h = "omni-0596"
    A(h, r"Triangle $A B C$ has vertices $A=(0,6), B=(0,0), C=(8,0)$. Find the area of the set of points inside the triangle at distance at most 2 from line $A B$.", frac(Fraction(12) - Fraction(3, 2)))
    A(h, r"Triangle $A B C$ has vertices $A=(0,6), B=(0,0), C=(8,0)$. Find the area of the set of points inside the triangle at distance at most 1 from line $B C$.", frac(Fraction(8) - Fraction(2, 3)))
    h = "omni-0848"
    A(h, r"A regular hexagon $A B C D E F$ has side length 2. Find the area of triangle $A C E$.", L(sp.sqrt(3) / 4 * 12))
    A(h, r"A regular hexagon $A B C D E F$ has side length 2. Find the area of triangle $A B C$.", L(sp.Rational(1, 2) * 4 * sp.sin(2 * sp.pi / 3)))
    h = "omni-0912"
    S2 = [frozenset(s) for s in ([], [1], [2], [1, 2])]
    S3 = [frozenset(s) for k in range(4) for s in __import__("itertools").combinations([1, 2, 3], k)]
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2\}$ satisfy $X \cap Y=\varnothing$?", L(sum(1 for X, Y in product(S2, repeat=2) if not X & Y)))
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2,3\}$ satisfy $X \subseteq Y$?", L(sum(1 for X, Y in product(S3, repeat=2) if X <= Y)))
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2\}$ satisfy $X \cup Y=\{1,2\}$?", L(sum(1 for X, Y in product(S2, repeat=2) if X | Y == frozenset([1, 2]))))
    h = "omni-2584"
    A(h, r"Two faces of a rectangular box have areas 12 and 20 and share an edge of length 4. Find the area of the third face.", L(3 * 5))
    A(h, r"The faces of a rectangular box have areas 6, 10 and 15. Find its volume.", L(sp.sqrt(6 * 10 * 15)))
    h = "omni-2215"
    A(h, r"Let $P(x)=x^{3}+a x^{2}+b x+c$ with $P(0)=1, P(1)=2$, and $P(-1)=0$. Find $P(2)$.", L(9))
    A(h, r"Let $P(x)=x^{2}+b x+c$ with $P(2)=7$ and $P(5)=28$. Find $P(3)$.", L(12))
    h = "omni-2546"
    A(h, r"A line through a point $Q$ meets a circle at $A$ and $B$ with $Q A=4$ and $A B=5$. Find the length of the tangent segment from $Q$ to the circle.", L(6))
    A(h, r"A line through a point $Q$ meets a circle at $C$ and $D$ with $Q C=9$ and $C D=7$. Find the length of the tangent segment from $Q$ to the circle.", L(12))
    h = "omni-1689"
    T22 = r"A student was counting his stones by {a}. He messed up $n$ times and instead counted {b}s and wound up at {t}. How many values of $n$ are possible?"
    for a_, b_, t in ((4, 3, 24), (2, 3, 12)):
        A(h, T22.format(a=a_, b=b_, t=t), L(count_pairs(b_, a_, t)))
    h = "omni-2600"
    A(h, r"A regular octagon $A B C D E F G H$ has side length 1. Find the length of diagonal $A C$.", L(sp.sqrt(2 + sp.sqrt(2))))
    A(h, r"A regular octagon $A B C D E F G H$ has side length 1. Find the area of triangle $A B C$.", L(sp.Rational(1, 2) * sp.sin(3 * sp.pi / 4)))
    h = "omni-1346"
    A(h, r"A cafe has 2 tables and no counter seats. Groups of 1 to 4 people enter, and each group sits at an empty table. $M$ groups with a total of $N$ people enter, after which both tables are occupied. What is the minimum possible value of $M+N$?", L(4))
    A(h, r"A cafe has 3 tables and no counter seats. Groups of 1 to 4 people enter, and each group sits at an empty table. $M$ groups with a total of $N$ people enter, after which all tables are occupied. What is the minimum possible value of $M+N$?", L(6))
    A(h, r"A cafe has no tables and 3 counter seats. Groups of 1 to 4 people enter and sit in counter seats if enough are free. $M$ groups with a total of $N$ people enter, after which every seat is occupied. What is the minimum possible value of $M+N$?", L(cafe(0, 3)))
    h = "omni-1900"
    Rr = r"A sequence satisfies $a_{0}=0, a_{1}=%d, a_{2}=%d$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n \geq 1$. Find $a_{%d}$."
    for a1, a2, k in ((2, 1, 5), (3, 2, 5), (2, 3, 4)):
        A(h, Rr % (a1, a2, k), L(rec_seq(a1, a2, k)))
    h = "omni-3709"
    T27 = r"How many permutations $(a_{{1}}, \ldots, a_{{{n}}})$ of $\{{1, \ldots, {n}\}}$ are there such that $k+a_{{k}}$ is a perfect square for all $1 \leq k \leq {n}$?"
    for n in (9, 13):
        A(h, T27.format(n=n), L(matchings(n, lambda k, v: is_sq(k + v))))
    h = "omni-1391"
    mp.mp.dps = 40
    cases = [(r"Suppose $x$ is a real number such that $\sin \left(\cos ^{2} x+\sin ^{4} x\right)=\frac{7}{10}$. Compute $\cos \left(\sin ^{2} x+\cos ^{4} x\right)$.", 0, mp.sin, mp.cos, mp.mpf(7) / 10, sp.sqrt(51) / 10),
             (r"Suppose $x$ is a real number such that $\cos \left(2+\cos ^{2} x+\sin ^{4} x\right)=-\frac{24}{25}$. Compute $\sin \left(2+\sin ^{2} x+\cos ^{4} x\right)$.", 2, mp.cos, mp.sin, -mp.mpf(24) / 25, sp.Rational(7, 25))]
    for text, s, fw, tg, val, ans in cases:
        assert abs(trig_case(s, fw, tg, val) - mp.mpf(sp.N(ans, 40))) < 1e-25, text
        A(h, text, L(ans))
    h = "omni-3926"
    D = r"For a positive integer $k$ let $g(k)$ be the sum of the squares of the digits of $k$. "
    A(h, D + r"Compute $g(g(g(g(4))))$.", L(digit_iter(4, 4, sum_sq_digits)))
    A(h, D + r"Compute $g\left(g\left(g\left(2^{10}\right)\right)\right)$.", L(digit_iter(2 ** 10, 3, sum_sq_digits)))
    h = "omni-1877"
    T8 = r"Let $A, B, C$ be points in that order along a line with $A B={a}$ and $B C={c}$. Let $\omega$ be the circle of radius ${r}$ centered at $B$, and let $\ell_{{1}}$, $\ell_{{2}}$ be the tangents to $\omega$ through $A$ and $C$ on the same side of line $A C$, meeting at $K$. Let $X$ on segment $K A$ and $Y$ on segment $K C$ be such that $X Y \| A C$ and $X Y$ is tangent to $\omega$. Find $X Y$."
    for a, c, r in ((15, 15, 9), (20, 20, 12)):
        A(h, T8.format(a=a, c=c, r=r), L(tangent_xy(a, c, r)))
    h = "omni-0757"
    A(h, r"Let $n$ be the maximum number of bishops that can be placed on a $2 \times 2$ chessboard so that no two attack each other, and let $k$ be the number of ways to place $n$ such bishops. Find $n+k$.", L(sum(placements(2, 2, bishop))))
    A(h, r"Let $n$ be the maximum number of rooks that can be placed on a $3 \times 3$ chessboard so that no two attack each other, and let $k$ be the number of ways to place $n$ such rooks. Find $n+k$.", L(sum(placements(3, 3, rook))))
    h = "omni-1299"
    A(h, r"What is the maximum number of points in which a circle and the boundary of a convex quadrilateral can intersect?", L(8))
    A(h, r"What is the maximum number of points in which two distinct parabolas in the plane can intersect?", L(4))
    h = "omni-2082"
    T15 = r"Let $A B C$ be a triangle with $A B=13, B C=14$, and $C A=15$. "
    A(h, T15 + r"The incircle touches $A B$ at $Z$. Find $A Z$.", L(21 - 14))
    A(h, T15 + r"Let $D$ be the foot of the altitude from $A$. The incircle of $A B C$ touches $B C$ at $T$. Find $D T$.", L(abs((21 - 15) - 5)))
    return [], C
