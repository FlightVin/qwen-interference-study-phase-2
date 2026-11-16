"""Round 3 of RELATED candidates (2026-11-16 ~01:00), for the 20 training problems still short after
round-2 screening (47/206 in band). Pitched between the round-1/2 items that scored 16/16 and those
that scored 0-2/16 for the same problem. Answers are computed here. Returns (related, [])."""

from fractions import Fraction
from itertools import product

import sympy as sp

from batch4 import P_square, digit_iter, sq_digit_sum
from common import L, frac

x = sp.Symbol("x")


def count_sol(a, b, t):
    return sum(1 for n in range(t // a + 1) if (t - a * n) % b == 0)


def min_score(seats, maxg=4):  # fill `seats` counter seats with groups of size <= maxg; minimise sum(s+1)
    best = {0: 0}
    for s in range(1, seats + 1):
        best[s] = min(best[s - g] + g + 1 for g in range(1, min(maxg, s) + 1))
    return best[seats]


def tan_seq(a2, k):
    s = [sp.Integer(0), sp.Integer(1), sp.Rational(a2)]
    for n in range(1, k - 1):
        s.append(sp.simplify((s[n] + s[n - 1]) / (s[n] * s[n - 1] - 1)))
    return s[k]


def build():
    R = []
    A = lambda h, p, a: R.append((h, p, a))
    h = "omni-1735"
    A(h, r"Write $\left(1+\frac{1}{8}\right)^{8}$ as a fraction in lowest terms.", frac(Fraction(9, 8) ** 8))
    A(h, r"Compute $\sum_{k=0}^{6} \binom{6}{k} \frac{1}{6^{k}}$.", frac(sum(Fraction(sp.binomial(6, k), 6 ** k) for k in range(7))))
    h = "omni-1814"
    A(h, r"Let $A=\left(3, \frac{1}{3}\right)$ and $B=\left(-12,-\frac{1}{12}\right)$. The point $X=\left(x, \frac{1}{x}\right)$ with $x>0$ satisfies $\angle A X B=90^{\circ}$. Find the distance from $X$ to the origin.", L(sp.sqrt(36 + sp.Rational(1, 36))))
    A(h, r"Let $A=\left(2, \frac{1}{2}\right)$ and $B=\left(-8,-\frac{1}{8}\right)$. The two points $X$ on $x y=1$ other than $A, B$ with $\angle A X B=90^{\circ}$ are symmetric about the origin. Find the distance between them.", L(sp.sqrt(4 * 16 + 4 * sp.Rational(1, 16))))
    h = "omni-2140"
    A(h, r"A hexagon has its vertices at distances alternately 1 and 3 from a point $O$, and consecutive vertices subtend $60^{\circ}$ at $O$. Find the area of the hexagon.", L(6 * sp.Rational(1, 2) * 3 * sp.sin(sp.pi / 3)))
    A(h, r"A triangle has two sides of lengths $\sqrt[3]{12}$ and $\sqrt[3]{18}$ with a $60^{\circ}$ angle between them. Find its area.", L(sp.Rational(1, 2) * 6 * sp.sin(sp.pi / 3)))
    h = "omni-0962"
    A(h, r"A fair coin is flipped repeatedly. What is the expected number of flips until three heads in a row appear for the first time?", L(14))
    A(h, r"A fair coin is flipped repeatedly (H for heads, T for tails). What is the expected number of flips until the pattern HTT appears for the first time?", L(8))
    h = "omni-0528"
    A(h, r"Let $F_{1}=F_{2}=1$ and $F_{n}=F_{n-1}+F_{n-2}$. Find the largest integer $n$ such that $2^{n}$ divides $F_{12}$.", L(4))
    A(h, r"How many points with all coordinates in $\{-1,0,1\}$ in four-dimensional space are at distance exactly $\sqrt{3}$ from the origin?", L(sum(1 for p in product((-1, 0, 1), repeat=4) if sum(c * c for c in p) == 3)))
    A(h, r"How many points with all coordinates in $\{-1,0,1\}$ in four-dimensional space are at distance exactly 2 from the point $(1,1,1,1)$?", L(sum(1 for p in product((-1, 0, 1), repeat=4) if sum((c - 1) ** 2 for c in p) == 4)))
    h = "omni-0848"
    A(h, r"Point $M$ satisfies $M A=2$, $M B=4$, $M C=6$, $\angle A M B=\angle B M C=60^{\circ}$ and $\angle A M C=120^{\circ}$. Find the area of quadrilateral $M A B C$.", L(sp.Rational(1, 2) * sp.sin(sp.pi / 3) * (8 + 24)))
    A(h, r"Find the area of a triangle with two sides of lengths 3 and 5 and an included angle of $60^{\circ}$.", L(sp.Rational(15, 2) * sp.sin(sp.pi / 3)))
    h = "omni-2082"
    A(h, r"A right triangle has legs 9 and 12. Find its inradius.", L(3))
    A(h, r"A right triangle has legs 5 and 12. Its incircle touches a leg at $X$. Find the distance from the right-angle vertex to $X$.", L(2))
    h = "omni-2584"
    A(h, r"A rectangular box has faces of areas 48, 80 and 240. Find the sum of its three edge lengths.", L(4 + 12 + 20))
    A(h, r"A rectangular box has faces of areas 48, 180 and 240. Find the sum of its three edge lengths.", L(6 + 8 + 30))
    A(h, r"A rectangular box has faces of areas 48, 180 and 240. Find its volume.", L(sp.sqrt(48 * 180 * 240)))
    h = "omni-2215"
    A(h, r"$P(x)$ is a monic polynomial of degree 4 with $P(1)=2, P(2)=4, P(3)=6$, and $P(4)=8$. Find $P(5)$.", L(24 + 10))
    A(h, r"$P(x)=x^{4}-10 x^{3}+a x^{2}+b x+c$ satisfies $P(1)=3, P(2)=5$, and $P(3)=7$. Find $P(0)$.", L((0 - 1) * (0 - 2) * (0 - 3) * (0 - 4) + 1))
    h = "omni-2207"
    A(h, r"How many functions $f:\{1,2,3\} \rightarrow\{1,2,3,4\}$ satisfy $f(j)<f(i)+j-i$ for all integers $1 \leq i<j \leq 3$?", L(sum(1 for f in product(range(1, 5), repeat=3) if all(f[j] < f[i] + j - i for i in range(3) for j in range(i + 1, 3)))))
    A(h, r"How many functions $f:\{1,2\} \rightarrow\{1,2,3,4,5\}$ satisfy $f(2)<f(1)+1$?", L(sum(1 for a, b in product(range(1, 6), repeat=2) if b < a + 1)))
    h = "omni-1689"
    for a, b, t in ((13, 17, 1000), (9, 11, 1500), (11, 13, 2000)):
        A(h, rf"How many pairs $(n, k)$ of nonnegative integers satisfy ${a} n+{b} k={t}$?", L(count_sol(a, b, t)))
    h = "omni-2600"
    A(h, r"An isosceles triangle has base 1 and area $\frac{1}{2}$. Find the cosine of its apex angle.", frac(Fraction(1 - Fraction(1, 4), 1 + Fraction(1, 4))))
    A(h, r"An isosceles triangle has base 1 and area $\frac{1}{2}$. Find the length of each leg.", L(sp.sqrt(1 + sp.Rational(1, 4))))
    h = "omni-1346"
    A(h, r"Groups of people of sizes between 1 and 4 fill exactly 6 counter seats. For a group of size $s$ we score $s+1$ points. What is the minimum possible total score?", L(min_score(6)))
    A(h, r"Groups of people of sizes between 1 and 4 fill exactly 7 counter seats. For a group of size $s$ we score $s+1$ points. What is the minimum possible total score?", L(min_score(7)))
    h = "omni-1728"
    A(h, r"The exterior angles of a convex 25-gon form an arithmetic progression with common difference $1^{\circ}$. Find the smallest exterior angle, in degrees.", frac(Fraction(360 - 300, 25)))
    A(h, r"The exterior angles of a convex 26-gon form an arithmetic progression with common difference $1^{\circ}$. Find the smallest exterior angle, in degrees.", frac(Fraction(360 - 325, 26)))
    h = "omni-1900"
    Rr = r"A sequence satisfies $a_{0}=0, a_{1}=1, a_{2}=%d$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n \geq 1$. Find $a_{7}$."
    for a2 in (3, 4):
        A(h, Rr % a2, L(tan_seq(a2, 7)))
    h = "omni-3709"
    T = r"Let $P(n)$ be the number of permutations $(a_{1}, \ldots, a_{n})$ of $\{1, \ldots, n\}$ for which $k a_{k}$ is a perfect square for all $k$. Find $P(%d)$."
    for n in (8, 16):
        A(h, T % n, L(P_square(n)))
    A(h, r"How many positive integers $k \leq 50$ are there such that $2 k$ is a perfect square?", L(sum(1 for k in range(1, 51) if int((2 * k) ** 0.5) ** 2 == 2 * k)))
    h = "omni-1391"
    A(h, r"How many real numbers $x$ with $0 \leq x<2 \pi$ satisfy $\cos ^{2} x+\sin ^{4} x=\frac{7}{8}$?", L(8))
    A(h, r"Find the minimum value of $\left(1+\cos ^{2} x+\sin ^{4} x\right)\left(1+\sin ^{2} x+\cos ^{4} x\right)$ over all real $x$.", frac(Fraction(7, 4) ** 2))
    h = "omni-0166"
    A(h, r"Two spheres of radius 1 have centers at distance $\sqrt{2}$. Find the radius of the circle in which they intersect.", L(sp.sqrt(sp.Rational(1, 2))))
    A(h, r"A rhombus has side length 1 and an angle of $60^{\circ}$. Find the sum of the lengths of its two diagonals.", L(1 + sp.sqrt(3)))
    h = "omni-1488"
    A(h, r"How many distinct real solutions does $x^{3}-3 x=\frac{3}{2}$ have?", L(len(sp.Poly(2 * x ** 3 - 6 * x - 3, x).real_roots())))
    A(h, r"Find the local maximum value of $f(x)=x^{3}-12 x$.", L(16))
    h = "omni-3926"
    D = r"For a positive integer $k$ let $f(k)$ be the square of the sum of the digits of $k$. "
    A(h, D + r"Compute $f\left(f\left(2^{10}\right)\right)$.", L(digit_iter(2 ** 10, 2, sq_digit_sum)))
    A(h, D + r"Find the smallest positive integer $k$ with $f(k)=289$.", L(next(k for k in range(1, 1000) if sq_digit_sum(k) == 289)))
    return R, []
