"""Round 2 of RELATED candidates (2026-11-16), calibrated on round-1 screening.

Round 1 had 20/128 related candidates in band (75 above, 33 below). For each training problem with
fewer than 2 in band, this batch adds key steps pitched between the round-1 items that scored 16/16
and those that scored 0-3/16: harder steps where round 1 was easy, easier ones where it was hard,
and several points along a difficulty knob where one exists. Answers are computed here.
Returns (related, []) — conflicting round 2 is in batch6.py.
"""

from fractions import Fraction
from itertools import product
from math import ceil, comb, floor, gcd

import sympy as sp

from batch1 import first_m_without_integer_log, linrec, prob_pattern_first, v2, fib
from batch2 import bishop, placements, start_money
from batch4 import P_square, cafe, digit_iter, is_sq, sq_digit_sum
from common import L, frac

x = sp.Symbol("x")


def coprime_count(N, n):
    return sum(1 for m in range(1, N + 1) if gcd(m, n) == 1)


def phi_k(n, k):
    return sum(1 for m in range(1, n // k + 1) if gcd(m, n) == 1)


def circle_xy(a, b):
    """A=(a,1/a), B=(b,1/b) on xy=1; the circle with diameter AB meets xy=1 again at X, Y. Return XY."""
    a, b = sp.Rational(a), sp.Rational(b)
    poly = sp.expand(((x - a) * (x - b) * x ** 2 + (1 - x / a) * (1 - x / b)) * a * b)
    roots = [r for r in sp.solve(poly, x) if sp.simplify(r - a) != 0 and sp.simplify(r - b) != 0]
    r1, r2 = roots
    return sp.simplify(sp.sqrt((r1 - r2) ** 2 + (1 / r1 - 1 / r2) ** 2))


def count_sol(a, b, t):
    return sum(1 for n in range(t // a + 1) if (t - a * n) % b == 0)


def tan_seq(a2, upto):
    seq = [sp.Integer(0), sp.Integer(1), sp.Rational(a2)]
    for n in range(1, upto - 1):
        seq.append(sp.simplify((seq[n] + seq[n - 1]) / (seq[n] * seq[n - 1] - 1)))
    return seq


def iso_sine(base, area):  # sine of the apex angle of an isosceles triangle
    h = Fraction(2 * area, 1) / base
    t = Fraction(base, 2) / h
    return 2 * t / (1 + t * t)


def build():
    R = []
    A = lambda h, p, a: R.append((h, p, a))
    # T0 omni-0793 (round 1: 14,16,16,2) — inclusion-exclusion counts between the two extremes
    h = "omni-0793"
    A(h, r"How many positive integers $m \leq 100$ are relatively prime to 105?", L(coprime_count(100, 105)))
    A(h, r"How many positive integers $m \leq 1000$ are relatively prime to 2001?", L(coprime_count(1000, 2001)))
    A(h, r"Compute $\varphi(3 \cdot 7 \cdot 11 \cdot 13)$, where $\varphi$ is Euler's totient function.", L(sp.totient(3 * 7 * 11 * 13)))
    A(h, r"Define $\varphi^{k}(n)$ as the number of positive integers that are at most $n / k$ and relatively prime to $n$. Compute $\varphi^{3}(1155)$.", L(phi_k(1155, 3)))
    A(h, r"Compute $\varphi\left(2002^{2}-1\right)$, given that 2003 is prime.", L(sp.totient(2002 ** 2 - 1)))
    # T1 omni-1735 (all 15-16) — harder steps: the e-limit and the binomial sum
    h = "omni-1735"
    A(h, r"Compute $\sum_{k=0}^{5} \binom{5}{k} \frac{1}{5^{k}}$.", frac(sum(Fraction(comb(5, k), 5 ** k) for k in range(6))))
    A(h, r"To the nearest integer, what is $100\left(1+\frac{1}{50}\right)^{50}$?", L(round(100 * (1 + 1 / 50) ** 50)))
    A(h, r"To the nearest integer, what is $1000\left(1+\frac{1}{1000}\right)^{1000}$?", L(round(1000 * (1 + 1 / 1000) ** 1000)))
    A(h, r"Let $p_{k}=\binom{10}{k} / 10^{k}$. Compute $p_{0}+p_{1}+\cdots+p_{10}$.", frac(sum(Fraction(comb(10, k), 10 ** k) for k in range(11))))
    # T2 omni-1814 (16,10,16,16) — the full perpendicularity step on other points
    h = "omni-1814"
    A(h, r"The points $A=\left(2, \frac{1}{2}\right)$ and $B=\left(-3,-\frac{1}{3}\right)$ lie on the hyperbola $x y=1$. The circle with diameter $A B$ intersects this hyperbola again at points $X$ and $Y$. Compute $X Y$.", L(circle_xy(2, -3)))
    A(h, r"Let $A=(1,1)$ and $B=\left(-4,-\frac{1}{4}\right)$. Find the positive number $x$ such that the point $X=\left(x, \frac{1}{x}\right)$ satisfies $\angle A X B=90^{\circ}$.", L(sp.sqrt(sp.Rational(1, 4))))
    A(h, r"Points $A=\left(3, \frac{1}{3}\right)$, $B=\left(-2,-\frac{1}{2}\right)$ and $X=(1,1)$ lie on $x y=1$. Find the product of the slopes of lines $A X$ and $B X$.", L(sp.Rational(-1, 3) * sp.Rational(1, 2)))
    # T3 omni-2140 (13,16,16,10)
    h = "omni-2140"
    A(h, r"A hexagon has its vertices at distances alternately 1 and 2 from a point $O$, and consecutive vertices subtend $60^{\circ}$ at $O$. Find the area of the hexagon.", L(6 * sp.Rational(1, 2) * 1 * 2 * sp.sin(sp.pi / 3)))
    A(h, r"The polynomial $x^{6}-19 x^{3}-216$ factors as $\left(x^{3}-a\right)\left(x^{3}+b\right)$ with $a, b>0$. Find $a$.", L(27))
    # T4 omni-2430 (16,16,16,12) — m_b for specific b (window of 2018 numbers)
    h = "omni-2430"
    D = r"For an integer $b \geq 2$, let $m_{b}$ be the least positive integer $m$ for which none of $\log _{b}(m), \log _{b}(m+1), \ldots, \log _{b}(m+2017)$ is an integer. Find $m_{%d}$."
    for b in (4, 10, 45, 46):
        A(h, D % b, L(first_m_without_integer_log(b, 2017)))
    # T5 omni-0962 (2,2,1,0) — easier pattern questions
    h = "omni-0962"
    A(h, r"A fair coin is flipped repeatedly. What is the expected number of flips until two heads in a row appear for the first time?", L(6))
    A(h, r"A fair coin is flipped repeatedly (H for heads, T for tails). What is the expected number of flips until the pattern HTH appears for the first time?", L(10))
    A(h, r"A fair coin is flipped repeatedly (H for heads, T for tails). What is the probability that the pattern HT appears before the pattern TT?", L(prob_pattern_first("10", "00")))
    A(h, r"A fair coin is flipped repeatedly (H for heads, T for tails). What is the probability that the pattern HHT appears before the pattern HTT?", L(prob_pattern_first("110", "100")))
    # T6 omni-0528 (14,16,16,7)
    h = "omni-0528"
    A(h, r"Let $a_{0}=1, a_{1}=4$, and $a_{i}=4 a_{i-1}+16 a_{i-2}$ for $i \geq 2$. Find the largest integer $n$ such that $2^{n}$ divides $a_{10}$.", L(v2(linrec(1, 4, 4, 16, 10))))
    A(h, r"Let $F_{1}=F_{2}=1$ and $F_{n}=F_{n-1}+F_{n-2}$. Find the largest integer $n$ such that $2^{n}$ divides $F_{96}$.", L(v2(fib(96))))
    # T7 omni-0596 (all 15-16) — extremes of the distance expressions
    h = "omni-0596"
    T = r"Triangle $A B C$ has $A B=6, B C=8, A C=10$. For a point $P$ inside or on the triangle, let $d(P, \ell)$ be its distance to side $\ell$. "
    a_, b_ = sp.symbols("a b")
    dAC = (24 - 3 * a_ - 4 * b_) / 5
    verts = [(0, 0), (8, 0), (0, 6)]  # (a, b) = (d to AB, d to BC) at B, C, A
    A(h, T + r"Find the maximum of $d(P, A C)-d(P, A B)$.", L(max((dAC - a_).subs({a_: p, b_: q}) for p, q in verts)))
    A(h, T + r"Find the minimum of $d(P, A C)-d(P, A B)$.", L(min((dAC - a_).subs({a_: p, b_: q}) for p, q in verts)))
    A(h, T + r"Find the maximum of $d(P, A B)+d(P, B C)+d(P, A C)$.", L(max((a_ + b_ + dAC).subs({a_: p, b_: q}) for p, q in verts)))
    # T8 omni-1877 (6,15,16,15)
    h = "omni-1877"
    A(h, r"In triangle $K A C$, $K A=60$, $K C=54$ and $A C=38$. Points $X$ on $K A$ and $Y$ on $K C$ satisfy $A X=20$ and $C Y=18$. Find $X Y$.", frac(Fraction(38 * 40, 60)))
    A(h, r"In triangle $K A C$, $A C=38$, $K A=20 t$ and $K C=18 t$ for some $t>1$. Points $X$ on $K A$ and $Y$ on $K C$ satisfy $A X=20$ and $C Y=18$, so $X Y \parallel A C$. Find $X Y$ when $t=10$.", frac(Fraction(38 * 9, 10)))
    # T9 omni-0757 (9,2,0,0) — easier bishop facts
    h = "omni-0757"
    A(h, r"What is the maximum number of bishops that can be placed on a $4 \times 4$ chessboard so that no two attack each other?", L(placements(4, 4, bishop)[0]))
    A(h, r"What is the maximum number of bishops that can be placed on a $5 \times 5$ chessboard so that no two attack each other?", L(placements(5, 5, bishop)[0]))
    A(h, r"In how many ways can 4 bishops be placed on a $3 \times 3$ chessboard so that no two attack each other?", L(placements(3, 3, bishop)[1]))
    # T10 omni-1299 (16,3,0,4) — convex-boundary counts between the extremes
    h = "omni-1299"
    A(h, r"What is the maximum number of points in which a ray can intersect the boundary of a square, if the ray contains no side of the square?", L(2))
    A(h, r"What is the maximum number of points in which the boundary of a convex quadrilateral and the boundary of a triangle can intersect, if no two sides overlap?", L(6))
    A(h, r"What is the maximum number of points in which the boundaries of two convex pentagons can intersect, if no two sides overlap?", L(10))
    # T11 omni-0848 (16,5,3,0)
    h = "omni-0848"
    A(h, r"Point $M$ satisfies $M A=3$, $M B=4$ and $\angle A M B=60^{\circ}$. Find the area of triangle $A M B$.", L(sp.Rational(1, 2) * 12 * sp.sin(sp.pi / 3)))
    A(h, r"Point $M$ satisfies $M A=2$, $M B=3$, $M C=4$ and $\angle A M B=\angle B M C=\angle C M A=120^{\circ}$. Find the area of triangle $A B C$.", L(sp.sin(2 * sp.pi / 3) / 2 * (6 + 12 + 8)))
    # T12 omni-1590 (13,3,1,8) — small-range versions
    h = "omni-1590"
    A(h, r"For how many integers $n$ with $0 \leq n<20$ is $\left\lceil\frac{3 n}{4}\right\rceil-\left\lfloor\frac{4 n}{5}\right\rfloor=1$?", L(sum(1 for n in range(20) if ceil(Fraction(3 * n, 4)) - floor(Fraction(4 * n, 5)) == 1)))
    A(h, r"For how many integers $n$ with $0 \leq n<6$ is $1+\left\lfloor\frac{n}{3}\right\rfloor=\left\lceil\frac{n}{2}\right\rceil$?", L(sum(1 for n in range(6) if 1 + floor(Fraction(n, 3)) == ceil(Fraction(n, 2)))))
    # T13 omni-0912 (5,0,1,4) — easier subset-pair counts
    h = "omni-0912"
    S2 = [frozenset(s) for s in ([], [1], [2], [1, 2])]
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2\}$ satisfy $X \subseteq Y$?", L(sum(1 for X, Y in product(S2, repeat=2) if X <= Y)))
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2\}$ satisfy that $X$ is a proper subset of $Y$?", L(sum(1 for X, Y in product(S2, repeat=2) if X < Y)))
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2\}$ are there such that neither is a proper subset of the other?", L(sum(1 for X, Y in product(S2, repeat=2) if not (X < Y or Y < X))))
    # T14 omni-1315 (16,1,8,3)
    h = "omni-1315"
    S14 = r"Stacy has $d$ dollars. She enters a mall with %d shops and a lottery stall. First she plays the lottery and her money is doubled, then she goes into the first shop and spends %d dollars. After that she alternates playing the lottery (her money is doubled) and going into a new shop and spending %d dollars. When she comes out of the last shop she has no money left. What is the minimum possible value of $d$?"
    for n, s in ((4, 1024), (5, 1024), (2, 100)):
        A(h, S14 % (n, s, s), frac(start_money(n, s)))
    # T15 omni-2082 (16,15,14,16) — the incircle steps of the solution
    h = "omni-2082"
    T15 = r"Let $A B C$ be a triangle with $A B=13, B C=14$, and $C A=15$, and let $D$ be the foot of the altitude from $A$ to $B C$. "
    A(h, T15 + r"The incircle of triangle $A B D$ touches $B C$ at $X$. Find $D X$.", L(Fraction(12 + 5 - 13, 2)))
    A(h, T15 + r"The incircles of triangles $A B D$ and $A C D$ touch $B C$ at $X$ and $Y$. Find $X Y$.", L(Fraction(12 + 5 - 13, 2) + Fraction(12 + 9 - 15, 2)))
    A(h, T15 + r"The incircle of triangle $A B D$ touches $A D$ at $P$ and $B C$ at $X$. Find $P X$.", L(2 * sp.sqrt(2)))
    # T16 omni-2543 (6,16,3,2)
    h = "omni-2543"
    A(h, r"Let $r_{1}, r_{2}, r_{3}$ be the complex roots of $x^{3}-5$. Compute $\left(2+r_{1}\right)\left(2+r_{2}\right)\left(2+r_{3}\right)$.", L(-((-2) ** 3 - 5)))
    A(h, r"Let $r_{1}, \ldots, r_{7}$ be the complex roots of $x^{7}-7$. Compute $\prod_{i=1}^{7}\left(1+r_{i}\right)$.", L(-((-1) ** 7 - 7)))
    # T17 omni-2584 (16,16,14,15)
    h = "omni-2584"
    trip = [(a, b, 240 // b) for a in range(1, 49) for b in range(a, 49) if a * b == 48 and 240 % b == 0 and 240 // b >= b]
    A(h, r"Over all triples $(s_{1}, s_{2}, s_{3})$ of positive integers with $s_{1} \leq s_{2} \leq s_{3}$, $s_{1} s_{2}=48$ and $s_{2} s_{3}=240$, find the sum of $s_{1}+s_{2}+s_{3}$.", L(sum(sum(t) for t in trip)))
    A(h, r"Over all triples $(s_{1}, s_{2}, s_{3})$ of positive integers with $s_{1} \leq s_{2} \leq s_{3}$, $s_{1} s_{2}=48$ and $s_{2} s_{3}=240$, find the sum of the products $s_{1} s_{2} s_{3}$.", L(sum(a * b * c for a, b, c in trip)))
    # T18 omni-2215 (all 16) — the root-sum step with a hint, and a cubic analogue
    h = "omni-2215"
    A(h, r"$P(x)$ is a monic polynomial of degree 4 whose $x^{3}$ coefficient is $-29$, and $Q(x)=P(x)-x-6$ has roots 5, 11 and 17. Find $P(0)$.", L((0 - 5) * (0 - 11) * (0 - 17) * (0 + 4) + 6))
    A(h, r"$P(x)$ is a monic cubic polynomial with $P(1)=3, P(2)=5$, and $P(3)=7$. Find $P(5)$.", L((5 - 1) * (5 - 2) * (5 - 3) + 2 * 5 + 1))
    # T19 omni-2546 (16,16,9,15)
    h = "omni-2546"
    A(h, r"Points $A, B$ lie on one ray from $Q$ and $C, D$ on another, with $Q A=6, Q B=24, Q C=8$ and $Q D=18$. Find $\frac{A D}{B C}$.", frac(Fraction(6, 8)))
    A(h, r"The tangent segment from a point $Q$ to a circle has length 12. A line through $Q$ meets the circle at $C$ and $D$ with $Q C<Q D$ and $C D=7$. Find $Q D$.", L(16))
    # T21 omni-2207 (16,3,16,16)
    h = "omni-2207"
    A(h, r"How many functions $f:\{1,2,3\} \rightarrow\{1,2,3\}$ satisfy $f(j)<f(i)+j-i$ for all integers $i, j$ with $1 \leq i<j \leq 3$?", L(sum(1 for f in product(range(1, 4), repeat=3) if all(f[j] < f[i] + j - i for i in range(3) for j in range(i + 1, 3)))))
    A(h, r"How many sequences of integers $g_{1}>g_{2}>g_{3}>g_{4}>g_{5}$ have every term in $\{-4,-3, \ldots, 4\}$?", L(comb(9, 5)))
    # T22 omni-1689 (16,16,16,4)
    h = "omni-1689"
    A(h, r"How many pairs $(n, k)$ of nonnegative integers satisfy $9 n+11 k=999$?", L(count_sol(9, 11, 999)))
    A(h, r"How many pairs $(n, k)$ of nonnegative integers satisfy $7 n+11 k=500$?", L(count_sol(7, 11, 500)))
    # T23 omni-2600 (8,16,3,12)
    h = "omni-2600"
    A(h, r"An isosceles triangle has base 1 and area $\frac{3}{8}$. Find the sine of its apex angle.", frac(iso_sine(1, Fraction(3, 8))))
    A(h, r"An isosceles triangle has base 1 and area $\frac{1}{3}$. Find the sine of its apex angle.", frac(iso_sine(1, Fraction(1, 3))))
    # T24 omni-1346 (2,3,1,9) — very small cafes
    h = "omni-1346"
    C24 = r"A cafe has %s and %s. People enter in groups of size between 1 and 4, and groups never share a table. A group of more than 1 sits at a table if one is free, and otherwise in counter seats. A group of 1 sits at the counter if a seat is free, and otherwise at a table. $M$ groups with a total of $N$ people enter and sit down, after which every table and counter seat is occupied. What is the minimum possible value of $M+N$?"
    A(h, C24 % ("1 table", "1 counter seat"), L(cafe(1, 1)))
    A(h, C24 % ("1 table", "2 counter seats"), L(cafe(1, 2)))
    # T25 omni-1728 (16,14,16,7)
    h = "omni-1728"
    A(h, r"The exterior angles of a convex 12-gon form an arithmetic progression with common difference $2^{\circ}$. Find the largest exterior angle, in degrees.", L(Fraction(360 - 66 * 2, 12) + 22))
    A(h, r"The interior angles of a convex hexagon form an arithmetic progression with common difference $4^{\circ}$. Find the largest angle, in degrees.", L(Fraction(720, 6) + Fraction(5, 2) * 4))
    # T26 omni-1900 (all 16) — run the recursion several steps
    h = "omni-1900"
    Rr = r"A sequence satisfies $a_{0}=0, a_{1}=1, a_{2}=%d$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n \geq 1$. Find $a_{%d}$."
    for a2, k in ((2, 7), (3, 6), (4, 5)):
        A(h, Rr % (a2, k), L(tan_seq(a2, k + 1)[k]))
    # T27 omni-3709 (3,16,1,0)
    h = "omni-3709"
    A(h, r"How many perfect squares are there among $1,2, \ldots, 4500$?", L(sum(1 for k in range(1, 4501) if is_sq(k))))
    A(h, r"How many permutations $(a_{1}, \ldots, a_{12})$ of $\{1,2, \ldots, 12\}$ are there such that $k a_{k}$ is a perfect square for all $1 \leq k \leq 12$?", L(P_square(12)))
    A(h, r"Let $P(n)$ be the number of permutations $(a_{1}, \ldots, a_{n})$ of $\{1, \ldots, n\}$ for which $k a_{k}$ is a perfect square for all $k$. Find the smallest $n$ such that $P(n)$ is a multiple of 5.", L(next(n for n in range(1, 200) if P_square(n) % 5 == 0)))
    # T28 omni-1391 (16,16,14,16)
    h = "omni-1391"
    A(h, r"If $\tan x=2$, compute $\cos ^{2} x+\sin ^{4} x$.", frac(Fraction(1, 5) + Fraction(16, 25)))
    A(h, r"Let $m$ and $M$ be the minimum and maximum of $1+\cos ^{2} x+\sin ^{4} x$ over all real $x$. Compute $M-m$.", frac(2 - Fraction(7, 4)))
    A(h, r"An angle $\alpha$ satisfies $\frac{7}{4} \leq \alpha \leq 2$ (in radians) and $\sin \alpha=\frac{12}{13}$. Find $\cos \alpha$.", frac(Fraction(-5, 13)))
    # T31 omni-3926 (15,16,16,16)
    h = "omni-3926"
    D31 = r"For a positive integer $k$ let $f_{1}(k)$ be the square of the sum of the digits of $k$, and let $f_{n+1}(k)=f_{1}\left(f_{n}(k)\right)$. "
    A(h, D31 + r"Compute $f_{1}\left(2^{30}\right)$.", L(sq_digit_sum(2 ** 30)))
    A(h, D31 + r"Compute $f_{3}\left(2^{30}\right)$.", L(digit_iter(2 ** 30, 3, sq_digit_sum)))
    A(h, D31 + r"Find the smallest positive integer $k$ with $f_{1}(k)=169$.", L(next(k for k in range(1, 1000) if sq_digit_sum(k) == 169)))
    return R, []
