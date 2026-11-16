"""Companion candidates for training problems T16-T23 (same rules as batch1.py)."""

from fractions import Fraction
from itertools import product
from math import comb

import sympy as sp

from common import L, frac

x = sp.Symbol("x")


def pair_sum_product(poly):
    """prod_{i<j} (r_i + r_j) over the roots of a monic integer polynomial (an integer)."""
    rs = sp.Poly(poly, x).nroots(n=60, maxsteps=200)
    k = sp.prod([rs[i] + rs[j] for i in range(len(rs)) for j in range(i + 1, len(rs))])
    v = sp.re(sp.N(k, 50))
    assert abs(sp.im(sp.N(k, 50))) < 1e-30 and abs(v - round(v)) < 1e-30
    return int(round(v))


def full_pair_product(poly):  # prod over all ordered (i, j), i = j included
    rs = sp.Poly(poly, x).nroots(n=60, maxsteps=200)
    v = sp.N(sp.prod([a + b for a in rs for b in rs]), 50)
    assert abs(sp.im(v)) < 1e-30
    return int(round(sp.re(v)))


def prism_third_faces(big, small):
    """Integer sides s1<=s2<=s3, s1 s2 = small, s2 s3 = big: possible third faces s1 s3 (not small/big)."""
    out = set()
    for s1 in range(1, small + 1):
        for s2 in range(s1, small + 1):
            if s1 * s2 == small and big % s2 == 0 and big // s2 >= s2:
                t = s1 * (big // s2)
                if t not in (small, big):
                    out.add(t)
    return out


def solve_poly(coeffs_known, unknowns, conds, at):
    sol = sp.solve([sp.Eq(coeffs_known.subs(x, a), b) for a, b in conds], unknowns, dict=True)[0]
    return sp.nsimplify(coeffs_known.subs(sol).subs(x, at))


def pisano(m):
    a, b, k = 0, 1, 0
    while True:
        a, b, k = b, (a + b) % m, k + 1
        if (a, b) == (0, 1):
            return k


def fib_mod(n, m):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % m
    return a


def count_funcs(n, ok):
    return sum(1 for f in product(range(1, n + 1), repeat=n)
               if all(ok(f[i - 1], f[j - 1], i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)))


def count_pairs(a, b, total, pos=False):  # nonnegative (or positive) n, k with a n + b k = total: number of n values
    lo = 1 if pos else 0
    return sum(1 for n in range(lo, total // a + 1) if (total - a * n) % b == 0 and (total - a * n) // b >= lo)


def octagon_sin(k):
    """Equilateral octagon, alternate equal angles, area = k [ACEG]; sin B as a Fraction (ACEG side 1)."""
    h = Fraction(k - 1, 2)  # each isosceles triangle on a side of the unit square has area (k-1)/4
    t = Fraction(1, 2) / h  # tan of half the apex angle
    return 2 * t / (1 + t * t)


def build():
    R, C = [], []
    # ---- T16 omni-2543: K^2 = a^(n-1) for x^n - a, n odd ----
    h = "omni-2543"
    assert pair_sum_product(x ** 7 - 7) ** 2 == 117649
    R += [(h, r"Let $r_{1}, \ldots, r_{5}$ be the complex roots of $x^{5}-2$. Compute $\prod_{i=1}^{5}\left(1+r_{i}\right)$.", L(round(sp.re(sp.N(sp.prod([1 + r for r in sp.Poly(x ** 5 - 2, x).nroots(n=40)]), 30))))),
          (h, r"Let $r_{1}, \ldots, r_{7}$ be the complex roots of $x^{7}-7$. Compute $r_{1} r_{2} \cdots r_{7}$.", L(7)),
          (h, r"Let $r_{1}, r_{2}, r_{3}$ be the complex roots of $x^{3}-2$. Compute $\prod_{i=1}^{3} \prod_{j=1}^{3}\left(r_{i}+r_{j}\right)$, the product over all ordered pairs $(i, j)$, including $i=j$.", L(full_pair_product(x ** 3 - 2))),
          (h, r"Let $r_{1}, \ldots, r_{5}$ be the distinct complex roots of $x^{5}-2$, and let $K=\prod_{1 \leq i<j \leq 5}\left(r_{i}+r_{j}\right)$. Determine $K^{2}$.", L(pair_sum_product(x ** 5 - 2) ** 2))]
    T16 = r"Let $r_{{1}}, \ldots, r_{{{n}}}$ be the distinct complex roots of the polynomial ${p}$. Let $K=\prod_{{1 \leq i<j \leq {n}}}\left(r_{{i}}+r_{{j}}\right)$. Determine the value of $K^{{2}}$."
    C += [(h, T16.format(n=6, p="x^{6}-7"), L(pair_sum_product(x ** 6 - 7) ** 2)),
          (h, T16.format(n=3, p="x^{3}-2 x-5"), L(pair_sum_product(x ** 3 - 2 * x - 5) ** 2)),
          (h, T16.format(n=4, p="x^{4}-4 x+2"), L(pair_sum_product(x ** 4 - 4 * x + 2) ** 2)),
          (h, T16.format(n=5, p="x^{5}-x-1"), L(pair_sum_product(x ** 5 - x - 1) ** 2))]
    # ---- T17 omni-2584: integer prism, faces 240 and 48 ----
    h = "omni-2584"
    assert sum(prism_third_faces(240, 48)) == 260
    R += [(h, r"How many ordered pairs $(s, t)$ of positive integers with $s \leq t$ satisfy $s t=48$?", L(sum(1 for s in range(1, 49) for t in range(s, 49) if s * t == 48))),
          (h, r"A rectangular box has faces of areas 48, 240 and 80. Find its volume.", L(sp.sqrt(48 * 240 * 80))),
          (h, r"How many triples $(s_{1}, s_{2}, s_{3})$ of positive integers with $s_{1} \leq s_{2} \leq s_{3}$ satisfy $s_{1} s_{2}=48$ and $s_{2} s_{3}=240$?", L(sum(1 for a in range(1, 49) for b in range(a, 49) if a * b == 48 and 240 % b == 0 and 240 // b >= b))),
          (h, r"The faces of a rectangular box have areas 24, 32 and 48. Find the sum of the three edge lengths of the box.", L(sum(sp.sqrt(24 * 32 * 48) / f for f in (24, 32, 48))))]
    T17 = r"Let $V$ be a rectangular prism with integer side lengths. The largest face has area {b} and the smallest face has area {s}. A third face has area $x$, where $x$ is not equal to {s} or {b}. What is the sum of all possible values of $x$?"
    C += [(h, r"Let $V$ be a rectangular prism with two faces of areas 48 and 240 and volume 960. Find the area of the third face.", L(Fraction(960 ** 2, 48 * 240))),
          (h, T17.format(b=60, s=12), L(sum(prism_third_faces(60, 12)))),
          (h, r"The areas of three faces of a rectangular box are 6, 8 and 12. Find the volume of the box.", L(sp.sqrt(6 * 8 * 12))),
          (h, T17.format(b=120, s=20), L(sum(prism_third_faces(120, 20))))]
    # ---- T18 omni-2215: Q = P - x - 6 has roots 5, 11, 17 ----
    h = "omni-2215"
    a, b, c, d = sp.symbols("a b c d")
    P4 = x ** 4 - 29 * x ** 3 + a * x ** 2 + b * x + c
    assert solve_poly(P4, [a, b, c], [(5, 11), (11, 17), (17, 23)], 1) == -3193
    R += [(h, r"A monic polynomial $Q$ of degree 4 has roots 5, 11 and 17, and the coefficient of $x^{3}$ in $Q(x)$ is $-29$. Find the fourth root of $Q$.", L(29 - 5 - 11 - 17)),
          (h, r"Compute $(1-5)(1-11)(1-17)(1+4)$.", L((1 - 5) * (1 - 11) * (1 - 17) * (1 + 4))),
          (h, r"Let $P(x)=x^{3}+a x^{2}+b x+c$ with $P(1)=2, P(2)=3$, and $P(3)=4$. Find $P(4)$.", L(solve_poly(x ** 3 + a * x ** 2 + b * x + c, [a, b, c], [(1, 2), (2, 3), (3, 4)], 4))),
          (h, r"Find the sum of the coefficients of the polynomial $P(x)=(x-2)(x-3)(x+1)(x-4)+x$.", L(((x - 2) * (x - 3) * (x + 1) * (x - 4) + x).subs(x, 1)))]
    T18 = r"Find the sum of the coefficients of the polynomial $P(x)=x^{{4}}-29 x^{{3}}+a x^{{2}}+b x+c$, given that $P(5)=11, P(11)=17$, and $P(17)={v}$."
    C += [(h, T18.format(v=25), L(solve_poly(P4, [a, b, c], [(5, 11), (11, 17), (17, 25)], 1))),
          (h, r"Let $P(x)=x^{4}+a x^{3}+b x^{2}+c x+d$ with $P(1)=2, P(2)=3, P(3)=4$, and $P(4)=6$. Find $P(0)$.", L(solve_poly(x ** 4 + a * x ** 3 + b * x ** 2 + c * x + d, [a, b, c, d], [(1, 2), (2, 3), (3, 4), (4, 6)], 0))),
          (h, r"Find the sum of the coefficients of the polynomial $P(x)=x^{3}+a x^{2}+b x+c$, given that $P(5)=11, P(11)=17$, and $P(17)=24$.", L(solve_poly(x ** 3 + a * x ** 2 + b * x + c, [a, b, c], [(5, 11), (11, 17), (17, 24)], 1))),
          (h, T18.format(v=29), L(solve_poly(P4, [a, b, c], [(5, 11), (11, 17), (17, 29)], 1)))]
    # ---- T19 omni-2546: power of a point, QAD ~ QCB ----
    h = "omni-2546"
    qa = sp.solve(sp.Eq(x * (x + 10), 144), x)
    qc = sp.solve(sp.Eq(x * (x + 7), 144), x)
    QA, QC = max(qa), max(qc)
    QB, QD = QA + 10, QC + 7
    R += [(h, r"A line through a point $Q$ meets a circle at $A$ and $B$, with $Q A<Q B$ and $A B=10$. The tangent segment from $Q$ to the circle has length 12. Find $Q A$.", L(QA)),
          (h, r"A line through a point $Q$ meets a circle at $C$ and $D$, with $Q C<Q D$ and $C D=7$. The tangent segment from $Q$ to the circle has length 12. Find $Q C$.", L(QC)),
          (h, r"Points $A$ and $B$ lie on one ray from $Q$, and points $C$ and $D$ lie on another ray from $Q$, with $Q A=4, Q B=18, Q C=6$, and $Q D=12$. Find $\frac{A D}{B C}$.", frac(Fraction(4, 6))),
          (h, r"The tangent segment from a point $Q$ to a circle has length 6. A line through $Q$ meets the circle at $A$ and $B$ with $Q A=4$ and $Q A<Q B$. Find $A B$.", frac(Fraction(36, 4) - 4))]
    T19 = r"Let $P$ and $Q$ be points on line $l$ with $P Q=12$. Two circles, $\omega$ and $\Omega$, are both tangent to $l$ at $P$ and are externally tangent to each other. A line through $Q$ intersects $\omega$ at $A$ and $B$, with $A$ closer to $Q$ than $B$, such that $A B=10$. Similarly, another line through $Q$ intersects $\Omega$ at $C$ and $D$, with $C$ closer to $Q$ than $D$, such that $C D=7$. "
    C += [(h, T19 + r"Find the ratio $Q B / Q D$.", L(QB / QD)),
          (h, T19 + r"Find $Q B-Q D$.", L(QB - QD)),
          (h, T19 + r"If the two lines through $Q$ are perpendicular, find the ratio $A C / B D$.", L(sp.sqrt(QA ** 2 + QC ** 2) / sp.sqrt(QB ** 2 + QD ** 2))),
          (h, T19 + r"If the two lines through $Q$ are perpendicular, find $A D$.", L(sp.sqrt(QA ** 2 + QD ** 2)))]
    # ---- T20 omni-1362: Fibonacci period mod 127 is 2p+2 = 256 ----
    h = "omni-1362"
    assert pisano(127) == 256
    R += [(h, r"The Fibonacci sequence is defined by $F_{0}=0, F_{1}=1$, and $F_{n}=F_{n-1}+F_{n-2}$ for $n \geq 2$. Find the smallest positive integer $m$ such that $F_{m} \equiv 0(\bmod 7)$ and $F_{m+1} \equiv 1(\bmod 7)$.", L(pisano(7))),
          (h, r"The Fibonacci sequence is defined by $F_{0}=0, F_{1}=1$, and $F_{n}=F_{n-1}+F_{n-2}$ for $n \geq 2$. Find the smallest positive integer $m$ such that $F_{m} \equiv 0(\bmod 3)$ and $F_{m+1} \equiv 1(\bmod 3)$.", L(pisano(3))),
          (h, r"The Fibonacci sequence is defined by $F_{0}=0, F_{1}=1$, and $F_{n}=F_{n-1}+F_{n-2}$ for $n \geq 2$. Find the remainder when $F_{127}$ is divided by 127.", L(fib_mod(127, 127))),
          (h, r"The Fibonacci sequence is defined by $F_{0}=0, F_{1}=1$, and $F_{n}=F_{n-1}+F_{n-2}$ for $n \geq 2$. Find the remainder when $F_{13}$ is divided by 13.", L(fib_mod(13, 13)))]
    T20 = r"The Fibonacci sequence is defined as follows: $F_{{0}}=0, F_{{1}}=1$, and $F_{{n}}=F_{{n-1}}+F_{{n-2}}$ for all integers $n \geq 2$. Find the smallest positive integer $m$ such that $F_{{m}} \equiv 0(\bmod {p})$ and $F_{{m+1}} \equiv 1(\bmod {p})$."
    C += [(h, T20.format(p=p), L(pisano(p))) for p in (11, 29, 47, 61)]
    # ---- T21 omni-2207: g(i) = f(i) - i strictly decreasing ----
    h = "omni-2207"
    assert count_funcs(5, lambda fi, fj, i, j: fj < fi + j - i) == comb(9, 5)
    R += [(h, r"How many sequences of integers $g_{1}>g_{2}>g_{3}$ have every term in $\{-2,-1,0,1,2\}$?", L(comb(5, 3))),
          (h, r"How many functions $f:\{1,2,3,4\} \rightarrow\{1,2,3,4\}$ satisfy $f(j)<f(i)+j-i$ for all integers $i, j$ such that $1 \leq i<j \leq 4$?", L(count_funcs(4, lambda fi, fj, i, j: fj < fi + j - i))),
          (h, r"How many strictly increasing functions $f:\{1,2,3\} \rightarrow\{1,2, \ldots, 8\}$ are there?", L(comb(8, 3))),
          (h, r"Let $f:\{1,2,3,4,5\} \rightarrow\{1,2,3,4,5\}$ be any function and let $g(i)=f(i)-i$. How many different values can $g(3)$ take?", L(len({v - 3 for v in range(1, 6)})))]
    T21 = r"How many functions $f:\{{1,2, \ldots, {n}\}} \rightarrow\{{1,2, \ldots, {n}\}}$ satisfy {cond} for all integers $i, j$ such that $1 \leq i<j \leq {n}$?"
    C += [(h, T21.format(n=5, cond=r"$f(j) \leq f(i)+j-i$"), L(count_funcs(5, lambda fi, fj, i, j: fj <= fi + j - i))),
          (h, T21.format(n=4, cond=r"$f(j) \leq f(i)+j-i$"), L(count_funcs(4, lambda fi, fj, i, j: fj <= fi + j - i))),
          (h, T21.format(n=5, cond=r"$f(j) \leq f(i)+2(j-i)$"), L(count_funcs(5, lambda fi, fj, i, j: fj <= fi + 2 * (j - i)))),
          (h, T21.format(n=5, cond=r"$|f(j)-f(i)|<j-i$"), L(count_funcs(5, lambda fi, fj, i, j: abs(fj - fi) < j - i)))]
    # ---- T22 omni-1689: 9n + 11k = 2007 ----
    h = "omni-1689"
    assert count_pairs(9, 11, 2007) == 21
    R += [(h, r"How many pairs $(n, k)$ of nonnegative integers satisfy $9 n+11 k=198$?", L(count_pairs(9, 11, 198))),
          (h, r"Find the smallest nonnegative integer $n$ such that $2007-9 n$ is divisible by 11.", L(next(n for n in range(11) if (2007 - 9 * n) % 11 == 0))),
          (h, r"How many integers $n$ with $0 \leq n \leq 100$ satisfy $n \equiv 3(\bmod 11)$?", L(sum(1 for n in range(101) if n % 11 == 3))),
          (h, r"A student was counting his stones by 7. He messed up $n$ times and instead counted 5s and wound up at 100. How many values of $n$ are possible?", L(count_pairs(5, 7, 100)))]
    T22 = r"A student named Kevin was counting his stones by {a}. He messed up $n$ times and instead counted {b}s and wound up at {t}. How many values of $n$ are possible?"
    C += [(h, r"A student named Kevin was counting his stones by 11. He messed up $n$ times and instead counted 9s and wound up at 2007. He counted at least one 11 and messed up at least once. How many values of $n$ are possible?", L(count_pairs(9, 11, 2007, pos=True))),
          (h, T22.format(a=11, b=12, t=2007), L(count_pairs(12, 11, 2007))),
          (h, T22.format(a=10, b=9, t=2007), L(count_pairs(9, 10, 2007))),
          (h, T22.format(a=11, b=9, t=2008), L(count_pairs(9, 11, 2008)))]
    # ---- T23 omni-2600: octagon = unit square ACEG + four isosceles triangles ----
    h = "omni-2600"
    assert octagon_sin(3) == Fraction(4, 5)
    R += [(h, r"An isosceles triangle has base 1 and area $\frac{1}{2}$. Its apex angle is $2 \theta$. Find $\tan \theta$.", frac(Fraction(1, 2) / 1)),
          (h, r"If $\tan \theta=\frac{1}{2}$, find $\sin 2 \theta$.", frac(Fraction(2 * Fraction(1, 2), 1 + Fraction(1, 4)))),
          (h, r"Let $A B C D E F G H$ be an equilateral octagon such that $A C E G$ is a square of side 1 and $\angle B=\angle D=\angle F=\angle H=90^{\circ}$, with $B, D, F, H$ outside the square. Find the area of the octagon.", L(1 + 4 * Fraction(1, 4))),
          (h, r"An isosceles triangle has base 2 and area 3. Find the sine of its apex angle.", frac(octagon_sin(4)))]
    T23 = r"Let $A B C D E F G H$ be an equilateral octagon with $\angle A \cong \angle C \cong \angle E \cong \angle G$ and $\angle B \cong \angle D \cong \angle F \cong \angle H$. If the area of $A B C D E F G H$ is {k} times the area of $A C E G$, then $\sin B$ can be written as $\frac{{m}}{{n}}$, where $m, n$ are positive integers and $\operatorname{{gcd}}(m, n)=1$. Find $100 m+n$."
    C += [(h, T23.format(k="two"), L(100 * octagon_sin(2).numerator + octagon_sin(2).denominator)),
          (h, T23.format(k=r"$\frac{5}{2}$"), L(100 * octagon_sin(Fraction(5, 2)).numerator + octagon_sin(Fraction(5, 2)).denominator)),
          (h, r"An equiangular octagon has sides that alternate in length between 1 and $\sqrt{2}$. Find its area.", L((1 + sp.sqrt(2) * sp.sqrt(2)) ** 2 - sp.sqrt(2) ** 2)),
          (h, T23.format(k="five"), L(100 * octagon_sin(5).numerator + octagon_sin(5).denominator))]
    return R, C
