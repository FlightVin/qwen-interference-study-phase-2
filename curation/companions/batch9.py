"""Round 4 of RELATED candidates (2026-11-16 ~02:00) for the 11 training problems still short after
round-3 screening (61/250 in band). Most remaining items sit at 12-14/16, so these are pitched a
little harder. Answers are computed here. Returns (related, [])."""

from fractions import Fraction
from itertools import product

import sympy as sp

from batch4 import P_square
from common import L, frac


def count_sol(a, b, t):
    return sum(1 for n in range(t // a + 1) if (t - a * n) % b == 0)


def tan_seq(a2, k):
    s = [sp.Integer(0), sp.Integer(1), sp.Rational(a2)]
    for n in range(1, k - 1):
        s.append(sp.simplify((s[n] + s[n - 1]) / (s[n] * s[n - 1] - 1)))
    return s[k]


def build():
    R = []
    A = lambda h, p, a: R.append((h, p, a))
    h = "omni-1735"
    A(h, r"Write $\left(1+\frac{1}{12}\right)^{12}$ as a fraction in lowest terms.", frac(Fraction(13, 12) ** 12))
    A(h, r"Compute $\sum_{k=0}^{7} \binom{7}{k} \frac{1}{7^{k}}$.", frac(sum(Fraction(sp.binomial(7, k), 7 ** k) for k in range(8))))
    h = "omni-0596"
    T = r"Triangle $A B C$ has $A B=6, B C=8, A C=10$. For a point $P$ inside or on the triangle, let $d(P, \ell)$ be its distance to side $\ell$. "
    a, b = sp.symbols("a b")
    dAC = (24 - 3 * a - 4 * b) / 5
    V = [(0, 0), (8, 0), (0, 6)]
    A(h, T + r"Find the minimum of $d(P, A C)-d(P, B C)$.", L(min((dAC - b).subs({a: p, b: q}) for p, q in V)))
    A(h, T + r"Find the maximum of $2 d(P, A B)+d(P, A C)$.", L(max((2 * a + dAC).subs({a: p, b: q}) for p, q in V)))
    h = "omni-0848"
    A(h, r"Point $M$ satisfies $M A=1$, $M B=2$, $M C=3$ and $\angle A M B=\angle B M C=\angle C M A=120^{\circ}$. Find the area of triangle $A B C$.", L(sp.sin(2 * sp.pi / 3) / 2 * (2 + 6 + 3)))
    A(h, r"Point $M$ satisfies $M A=M B=3$ and $\angle A M B=120^{\circ}$. Find the area of triangle $M A B$.", L(sp.Rational(9, 2) * sp.sin(2 * sp.pi / 3)))
    h = "omni-2082"
    T15 = r"Let $A B C$ be a triangle with $A B=13, B C=14$, and $C A=15$, and let $D$ be the foot of the altitude from $A$ to $B C$. "
    A(h, T15 + r"The incircle of triangle $A C D$ touches $B C$ at $Y$. Find $D Y$.", L(Fraction(12 + 9 - 15, 2)))
    A(h, T15 + r"The incircle of triangle $A C D$ touches $A D$ at $Q$. Find $A Q$.", L(12 - Fraction(12 + 9 - 15, 2)))
    h = "omni-2584"
    trip = [(x, y, 240 // y) for x in range(1, 49) for y in range(x, 49) if x * y == 48 and 240 % y == 0 and 240 // y >= y]
    P = r"Consider all rectangular boxes with integer side lengths $s_{1} \leq s_{2} \leq s_{3}$ such that $s_{1} s_{2}=48$ and $s_{2} s_{3}=240$. "
    A(h, P + r"Find the sum of all possible values of $s_{2}$.", L(sum(t[1] for t in trip)))
    A(h, P + r"Find the largest possible volume.", L(max(x * y * z for x, y, z in trip)))
    A(h, P + r"Find the smallest possible value of $s_{1}+s_{2}+s_{3}$.", L(min(sum(t) for t in trip)))
    h = "omni-2215"
    A(h, r"Find $P(0)$ for the polynomial $P(x)=x^{4}-20 x^{3}+a x^{2}+b x+c$, given that $P(2)=5, P(4)=9$, and $P(6)=13$.", L(2 * 4 * 6 * 8 + 1))
    A(h, r"Find the sum of the coefficients of $P(x)=x^{4}-20 x^{3}+a x^{2}+b x+c$, given that $P(2)=5, P(4)=9$, and $P(6)=13$.", L((1 - 2) * (1 - 4) * (1 - 6) * (1 - 8) + 3))
    h = "omni-2207"
    A(h, r"How many functions $f:\{1,2,3\} \rightarrow\{1,2,3,4,5\}$ satisfy $f(j)<f(i)+j-i$ for all integers $1 \leq i<j \leq 3$?", L(sum(1 for f in product(range(1, 6), repeat=3) if all(f[j] < f[i] + j - i for i in range(3) for j in range(i + 1, 3)))))
    h = "omni-1689"
    for x_, y_, t in ((17, 19, 2000), (13, 15, 3000)):
        A(h, rf"How many pairs $(n, k)$ of nonnegative integers satisfy ${x_} n+{y_} k={t}$?", L(count_sol(x_, y_, t)))
    h = "omni-1728"
    A(h, r"The exterior angles of a convex 27-gon form an arithmetic progression with common difference $1^{\circ}$. Find the largest exterior angle, in degrees.", frac(Fraction(360 - 351, 27) + 26))
    A(h, r"The exterior angles of a convex 24-gon form an arithmetic progression with common difference $1^{\circ}$. Find the smallest exterior angle, in degrees.", frac(Fraction(360 - 276, 24)))
    h = "omni-1900"
    for k in (6, 7):
        A(h, rf"A sequence satisfies $a_{{0}}=0, a_{{1}}=1, a_{{2}}=5$, and $a_{{n+2}} a_{{n}} a_{{n-1}}=a_{{n+2}}+a_{{n}}+a_{{n-1}}$ for $n \geq 1$. Find $a_{{{k}}}$.", L(tan_seq(5, k)))
    h = "omni-3709"
    T = r"Let $P(n)$ be the number of permutations $(a_{1}, \ldots, a_{n})$ of $\{1, \ldots, n\}$ for which $k a_{k}$ is a perfect square for all $k$. Find $P(%d)$."
    for n in (4, 6):
        A(h, T % n, L(P_square(n)))
    return R, []
