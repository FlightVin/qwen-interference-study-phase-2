"""Round 4 of CONFLICTING candidates (2026-11-16 ~02:30) for the 14 training problems still short
after round-3 screening (56/242 in band). Same-setting lookalikes between each problem's 14-16/16
items and its 0-3/16 items. Answers are computed here. Returns ([], conflicting)."""

from fractions import Fraction

import mpmath as mp
import sympy as sp

from batch2 import start_money
from batch4 import digit_iter, sum_sq_digits, trig_case
from batch6 import rec_seq, reg_poly_area
from common import L, frac

x = sp.Symbol("x")


def build():
    C = []
    A = lambda h, p, a: C.append((h, p, a))
    h = "omni-1735"
    A(h, r"We keep rolling a fair 7-sided die (faces 1 to 7) until we get a value less than or equal to the previous roll. What is the probability that exactly 2 rolls are made?", frac(1 - Fraction(21, 49)))
    A(h, r"We keep rolling a fair 4-sided die (faces 1 to 4) until we get a value less than or equal to the previous roll. What is the probability that at least 3 rolls are made?", frac(Fraction(6, 16)))
    h = "omni-1814"
    A(h, r"The circle $x^{2}+y^{2}=5$ meets the hyperbola $x y=2$ at four points, which form a rectangle. Find its area.", L(sp.sqrt(2) * 3 * sp.sqrt(2)))
    A(h, r"The circle $x^{2}+y^{2}=5$ meets the hyperbola $x y=2$ at four points, which form a rectangle. Find its perimeter.", L(2 * (sp.sqrt(2) + 3 * sp.sqrt(2))))
    h = "omni-2140"
    A(h, r"The six roots of $z^{6}=27$ form a regular hexagon in the complex plane. Find its area.", L(reg_poly_area(6, sp.sqrt(3))))
    A(h, r"The eight roots of $z^{8}=16$ form a regular octagon in the complex plane. Find its area.", L(reg_poly_area(8, sp.sqrt(2))))
    h = "omni-1877"
    A(h, r"A circle of radius 3 has center $B$, and point $A$ satisfies $A B=5$. Find the length of a tangent segment from $A$ to the circle.", L(4))
    A(h, r"A circle of radius 12 has center $B$, and point $A$ satisfies $A B=20$. A tangent from $A$ touches the circle at $T$. Find $\sin \angle T A B$.", frac(Fraction(12, 20)))
    A(h, r"A circle of radius 6 has center $B$, and points $A$ and $C$ lie on a line through $B$ with $A B=B C=10$. The tangents from $A$ and $C$ on the same side of the line meet at $K$. Find the distance from $K$ to $B$.", frac(Fraction(10 * 6, 8)))
    h = "omni-1299"
    A(h, r"A $V$-shape is a figure consisting of two rays with a common endpoint that do not lie on one line. What is the maximum number of points in which a circle and a $V$-shape can intersect?", L(4))
    A(h, r"What is the maximum number of points in which a line and two circles can intersect in total?", L(4))
    h = "omni-1315"
    S = r"Stacy has $d$ dollars. She enters a mall with %d shops and a lottery stall. %s When she comes out of the last %s she has no money left. What is the minimum possible value of $d$?"
    A(h, S % (2, r"First she plays the lottery and her money is doubled, then she goes into a shop and spends 1000 dollars. She does this once more.", "shop"), frac(start_money(2, 1000)))
    A(h, S % (2, r"First she goes into a shop and spends 1024 dollars, then she plays the lottery and her money is doubled. She does this once more.", "lottery"), frac(start_money(2, 1024, spend_first=True)))
    h = "omni-2584"
    A(h, r"The faces of a rectangular box have areas 8, 12 and 24. Find its volume.", L(sp.sqrt(8 * 12 * 24)))
    A(h, r"Two faces of a rectangular box have areas 18 and 24 and share an edge of length 6. Find the area of the third face.", L(3 * 4))
    h = "omni-2215"
    A(h, r"$P(x)$ is a monic cubic polynomial with $P(1)=0, P(2)=0$, and $P(3)=6$. Find $P(4)$.", L(sp.expand((x - 1) * (x - 2) * x).subs(x, 4)))
    A(h, r"Let $P(x)=x^{2}+b x+c$ with $P(1)=4$ and $P(2)=9$. Find $P(3)$.", L(16))
    h = "omni-0848"
    A(h, r"A regular hexagon $A B C D E F$ has side length 1. Find the total area of the six triangles $A B C, B C D, C D E, D E F, E F A, F A B$.", L(6 * sp.sqrt(3) / 4))
    A(h, r"A regular hexagon $A B C D E F$ has side length 2. Find the area of quadrilateral $A B C D$.", L(reg_poly_area(6, 2) / 2))
    h = "omni-0912"
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2,3\}$ are there such that $X$ is a proper subset of $Y$?", L(27 - 8))
    A(h, r"How many ordered pairs $(X, Y)$ of subsets of $\{1,2\}$ are there such that neither $X \subseteq Y$ nor $Y \subseteq X$?", L(2))
    h = "omni-2546"
    A(h, r"Two lines through a point $Q$ meet a circle at $A, B$ and at $C, D$, with $Q A=4, Q B=9$, and $Q C=3$. Find $Q D$.", L(12))
    A(h, r"Two lines through a point $Q$ meet a circle at $A, B$ and at $C, D$, with $Q A=2$, $A B=6$, and $Q C=1$, where $Q C<Q D$. Find $C D$.", L(15))
    h = "omni-1900"
    Rr = r"A sequence satisfies $a_{0}=0, a_{1}=%d, a_{2}=%d$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n \geq 1$. Find $a_{6}$."
    for a1, a2 in ((2, 1), (3, 2), (2, 3)):
        A(h, Rr % (a1, a2), L(rec_seq(a1, a2, 6)))
    h = "omni-1391"
    mp.mp.dps = 40
    cases = [(r"Suppose $x$ is a real number such that $\sin \left(1+\cos ^{2} x+\sin ^{4} x\right)=\frac{24}{25}$. Compute $\cos \left(1+\sin ^{2} x+\cos ^{4} x\right)$.", 1, mp.sin, mp.cos, mp.mpf(24) / 25, sp.Rational(-7, 25)),
             (r"Suppose $x$ is a real number such that $\cos \left(1+\cos ^{2} x+\sin ^{4} x\right)=-\frac{1}{3}$. Compute $\sin \left(1+\sin ^{2} x+\cos ^{4} x\right)$.", 1, mp.cos, mp.sin, -mp.mpf(1) / 3, 2 * sp.sqrt(2) / 3)]
    for text, s, fw, tg, val, ans in cases:
        assert abs(trig_case(s, fw, tg, val) - mp.mpf(sp.N(ans, 40))) < 1e-25, text
        A(h, text, L(ans))
    h = "omni-3926"
    D = r"For a positive integer $k$ let $g(k)$ be the sum of the squares of the digits of $k$, and let $g_{1}=g$, $g_{n+1}(k)=g\left(g_{n}(k)\right)$. "
    for n in (6, 7):
        A(h, D + rf"Compute $g_{{{n}}}(4)$.", L(digit_iter(4, n, sum_sq_digits)))
    return [], C
