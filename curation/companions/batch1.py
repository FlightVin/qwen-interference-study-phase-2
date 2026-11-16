"""Companion candidates for training problems T0-T7 (curation/hard_train_ids.json order).

related: easy problems that each isolate one step of the hard problem's solution.
conflicting: lookalikes in the same setting, correctly solved by a method that fails on the hard
problem (brute force on a small case, or the tempting wrong shortcut).
Every answer is computed here (SymPy or brute force), never typed by hand.
"""

from fractions import Fraction
from itertools import product
from math import comb, gcd

import sympy as sp

from common import L, frac, linrec


def phi_k(n, k):  # number of positive integers <= n/k coprime to n
    return sum(1 for m in range(1, n // k + 1) if gcd(m, n) == 1)


def first_m_without_integer_log(b, window):  # least m with no power of b among m..m+window
    pw = {b ** e for e in range(0, 60)}
    m = 1
    while any(x in pw for x in range(m, m + window + 1)):
        m += 1
    return m


def prob_pattern_first(p_a, p_b):
    """P(pattern p_a appears before p_b) in fair coin flips, by solving the prefix Markov chain."""
    states = {""}
    for p in (p_a, p_b):
        states |= {p[:i] for i in range(len(p) + 1)}
    def step(s, c):
        t = s + c
        if t.endswith(p_a) or t.endswith(p_b):
            return t
        while t and not any(p.startswith(t) for p in (p_a, p_b)):
            t = t[1:]
        return t
    states = sorted(states, key=len)
    x = {s: sp.Symbol("x_" + (s or "e")) for s in states}
    eqs = []
    for s in states:
        if s == p_a:
            eqs.append(sp.Eq(x[s], 1))
        elif s == p_b:
            eqs.append(sp.Eq(x[s], 0))
        else:
            eqs.append(sp.Eq(x[s], sp.Rational(1, 2) * (x[step(s, "1")] + x[step(s, "0")])))
    return sp.solve(eqs, list(x.values()))[x[""]]


def p_contains(n, pats):  # probability n fair flips contain all patterns in pats
    hits = sum(all(p in "".join(w) for p in pats) for w in product("01", repeat=n))
    return Fraction(hits, 2 ** n)


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def v2(n):
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def walks(length, start=(0, 0, 0, 0), closed=False):
    pts = list(product((-1, 0, 1), repeat=4))
    nbr = {p: [q for q in pts if sum((a - b) ** 2 for a, b in zip(p, q)) == 4] for p in pts}
    count = {start: 1}
    for _ in range(length):
        nxt = {}
        for p, c in count.items():
            for q in nbr[p]:
                nxt[q] = nxt.get(q, 0) + c
        count = nxt
    return count.get(start, 0) if closed else sum(count.values())


def area(*pts):
    (x1, y1), (x2, y2), (x3, y3) = pts
    return sp.Abs(sp.Rational(1, 2) * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))


def build():
    R, C = [], []
    # ---- T0 omni-0793: phi^k(n), answer phi(2001)+1 ----
    h = "omni-0793"
    R += [(h, "Compute $\\varphi(2001)$, where $\\varphi$ is Euler's totient function.", L(sp.totient(2001))),
          (h, "Write $2002^{2}-1$ as a product of two positive integers that differ by 2. What is the smaller one?", L(2001)),
          (h, "Define $\\varphi^{k}(n)$ as the number of positive integers that are at most $n / k$ and relatively prime to $n$. Compute $\\varphi^{5}(35)$.", L(phi_k(35, 5))),
          (h, "How many positive integers $m \\leq 2003$ are relatively prime to 2001?", L(sum(1 for m in range(1, 2004) if gcd(m, 2001) == 1)))]
    # Lookalikes where rad(n) divides n/k, so the proportional count phi(n)/k is exact; on the
    # hard problem (n = 2001*2003 squarefree) that shortcut fails.
    D = "Define $\\varphi^{k}(n)$ as the number of positive integers that are at most $n / k$ and relatively prime to $n$. "
    C += [(h, D + "Compute $\\varphi^{3}(90)$.", L(phi_k(90, 3))),
          (h, D + "Compute $\\varphi^{4}(200)$.", L(phi_k(200, 4))),
          (h, D + "Compute $\\varphi^{6}(216)$.", L(phi_k(216, 6))),
          (h, D + "Compute $\\varphi^{5}(1000)$.", L(phi_k(1000, 5)))]
    # ---- T1 omni-1735: die rolled until non-increase, E = (1+1/n)^n ----
    h = "omni-1735"
    R += [(h, "A fair 6-sided die is rolled 3 times. What is the probability that the three results are strictly increasing?", frac(Fraction(comb(6, 3), 6 ** 3))),
          (h, "Compute $\\sum_{k=0}^{4} \\binom{4}{k} \\frac{1}{4^{k}}$.", frac(sum(Fraction(comb(4, k), 4 ** k) for k in range(5)))),
          (h, "To the nearest integer, what is $1000\\left(1+\\frac{1}{10}\\right)^{10}$?", L(round(1000 * (1 + 0.1) ** 10))),
          (h, "We roll a fair die with faces $1,2,\\ldots,n$. What is the probability that $k$ rolls come out strictly increasing, for $n=10$ and $k=4$?", frac(Fraction(comb(10, 4), 10 ** 4)))]
    C += [(h, "We keep rolling a fair 2-sided die (faces 1 and 2) until we get a value less than or equal to the previous roll. What is the expected number of rolls?", frac(Fraction(3, 2) ** 2)),
          (h, "We keep rolling a fair 3-sided die (faces 1, 2, 3) until we get a value less than or equal to the previous roll. What is the expected number of rolls?", frac(Fraction(4, 3) ** 3)),
          (h, "A fair 4-sided die is rolled twice. What is the probability that the second roll is less than or equal to the first?", frac(Fraction(sum(b <= a for a in range(1, 5) for b in range(1, 5)), 16))),
          (h, "We keep rolling a fair 3-sided die (faces 1, 2, 3) until we get a value less than or equal to the previous roll. What is the probability that exactly 3 rolls are made?", frac(Fraction(comb(3, 2), 9) - Fraction(comb(3, 3), 27)))]
    # ---- T2 omni-1814: circle with diameter AB meets xy=1 ----
    h = "omni-1814"
    x = sp.symbols("x", positive=True)
    R += [(h, "Find the slope of the line through the points $\\left(2, \\frac{1}{2}\\right)$ and $\\left(5, \\frac{1}{5}\\right)$.", L(sp.Rational(1, 5 - 2) * (sp.Rational(1, 5) - sp.Rational(1, 2)))),
          (h, "Find the positive number $x$ for which the line through $\\left(2, \\frac{1}{2}\\right)$ and $\\left(x, \\frac{1}{x}\\right)$ is perpendicular to the line through $\\left(-3,-\\frac{1}{3}\\right)$ and $\\left(x, \\frac{1}{x}\\right)$.", L(sp.solve(sp.Eq(1 / (2 * x) * 1 / (-3 * x), -1), x)[0])),
          (h, "Compute the distance between the points $\\left(\\sqrt{6}, \\frac{1}{\\sqrt{6}}\\right)$ and $\\left(-\\sqrt{6},-\\frac{1}{\\sqrt{6}}\\right)$.", L(sp.sqrt((2 * sp.sqrt(6)) ** 2 + (2 / sp.sqrt(6)) ** 2))),
          (h, "The point $P=(1, y)$ with $y>0$ lies on the circle with diameter $AB$, where $A=(0,0)$ and $B=(4,0)$. Find $y$.", L(sp.solve(sp.Eq(1 * (1 - 4) + x ** 2, 0), x)[0]))]
    C += [(h, "Find the distance between the two points where the circle $x^{2}+y^{2}=2$ meets the hyperbola $x y=1$.", L(sp.sqrt(8))),
          (h, "The circle $x^{2}+y^{2}=\\frac{17}{4}$ meets the hyperbola $x y=1$ at four points. Find the largest $x$-coordinate among them.", L(max(sp.solve(sp.Eq(x ** 2 + 1 / x ** 2, sp.Rational(17, 4)), x)))),
          (h, "Find the distance between the two points where the line $y=x$ meets the hyperbola $x y=4$.", L(sp.sqrt(32))),
          (h, "The points $A=(1,1)$ and $B=(-1,-1)$ lie on the hyperbola $x y=1$. Find the radius of the circle with diameter $AB$.", L(sp.sqrt(2)))]
    # ---- T3 omni-2140: roots of x^6+6x^3-216 form a hexagon ----
    h = "omni-2140"
    R += [(h, "Factor $x^{6}+6 x^{3}-216$ as a product of two cubic polynomials with integer coefficients. What is the constant term of the factor whose constant term is positive?", L(18)),
          (h, "Compute $\\sqrt[3]{12} \\cdot \\sqrt[3]{18}$.", L(sp.nsimplify(sp.cbrt(216)))),
          (h, "A triangle has two sides of lengths 2 and 3 with a $60^{\\circ}$ angle between them. Find its area.", L(sp.Rational(1, 2) * 6 * sp.sin(sp.pi / 3))),
          (h, "The three roots of $z^{3}=8$ form a triangle in the complex plane. Find its area.", L(3 * sp.sqrt(3) / 4 * (2 * sp.sqrt(3)) ** 2 / 3))]
    C += [(h, "The six roots of $z^{6}=64$ form a regular hexagon in the complex plane. Find its area.", L(3 * sp.sqrt(3) / 2 * 4)),
          (h, "The six roots of $z^{6}-1=0$ form a regular hexagon in the complex plane. Find its area.", L(3 * sp.sqrt(3) / 2)),
          (h, "The four roots of $z^{4}=16$ form a square in the complex plane. Find its area.", L(8)),
          (h, "The three roots of $z^{3}=27$ form a triangle in the complex plane. Find its area.", L(3 * sp.sqrt(3) / 4 * (3 * sp.sqrt(3)) ** 2 / 3))]
    # ---- T4 omni-2430: least m with no integer log_b in a window ----
    h = "omni-2430"
    R += [(h, "Find the smallest integer $n$ such that $3^{n+1}-3^{n}>2018$.", L(next(n for n in range(30) if 3 ** (n + 1) - 3 ** n > 2018))),
          (h, "Find the smallest integer $n$ such that $2^{n+1}-2^{n}>2018$.", L(next(n for n in range(30) if 2 ** (n + 1) - 2 ** n > 2018))),
          (h, "Find the largest positive integer $b$ such that $b^{2}-b \\leq 2018$.", L(max(b for b in range(1, 100) if b * b - b <= 2018))),
          (h, "Find the smallest positive integer $m$ such that none of $\\log _{2}(m), \\log _{2}(m+1), \\ldots, \\log _{2}(m+10)$ is an integer.", L(first_m_without_integer_log(2, 10)))]
    C += [(h, "Find the smallest positive integer $m$ such that none of $\\log _{3}(m), \\log _{3}(m+1), \\log _{3}(m+2), \\log _{3}(m+3)$ is an integer.", L(first_m_without_integer_log(3, 3))),
          (h, "Find the smallest positive integer $m$ such that none of $\\log _{2}(m), \\log _{2}(m+1), \\ldots, \\log _{2}(m+5)$ is an integer.", L(first_m_without_integer_log(2, 5))),
          (h, "Find the smallest positive integer $m$ such that none of $\\log _{10}(m), \\log _{10}(m+1), \\ldots, \\log _{10}(m+20)$ is an integer.", L(first_m_without_integer_log(10, 20))),
          (h, "Find the smallest positive integer $m$ such that none of $\\log _{5}(m), \\log _{5}(m+1), \\ldots, \\log _{5}(m+7)$ is an integer.", L(first_m_without_integer_log(5, 7)))]
    # ---- T5 omni-0962: 10101 before 010101 ----
    h = "omni-0962"
    R += [(h, "A fair coin is flipped repeatedly (1 for heads, 0 for tails). What is the probability that the sequence 11 appears before the sequence 01?", L(prob_pattern_first("11", "01"))),
          (h, "A fair coin is flipped repeatedly (1 for heads, 0 for tails). What is the probability that the sequence 110 appears before the sequence 100?", L(prob_pattern_first("110", "100"))),
          (h, "A fair coin is flipped repeatedly (1 for heads, 0 for tails). What is the probability that the sequence 101 appears before the sequence 011?", L(prob_pattern_first("101", "011"))),
          (h, "A fair coin is flipped repeatedly (1 for heads, 0 for tails). What is the probability that the sequence 0101 appears before the sequence 1011?", L(prob_pattern_first("0101", "1011")))]
    C += [(h, "A fair coin is flipped 4 times (1 for heads, 0 for tails). What is the probability that the sequence 101 appears in the results as consecutive flips?", frac(p_contains(4, ["101"]))),
          (h, "A fair coin is flipped 6 times (1 for heads, 0 for tails). What is the probability that both 101 and 010 appear as consecutive flips?", frac(p_contains(6, ["101", "010"]))),
          (h, "A fair coin is flipped 4 times. What is the probability that two heads in a row appear at least once?", frac(p_contains(4, ["11"]))),
          (h, "A fair coin is flipped 7 times (1 for heads, 0 for tails). What is the probability that the sequence 10101 appears as consecutive flips?", frac(p_contains(7, ["10101"])))]
    # ---- T6 omni-0528: 2-adic valuation of closed walks ----
    h = "omni-0528"
    R += [(h, "Let $F_{1}=F_{2}=1$ and $F_{n}=F_{n-1}+F_{n-2}$. Find the largest integer $n$ such that $2^{n}$ divides $F_{24}$.", L(v2(fib(24)))),
          (h, "Let $F_{1}=F_{2}=1$ and $F_{n}=F_{n-1}+F_{n-2}$. Find the remainder when $F_{2019}$ is divided by 4.", L(fib(2019) % 4)),
          (h, "Let $a_{0}=1, a_{1}=4$, and $a_{i}=4 a_{i-1}+16 a_{i-2}$ for $i \\geq 2$. Compute $\\frac{a_{6}}{4^{6}}$.", L(sp.Rational(linrec(1, 4, 4, 16, 6), 4 ** 6))),
          (h, "How many points with all coordinates in $\\{-1,0,1\\}$ in four-dimensional space are at distance exactly 2 from the origin?", L(sum(1 for p in product((-1, 0, 1), repeat=4) if sum(c * c for c in p) == 4)))]
    C += [(h, "Let $S$ be the set of points in four-dimensional space with every coordinate in $\\{-1,0,1\\}$. How many sequences $P_{1}, P_{2}, P_{3}$ of points in $S$ have $P_{1}=(0,0,0,0)$ and $P_{1} P_{2}=P_{2} P_{3}=2$?", L(walks(2))),
          (h, "Let $S$ be the set of points in four-dimensional space with every coordinate in $\\{-1,0,1\\}$. How many sequences $P_{1}, P_{2}, P_{3}, P_{4}$ of points in $S$ have $P_{1}=(0,0,0,0)$, $P_{i} P_{i+1}=2$ for $1 \\leq i \\leq 4$, where $P_{5}=P_{1}$?", L(walks(4, closed=True))),
          (h, "Let $S$ be the set of points in four-dimensional space with every coordinate in $\\{-1,0,1\\}$. How many sequences $P_{1}, \\ldots, P_{6}$ of points in $S$ have $P_{1}=(0,0,0,0)$, $P_{i} P_{i+1}=2$ for $1 \\leq i \\leq 6$, where $P_{7}=P_{1}$?", L(walks(6, closed=True))),
          (h, "Let $N$ be the number of sequences $P_{1}, P_{2}, P_{3}, P_{4}$ of points with coordinates in $\\{-1,0,1\\}$ (four-dimensional space) such that $P_{1}=(0,0,0,0)$ and $P_{i} P_{i+1}=2$ for $1 \\leq i \\leq 4$, where $P_{5}=P_{1}$. Find the largest integer $n$ such that $2^{n}$ divides $N$.", L(v2(walks(4, closed=True))))]
    # ---- T7 omni-0596: affine image of a triangle ----
    h = "omni-0596"
    R += [(h, "Find the distance from the point $(1,1)$ to the line $6 x+8 y=48$.", L(sp.Rational(abs(6 + 8 - 48), 10))),
          (h, "Find the area of the triangle with vertices $(1,2),(0,-3)$, and $(-2,0)$.", L(area((1, 2), (0, -3), (-2, 0)))),
          (h, "A point $P=(a, b)$ lies inside the triangle with vertices $(0,0),(4,0),(0,3)$. If $P=\\left(1, \\frac{1}{2}\\right)$, find the distance from $P$ to the side joining $(4,0)$ and $(0,3)$.", L(sp.Rational(12 - 3 - 2, 5))),
          (h, "A map sends each point $(a, b)$ to $(1+2 a, 1+3 b)$. Find the area of the image of the triangle with vertices $(0,0),(1,0),(0,1)$.", L(area((1, 1), (3, 1), (1, 4))))]
    C += [(h, "Find the area of the triangle with vertices $(0,6),(0,0)$, and $(8,0)$.", L(24)),
          (h, "Find the distance from the point $(2,1)$ to the line through $(0,6)$ and $(8,0)$.", L(sp.Rational(abs(12 + 8 - 48), 10))),
          (h, "For points $P=(a, b)$ in the triangle with vertices $(0,0),(8,0),(0,6)$, find the largest possible value of $a+b$.", L(8)),
          (h, "Triangle $ABC$ has $AB=6, BC=8, AC=10$. Find the distance from $B$ to side $AC$.", L(sp.Rational(48, 10)))]
    return R, C
