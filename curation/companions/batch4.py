"""Companion candidates for training problems T24-T31 (same rules as batch1.py)."""

import heapq
from fractions import Fraction
from math import factorial, isqrt

import mpmath as mp
import sympy as sp

from common import L, frac

x = sp.Symbol("x")


def cafe(tables, seats, sizes=(1, 2, 3, 4), single_table_first=False, big_counter=True):
    """Least M + N (groups + people) that leaves every table and counter seat occupied.
    A group of size s > 1 takes a free table, else s free counter seats (if allowed), else cannot
    sit. A group of 1 takes a counter seat first (or a table first if single_table_first)."""
    start, best = (tables, seats), {}
    pq = [(0, start)]
    while pq:
        cost, (t, c) = heapq.heappop(pq)
        if (t, c) == (0, 0):
            return cost
        if best.get((t, c), 1e9) <= cost:
            continue
        best[(t, c)] = cost
        for s in sizes:
            if s == 1:
                order = ("t", "c") if single_table_first else ("c", "t")
            else:
                order = ("t", "c") if big_counter else ("t",)
            nxt = None
            for o in order:
                if o == "t" and t > 0:
                    nxt = (t - 1, c)
                    break
                if o == "c" and c >= s:
                    nxt = (t, c - s)
                    break
            if nxt:
                heapq.heappush(pq, (cost + s + 1, nxt))
    return None


def max_ap_polygon(diffs, integer_angles=False):
    """Largest n for a convex n-gon whose angles form an AP with common difference in diffs."""
    best = 0
    for n in range(3, 400):
        for a in diffs:
            xs = Fraction(360 - Fraction(n * (n - 1), 2) * a, n)  # smallest exterior angle
            if xs > 0 and xs + (n - 1) * a < 180 and (not integer_angles or xs.denominator == 1):
                best = max(best, n)
    return best


def tan_recursion(k, a1=1):
    """a_0 = 0, a_1 = a1, a_2 = a > 0, a_{n+2} a_n a_{n-1} = a_{n+2} + a_n + a_{n-1}; the sequence
    reaches a_{k-1} but cannot be extended to a_k. Smallest such a (exact)."""
    a = sp.Symbol("a", positive=True)
    seq = [sp.Integer(0), sp.Integer(a1), a]
    for n in range(1, k - 2):
        p = sp.together(seq[n] * seq[n - 1] - 1)
        seq.append(sp.simplify((seq[n] + seq[n - 1]) / p))
    u, v = seq[k - 2], seq[k - 3]
    sols = []
    for r in sp.solve(sp.Eq(sp.simplify(u * v), 1), a):
        if not r.is_real or r <= 0:
            continue
        vals = [sp.simplify(t.subs(a, r)) for t in seq]
        if all(t.is_finite for t in vals) and sp.simplify(vals[k - 2] + vals[k - 3]) != 0:
            # every earlier step must be solvable (a_n a_{n-1} != 1 for n < k-2)
            if all(sp.simplify(vals[n] * vals[n - 1] - 1) != 0 for n in range(1, k - 3)):
                sols.append(sp.nsimplify(r))
    return min(sols, key=lambda r: float(r)) if sols else None


def matchings(n, ok):  # number of permutations a of [n] with ok(k, a_k) for all k (bitmask DP)
    dp = {0: 1}
    for k in range(1, n + 1):
        nd = {}
        for mask, c in dp.items():
            for v in range(1, n + 1):
                if not mask >> v & 1 and ok(k, v):
                    nd[mask | 1 << v] = nd.get(mask | 1 << v, 0) + c
        dp = nd
    return sum(dp.values())


def sqfree(k):
    return sp.prod([p ** (e % 2) for p, e in sp.factorint(k).items()]) if k > 1 else 1


def P_square(n):  # product over square classes of (class size)!
    cls = {}
    for k in range(1, n + 1):
        cls[sqfree(k)] = cls.get(sqfree(k), 0) + 1
    return sp.prod([factorial(c) for c in cls.values()])


is_sq = lambda m: isqrt(m) ** 2 == m
is_cube = lambda m: round(m ** (1 / 3)) ** 3 == m


def trig_case(inner_shift, fwd, target, value, sign_check=True):
    """Find x with fwd(s + cos^2 + sin^4) = value numerically, then return target(s + sin^2 + cos^4)."""
    f = lambda t: fwd(inner_shift + mp.cos(t) ** 2 + mp.sin(t) ** 4) - value
    for t0 in [i * 0.05 for i in range(1, 32)]:
        try:
            t = mp.findroot(f, t0)
            return target(inner_shift + mp.sin(t) ** 2 + mp.cos(t) ** 4)
        except Exception:
            continue
    raise ValueError("no x")


def real_root_count(poly):
    return len(set(sp.Poly(sp.expand(poly), x).real_roots()))


def digit_iter(k, n, f):
    for _ in range(n):
        k = f(k)
    return k


sq_digit_sum = lambda k: sum(map(int, str(k))) ** 2
sum_sq_digits = lambda k: sum(int(d) ** 2 for d in str(k))


def build():
    R, C = [], []
    # ---- T24 omni-1346: cafe, minimum M + N = 16 ----
    h = "omni-1346"
    assert cafe(3, 5) == 16
    base = r"A cafe has {t} and {c} individual counter seats. People enter in groups of size between 1 and {m}, inclusive, and groups never share a table. {rule} One morning, $M$ groups consisting of a total of $N$ people enter and sit down. Then, a single person walks in, and realizes that all the tables and counter seats are occupied by some person or group. What is the minimum possible value of $M+N$?"
    std = r"A group of more than 1 will always try to sit at a table, but will sit in counter seats if no tables are available. Conversely, a group of 1 will always try to sit at the counter first."
    tw = lambda n: {1: "1 table", 2: "2 tables", 3: "3 tables", 4: "4 tables"}[n]
    R += [(h, base.format(t=tw(2), c=3, m=4, rule=std), L(cafe(2, 3))),
          (h, base.format(t=tw(1), c=4, m=4, rule=std), L(cafe(1, 4))),
          (h, base.format(t=tw(2), c=5, m=4, rule=std), L(cafe(2, 5))),
          (h, r"Groups of people of sizes between 1 and 4 sit at a counter with 5 seats, filling every seat. For a group of size $s$ we score $s+1$ points. What is the minimum possible total score?", L(min(5 + sum(p.values()) for p in sp.utilities.iterables.partitions(5, k=4))))]
    C += [(h, base.format(t=tw(3), c=5, m=4, rule=r"A group of more than 1 will always try to sit at a table, but will sit in counter seats if no tables are available. A group of 1 will also always try to sit at a table first, and sits at the counter only if no tables are available."), L(cafe(3, 5, single_table_first=True))),
          (h, base.format(t=tw(2), c=5, m=4, rule=r"A group of more than 1 will only sit at a table. A group of 1 will always try to sit at the counter first."), L(cafe(2, 5, big_counter=False))),
          (h, base.format(t=tw(4), c=5, m=4, rule=r"A group of more than 1 will always try to sit at a table, but will sit in counter seats if no tables are available. A group of 1 will also always try to sit at a table first, and sits at the counter only if no tables are available."), L(cafe(4, 5, single_table_first=True))),
          (h, base.format(t=tw(4), c=2, m=4, rule=std), L(cafe(4, 2)))]
    # ---- T25 omni-1728: polygon angles in AP, n <= 27 ----
    h = "omni-1728"
    assert max_ap_polygon(range(1, 200)) == 27
    R += [(h, r"Find the largest positive integer $n$ such that $\frac{n(n-1)}{2}<180$.", L(max(n for n in range(1, 100) if n * (n - 1) < 360))),
          (h, r"The exterior angles of a convex 9-gon form an arithmetic progression with common difference $4^{\circ}$. Find the smallest exterior angle, in degrees.", L(Fraction(360 - 36 * 4, 9))),
          (h, r"The interior angles of a convex pentagon form an arithmetic progression with common difference $10^{\circ}$. Find the largest angle, in degrees.", L(108 + 20)),
          (h, r"The angles of a convex $n$-sided polygon form an arithmetic progression with common difference $1^{\circ}$ and $n=20$. Find the smallest exterior angle, in degrees.", L(Fraction(360 - 190, 20)))]
    T25 = r"The angles of a convex $n$-sided polygon form an arithmetic progression whose common difference (in degrees) is {d}. Find the largest possible value of $n$ for which this is possible."
    C += [(h, T25.format(d="an integer greater than 1"), L(max_ap_polygon(range(2, 200)))),
          (h, T25.format(d="5"), L(max_ap_polygon([5]))),
          (h, r"The angles of a convex $n$-sided polygon are all integers (in degrees) and form an arithmetic progression whose common difference is a non-zero integer. Find the largest possible value of $n$ for which this is possible.", L(max_ap_polygon(range(1, 200), integer_angles=True))),
          (h, T25.format(d="a positive multiple of 3"), L(max_ap_polygon(range(3, 200, 3))))]
    # ---- T26 omni-1900: tangent substitution, a_2 = sqrt2 - 1 ----
    h = "omni-1900"
    assert sp.simplify(tan_recursion(10) - (sp.sqrt(2) - 1)) == 0
    a = sp.Symbol("a")
    R += [(h, r"A sequence satisfies $a_{0}=0, a_{1}=1$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n=1$. Find $a_{3}$.", L(-1)),
          (h, r"Compute $\tan \frac{\pi}{12}$.", L(sp.tan(sp.pi / 12))),
          (h, r"Real numbers $x, y, z$ satisfy $x y z=x+y+z$, $x=1$ and $y=2$. Find $z$.", L(sp.solve(sp.Eq(2 * a, 3 + a), a)[0])),
          (h, r"A sequence satisfies $a_{0}=0, a_{1}=1, a_{2}=2$, and $a_{n+2} a_{n} a_{n-1}=a_{n+2}+a_{n}+a_{n-1}$ for $n=1,2$. Find $a_{4}$.", L(sp.Rational(2 + 1, 2 * 1 - 1)))]
    T26 = r"A sequence of real numbers $a_{{0}}, a_{{1}}, \ldots, a_{{{m}}}$ with $a_{{0}}=0, a_{{1}}={a1}$, and $a_{{2}}>0$ satisfies $$a_{{n+2}} a_{{n}} a_{{n-1}}=a_{{n+2}}+a_{{n}}+a_{{n-1}}$$ for all $1 \leq n \leq {last}$, but cannot be extended to $a_{{{k}}}$. In other words, no value of $a_{{{k}}} \in \mathbb{{R}}$ satisfies $$a_{{{k}}} a_{{{k2}}} a_{{{k3}}}=a_{{{k}}}+a_{{{k2}}}+a_{{{k3}}}$$ Compute the smallest possible value of $a_{{2}}$."
    for k, a1 in ((9, 1), (10, 2), (11, 1), (12, 1)):
        r = tan_recursion(k, a1)
        if r is not None:
            C.append((h, T26.format(m=k - 1, a1=a1, last=k - 3, k=k, k2=k - 2, k3=k - 3), L(r)))
    # ---- T27 omni-3709: P(n) = product of (square class sizes)! ----
    h = "omni-3709"
    assert P_square(9) == matchings(9, lambda k, v: is_sq(k * v))
    assert next(n for n in range(1, 5000) if P_square(n) % 2010 == 0) == 4489
    R += [(h, r"How many permutations $(a_{1}, \ldots, a_{10})$ of $\{1,2, \ldots, 10\}$ are there such that $k a_{k}$ is a perfect square for all $1 \leq k \leq 10$?", L(P_square(10))),
          (h, r"How many positive integers $k \leq 1000$ are there such that $2 k$ is a perfect square?", L(sum(1 for k in range(1, 1001) if is_sq(2 * k)))),
          (h, r"Let $P(n)$ be the number of permutations $(a_{1}, \ldots, a_{n})$ of $\{1, \ldots, n\}$ for which $k a_{k}$ is a perfect square for all $1 \leq k \leq n$. Find the smallest $n$ such that $P(n)$ is a multiple of 7.", L(next(n for n in range(1, 500) if P_square(n) % 7 == 0))),
          (h, r"How many permutations $(a_{1}, \ldots, a_{20})$ of $\{1,2, \ldots, 20\}$ are there such that $k a_{k}$ is a perfect square for all $1 \leq k \leq 20$?", L(P_square(20)))]
    T27 = r"How many permutations $(a_{{1}}, \ldots, a_{{{n}}})$ of $\{{1,2, \ldots, {n}\}}$ are there such that {cond} for all $1 \leq k \leq {n}$?"
    C += [(h, T27.format(n=n, cond=r"$k+a_{k}$ is a perfect square"), L(matchings(n, lambda k, v: is_sq(k + v)))) for n in (10, 15, 16)]
    C += [(h, T27.format(n=8, cond=r"$k+a_{k}$ is a prime number"), L(matchings(8, lambda k, v: sp.isprime(k + v))))]
    # ---- T28 omni-1391: equal arguments, sign from the range of the argument ----
    h = "omni-1391"
    mp.mp.dps = 40
    chk = trig_case(1, mp.sin, mp.cos, mp.mpf(13) / 14)
    assert abs(chk + 3 * mp.sqrt(3) / 14) < 1e-25
    R += [(h, r"Find the minimum value of $t^{2}-t+2$ for $0 \leq t \leq 1$.", frac(Fraction(7, 4))),
          (h, r"If $\sin \alpha=\frac{3}{5}$ and $\frac{\pi}{2}<\alpha<\pi$, find $\cos \alpha$.", frac(Fraction(-4, 5))),
          (h, r"Find the maximum value of $1+\cos ^{2} x+\sin ^{4} x$ over all real $x$.", L(2)),
          (h, r"If $\sin ^{2} x=\frac{1}{3}$, compute $1+\cos ^{2} x+\sin ^{4} x$.", frac(1 + Fraction(2, 3) + Fraction(1, 9)))]
    cases = [(r"Suppose $x$ is a real number such that $\sin \left(\cos ^{2} x+\sin ^{4} x\right)=\frac{4}{5}$. Compute $\cos \left(\sin ^{2} x+\cos ^{4} x\right)$.", 0, mp.sin, mp.cos, mp.mpf(4) / 5, sp.Rational(3, 5)),
             (r"Suppose $x$ is a real number such that $\cos \left(1+\cos ^{2} x+\sin ^{4} x\right)=-\frac{1}{5}$. Compute $\sin \left(1+\sin ^{2} x+\cos ^{4} x\right)$.", 1, mp.cos, mp.sin, -mp.mpf(1) / 5, 2 * sp.sqrt(6) / 5),
             (r"Suppose $x$ is a real number such that $\sin \left(2+\cos ^{2} x+\sin ^{4} x\right)=\frac{1}{3}$. Compute $\cos \left(2+\sin ^{2} x+\cos ^{4} x\right)$.", 2, mp.sin, mp.cos, mp.mpf(1) / 3, -2 * sp.sqrt(2) / 3),
             (r"Suppose $x$ is a real number such that $\sin \left(\cos ^{2} x+\sin ^{4} x\right)=\frac{3}{4}$. Compute $\cos \left(\sin ^{2} x+\cos ^{4} x\right)$.", 0, mp.sin, mp.cos, mp.mpf(3) / 4, sp.sqrt(7) / 4)]
    for text, s, fw, tg, val, ans in cases:
        assert abs(trig_case(s, fw, tg, val) - mp.mpf(sp.N(ans, 40))) < 1e-25, text
        C.append((h, text, L(ans)))
    # ---- T29 omni-0166: rhombus of side 1 with CD = 1, AB = sqrt3 ----
    h = "omni-0166"
    R += [(h, r"A rhombus $A C B D$ has side length 1 and diagonal $C D=1$. Find the length of the diagonal $A B$.", L(sp.sqrt(3))),
          (h, r"Two spheres of radius 1 have centers $A$ and $B$ with $A B=\sqrt{3}$. Find the radius of the circle in which they intersect.", L(sp.sqrt(1 - sp.Rational(3, 4)))),
          (h, r"Find the maximum value of $\sin \theta+\cos \theta$ over all real $\theta$.", L(sp.sqrt(2))),
          (h, r"Points $C$ and $D$ lie on a circle of radius $\frac{1}{2}$ with $C D=1$. Find the angle $\angle C O D$ in degrees, where $O$ is the center.", L(180))]
    C += [(h, r"Let $A, B, C$ denote three points such that at most one of the distances $A B, A C, B C$ is greater than 1. Determine the maximum value of the sum of the three distances.", L(4)),
          (h, r"Let $A, B, C, D$ denote four points in space such that all six distances $A B, A C, A D, B C, B D, C D$ are at most 1. Determine the maximum value of the sum of the six distances.", L(6)),
          (h, r"Let $A, B, C$ denote three points in the plane such that all three distances $A B, A C, B C$ are at most 1. Determine the maximum value of the sum of the three distances.", L(3))]
    C.append((h, r"Let $A, B, C, D$ be four points in space such that $A C=B C=A D=B D=1$ and $C D=\frac{1}{2}$. Find the largest possible value of $A B$.", L(2 * sp.sqrt(1 - sp.Rational(1, 16)))))
    # ---- T30 omni-1488: f = x^3 - 3x, g values {1, 5, 9} ----
    h = "omni-1488"
    f3 = x ** 3 - 3 * x
    R += [(h, r"The function $f(x)=x^{3}+a x$ has a local maximum at $x=-1$. Find $a$.", L(sp.solve(sp.diff(x ** 3 + a * x, x).subs(x, -1), a)[0])),
          (h, r"Find the local maximum value of $f(x)=x^{3}-3 x$.", L(f3.subs(x, -1))),
          (h, r"How many distinct real solutions does $x^{3}-3 x=1$ have?", L(real_root_count(f3 - 1))),
          (h, r"Let $f(x)=x^{3}-3 x$. How many distinct real solutions does $f(f(x))=0$ have?", L(real_root_count(f3.subs(x, f3))))]
    C += [(h, r"Let $f(x)=x^{3}-3 x$. How many distinct real solutions does $f(f(x))=2$ have?", L(real_root_count(f3.subs(x, f3) - 2))),
          (h, r"Let $f(x)=x^{3}-x$. How many distinct real solutions does $f(f(x))=0$ have?", L(real_root_count((x ** 3 - x).subs(x, x ** 3 - x)))),
          (h, r"Let $f(x)=x^{3}-4 x$. How many distinct real solutions does $f(f(x))=0$ have?", L(real_root_count((x ** 3 - 4 * x).subs(x, x ** 3 - 4 * x)))),
          (h, r"Let $f(x)=x^{3}-3 x$. How many distinct real solutions does $f(f(x))=5$ have?", L(real_root_count(f3.subs(x, f3) - 5)))]
    # ---- T31 omni-3926: f1 = (digit sum)^2, cycle 169 <-> 256 ----
    h = "omni-3926"
    assert digit_iter(2 ** 1990, 1991, sq_digit_sum) == 256
    R += [(h, r"Find the remainder when $2^{1990}$ is divided by 9.", L(pow(2, 1990, 9))),
          (h, r"For a positive integer $k$ let $f(k)$ be the square of the sum of the digits of $k$. Compute $f\left(2^{10}\right)$.", L(sq_digit_sum(2 ** 10))),
          (h, r"For a positive integer $k$ let $f(k)$ be the square of the sum of the digits of $k$. Compute $f\left(2^{20}\right)$.", L(sq_digit_sum(2 ** 20))),
          (h, r"For a positive integer $k$ denote the square of the sum of its digits by $f_{1}(k)$ and let $f_{n+1}(k)=f_{1}\left(f_{n}(k)\right)$. Determine $f_{3}(99)$.", L(digit_iter(99, 3, sq_digit_sum)))]
    T31 = r"For a given positive integer $k$ denote {fdef} by $f_{{1}}(k)$ and let $f_{{n+1}}(k)=f_{{1}}\left(f_{{n}}(k)\right)$. Determine the value of $f_{{{n}}}\left({b}^{{{e}}}\right)$."
    C += [(h, T31.format(fdef="the square of the sum of its digits", n=1990, b=2, e=1990), L(digit_iter(2 ** 1990, 1990, sq_digit_sum))),
          (h, T31.format(fdef="the square of the sum of its digits", n=1991, b=3, e=1990), L(digit_iter(3 ** 1990, 1991, sq_digit_sum))),
          (h, T31.format(fdef="the square of the sum of its digits", n=1991, b=5, e=100), L(digit_iter(5 ** 100, 1991, sq_digit_sum))),
          (h, T31.format(fdef="the sum of the squares of its digits", n=1991, b=2, e=1990), L(digit_iter(2 ** 1990, 1991, sum_sq_digits)))]
    return R, C
