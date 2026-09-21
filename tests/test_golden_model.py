"""Validation of the golden reference model (src/golden_model.py, src/galois_field.py).

The golden model is the single oracle every library entry is graded against, so it
must not be "correct because we say so". These tests pin it to two things that are
independent of our code:

  1. The mathematical definition of the two ALU families (see /format):
       Family M (k prime or composite, "modular"):   the ring Z / k^n Z with
           carry/borrow flags — radix-k integer arithmetic with carry propagation.
       Family F (k a prime power, "Galois-field"):   the ring GF(q)[x] / (x^n),
           q = p^m: words are polynomials of degree < n over GF(q); ADD/SUB/NEG are
           digit-wise field operations, MUL is polynomial multiplication truncated
           to n digits, INC/DEC add/subtract the field element 1 in digit 0.
     Family M is checked against Python's arbitrary-precision integers (an
     independent implementation of Z/mZ). Family F is checked against the
     `galois` package (Hostetter, 2020-) for the field tables and against a
     straightforward polynomial-ring implementation for the word operations.

  2. Algebraic axioms that must hold whatever the implementation: group/ring laws,
     inverses, consistency between operations (NEG = SUB(0, a), INC = ADD(a, 1) …).

Small parameter sets are checked exhaustively; larger ones with a fixed random seed.
Run: python -m pytest tests/test_golden_model.py -q
"""
import itertools
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from galois_field import compute_gf_tables, resolve_logic_type          # noqa: E402
from golden_model import (GoldenModel, OP_ADD, OP_SUB, OP_MUL, OP_NEG,    # noqa: E402
                          OP_INC, OP_DEC)

MODULAR_CASES = [(2, 8), (3, 4), (3, 8), (5, 4), (6, 4), (7, 3), (2, 14), (3, 14), (7, 14)]
FIELD_CASES = [(4, 1), (4, 2), (4, 4), (4, 8), (4, 14)]
OPS = [OP_ADD, OP_SUB, OP_MUL, OP_NEG, OP_INC, OP_DEC]


def _pairs(mod: int, limit: int = 20000, seed: int = 1):
    """All (a, b) pairs when the space is small, otherwise `limit` seeded random pairs + edges."""
    if mod * mod <= limit:
        yield from itertools.product(range(mod), repeat=2)
        return
    rnd = random.Random(seed)
    edges = [0, 1, mod - 2, mod - 1, mod // 2]
    for a in edges:
        for b in edges:
            yield a, b
    for _ in range(limit):
        yield rnd.randrange(mod), rnd.randrange(mod)


# ---------------------------------------------------------------------------
# Family M: Z / k^n Z with carry — oracle: Python integers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('k,n', MODULAR_CASES)
def test_modular_family_matches_integer_arithmetic(k, n):
    info = resolve_logic_type(k)
    assert info['category'] in ('prime_field', 'integer_ring')
    g = GoldenModel(k, n)
    mod = k ** n
    assert g.mod == mod
    for a, b in _pairs(mod):
        r = g.execute(OP_ADD, a, b)
        assert (r.result, r.carry) == ((a + b) % mod, a + b >= mod)
        r = g.execute(OP_SUB, a, b)
        assert (r.result, r.carry) == ((a - b) % mod, a < b)
        r = g.execute(OP_MUL, a, b)
        assert (r.result, r.carry) == ((a * b) % mod, False)
        r = g.execute(OP_NEG, a)
        assert (r.result, r.carry) == ((-a) % mod, False)
        r = g.execute(OP_INC, a)
        assert (r.result, r.carry) == ((a + 1) % mod, a == mod - 1)
        r = g.execute(OP_DEC, a)
        assert (r.result, r.carry) == ((a - 1) % mod, a == 0)
        # flags are a pure function of the result
        for op in OPS:
            res = g.execute(op, a, b)
            assert res.zero == (res.result == 0)
            assert res.negative == (res.result >= mod // 2)


@pytest.mark.parametrize('k,n', [(2, 4), (3, 3), (5, 2), (6, 2)])
def test_modular_family_ring_axioms(k, n):
    """Exhaustive ring laws on small instances — independent of any oracle."""
    g = GoldenModel(k, n)
    mod = k ** n
    add = lambda x, y: g.execute(OP_ADD, x, y).result
    mul = lambda x, y: g.execute(OP_MUL, x, y).result
    for a, b, c in itertools.product(range(mod), repeat=3):
        assert add(add(a, b), c) == add(a, add(b, c))
        assert mul(mul(a, b), c) == mul(a, mul(b, c))
        assert mul(a, add(b, c)) == add(mul(a, b), mul(a, c))
    for a, b in itertools.product(range(mod), repeat=2):
        assert add(a, b) == add(b, a) and mul(a, b) == mul(b, a)
        assert g.execute(OP_SUB, add(a, b), b).result == a            # SUB undoes ADD
    for a in range(mod):
        assert add(a, 0) == a and mul(a, 1) == a and mul(a, 0) == 0
        assert add(a, g.execute(OP_NEG, a).result) == 0                # additive inverse
        assert g.execute(OP_NEG, a).result == g.execute(OP_SUB, 0, a).result
        assert g.execute(OP_INC, a).result == add(a, 1)
        assert g.execute(OP_DEC, a).result == g.execute(OP_SUB, a, 1).result


def test_k2_is_ordinary_binary_arithmetic():
    """k = 2 must degenerate to the familiar unsigned binary ALU."""
    g = GoldenModel(2, 8)
    for a, b in itertools.product(range(256), repeat=2):
        r = g.execute(OP_ADD, a, b)
        assert r.result == (a + b) & 0xFF and r.carry == ((a + b) >> 8 == 1)
        assert g.execute(OP_MUL, a, b).result == (a * b) & 0xFF


# ---------------------------------------------------------------------------
# Family F: GF(q)[x]/(x^n) — oracle: the `galois` package + a plain polynomial ring
# ---------------------------------------------------------------------------

galois = pytest.importorskip('galois')


@pytest.mark.parametrize('p,m', [(2, 2), (2, 3), (2, 4), (3, 2)])
def test_field_tables_match_galois_package(p, m):
    """Addition/multiplication tables of GF(p^m) equal those of an independent library,
    provided both use the same irreducible polynomial (checked explicitly)."""
    ours = compute_gf_tables(p, m)
    q = p ** m
    coeffs_desc = list(reversed(ours['irreducible_coeffs']))          # ours are [a0..an]
    poly = galois.Poly(coeffs_desc, field=galois.GF(p))
    GF = galois.GF(q, irreducible_poly=poly)
    A = GF(list(range(q)))
    for a in range(q):
        row_add = (GF(a) + A).tolist()
        row_mul = (GF(a) * A).tolist()
        assert ours['add_table'][a] == row_add, f'GF({q}) add row {a}'
        assert ours['mul_table'][a] == row_mul, f'GF({q}) mul row {a}'


def test_field_tables_are_a_field():
    """Axioms directly: GF(4) tables form a field (every non-zero element invertible)."""
    t = compute_gf_tables(2, 2)
    add, mul = t['add_table'], t['mul_table']
    q = 4
    for a, b, c in itertools.product(range(q), repeat=3):
        assert add[add[a][b]][c] == add[a][add[b][c]]
        assert mul[mul[a][b]][c] == mul[a][mul[b][c]]
        assert mul[a][add[b][c]] == add[mul[a][b]][mul[a][c]]
    for a in range(q):
        assert add[a][0] == a and mul[a][1] == a and mul[a][0] == 0
        assert any(add[a][x] == 0 for x in range(q))
        if a:
            assert any(mul[a][x] == 1 for x in range(q))


def _poly_ring_oracle(k, n):
    """Independent reference for family F: words as coefficient lists over GF(k),
    using the galois package for digit arithmetic."""
    info = resolve_logic_type(k)
    t = compute_gf_tables(info['p'], info['n'])
    coeffs_desc = list(reversed(t['irreducible_coeffs']))
    GF = galois.GF(k, irreducible_poly=galois.Poly(coeffs_desc, field=galois.GF(info['p'])))

    def digits(v):
        return [(v // k ** i) % k for i in range(n)]

    def word(ds):
        return sum(int(d) * k ** i for i, d in enumerate(ds))

    def op(o, a, b):
        A, B = GF(digits(a)), GF(digits(b))
        if o == OP_ADD:
            return word(A + B)
        if o == OP_SUB:
            return word(A - B)
        if o == OP_NEG:
            return word(-A)
        if o == OP_INC:
            return word(A + GF([1] + [0] * (n - 1)))
        if o == OP_DEC:
            return word(A - GF([1] + [0] * (n - 1)))
        if o == OP_MUL:  # polynomial product truncated to n coefficients
            out = GF([0] * n)
            for i in range(n):
                for j in range(n - i):
                    out[i + j] = out[i + j] + A[i] * B[j]
            return word(out)
        raise ValueError(o)
    return op


@pytest.mark.parametrize('k,n', FIELD_CASES)
def test_field_family_matches_polynomial_ring(k, n):
    assert resolve_logic_type(k)['category'] == 'extension_field'
    g = GoldenModel(k, n)
    oracle = _poly_ring_oracle(k, n)
    for a, b in _pairs(k ** n, limit=5000):
        for o in OPS:
            r = g.execute(o, a, b)
            assert r.result == oracle(o, a, b), (k, n, o, a, b)
            assert r.carry is False and r.negative is False
            assert r.zero == (r.result == 0)


@pytest.mark.parametrize('k,n', [(4, 2), (4, 3)])
def test_field_family_ring_axioms(k, n):
    g = GoldenModel(k, n)
    mod = k ** n
    add = lambda x, y: g.execute(OP_ADD, x, y).result
    mul = lambda x, y: g.execute(OP_MUL, x, y).result
    for a, b, c in itertools.product(range(mod), repeat=3):
        assert add(add(a, b), c) == add(a, add(b, c))
        assert mul(mul(a, b), c) == mul(a, mul(b, c))
        assert mul(a, add(b, c)) == add(mul(a, b), mul(a, c))
    for a, b in itertools.product(range(mod), repeat=2):
        assert add(a, b) == add(b, a) and mul(a, b) == mul(b, a)
        assert g.execute(OP_SUB, add(a, b), b).result == a
    for a in range(mod):
        assert add(a, g.execute(OP_NEG, a).result) == 0
        assert g.execute(OP_INC, a).result == add(a, 1)
        assert g.execute(OP_DEC, a).result == g.execute(OP_SUB, a, 1).result
        assert mul(a, 1) == a   # the word "1" (digit 0 = 1) is the multiplicative identity


# ---------------------------------------------------------------------------
# Test-vector generation is deterministic (the "random(N, seed)" strength claim)
# ---------------------------------------------------------------------------

def test_vector_generation_is_deterministic():
    g = GoldenModel(3, 8)
    v1 = [(v.op, v.a, v.b, v.expected.result) for v in g.generate_test_vectors(random_count=50, seed=42)]
    v2 = [(v.op, v.a, v.b, v.expected.result) for v in GoldenModel(3, 8).generate_test_vectors(random_count=50, seed=42)]
    assert v1 == v2
    assert len(v1) >= 50
    v3 = [(v.op, v.a, v.b) for v in g.generate_test_vectors(random_count=50, seed=43)]
    assert [(o, a, b) for o, a, b, _ in v1] != v3
