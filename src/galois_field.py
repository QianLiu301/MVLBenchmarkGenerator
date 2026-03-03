"""
Galois Field Arithmetic for MVL Benchmark Generator
====================================================
Pre-computes addition and multiplication tables for GF(p^n) extension fields.

Supports:
- GF(p)   : Prime fields (k=2,3,5,7,11,13) — standard mod p arithmetic
- GF(p^n) : Extension fields (k=4,8,9,16) — polynomial arithmetic over irreducible poly
- Z/kZ    : Integer rings (k=6,10,12,14,15) — standard mod k arithmetic
"""

from typing import Dict, List, Optional, Tuple


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _factor_prime_power(k: int) -> Optional[Tuple[int, int]]:
    """If k = p^n for some prime p and n>=1, return (p, n). Otherwise None."""
    if k < 2:
        return None
    if _is_prime(k):
        return (k, 1)
    for p in range(2, int(k ** 0.5) + 1):
        if not _is_prime(p):
            continue
        val, n = p, 1
        while val < k:
            val *= p
            n += 1
        if val == k:
            return (p, n)
    return None


# Well-known irreducible polynomials over GF(p) for small extension fields.
# Each polynomial is stored as a list of coefficients [a0, a1, ..., an]
# representing a0 + a1*x + a2*x^2 + ... + an*x^n.
# The leading coefficient (an) is always 1 and included.
IRREDUCIBLE_POLYS: Dict[Tuple[int, int], List[int]] = {
    # GF(2^2) = GF(4):  x^2 + x + 1
    (2, 2): [1, 1, 1],
    # GF(2^3) = GF(8):  x^3 + x + 1
    (2, 3): [1, 1, 0, 1],
    # GF(2^4) = GF(16): x^4 + x + 1
    (2, 4): [1, 1, 0, 0, 1],
    # GF(3^2) = GF(9):  x^2 + 1
    (3, 2): [1, 0, 1],
}


def _poly_to_int(coeffs: List[int], p: int) -> int:
    """Convert polynomial coefficients [a0, a1, ..., a_{n-1}] to integer in mixed-radix."""
    result = 0
    for i, c in enumerate(coeffs):
        result += c * (p ** i)
    return result


def _int_to_poly(val: int, p: int, n: int) -> List[int]:
    """Convert integer to polynomial coefficients [a0, a1, ..., a_{n-1}] in base p."""
    coeffs = []
    for _ in range(n):
        coeffs.append(val % p)
        val //= p
    return coeffs


def _poly_add(a: List[int], b: List[int], p: int) -> List[int]:
    """Add two polynomials coefficient-wise mod p."""
    length = max(len(a), len(b))
    result = [0] * length
    for i in range(length):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        result[i] = (ai + bi) % p
    return result


def _poly_mul_mod(a: List[int], b: List[int], irr: List[int], p: int) -> List[int]:
    """Multiply two polynomials mod irreducible polynomial, coefficients mod p."""
    n = len(irr) - 1  # degree of irreducible polynomial
    # Raw multiplication
    raw = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            raw[i + j] = (raw[i + j] + ai * bj) % p

    # Reduce mod irreducible polynomial
    while len(raw) >= len(irr):
        if raw[-1] != 0:
            coeff = raw[-1]
            deg = len(raw) - 1
            for i in range(len(irr)):
                idx = deg - (n - i)
                if 0 <= idx < len(raw):
                    raw[idx] = (raw[idx] - coeff * irr[n - i]) % p
        raw.pop()

    # Pad to length n
    while len(raw) < n:
        raw.append(0)
    return raw[:n]


def compute_gf_tables(p: int, n: int) -> Dict:
    """
    Compute addition and multiplication tables for GF(p^n).

    Returns dict with:
        'add_table': k x k list of lists
        'mul_table': k x k list of lists
        'irreducible_poly': human-readable string
        'irreducible_coeffs': list of coefficients
    """
    k = p ** n
    irr_key = (p, n)
    if irr_key not in IRREDUCIBLE_POLYS:
        raise ValueError(
            f"No irreducible polynomial defined for GF({p}^{n}). "
            f"Supported: {list(IRREDUCIBLE_POLYS.keys())}"
        )

    irr = IRREDUCIBLE_POLYS[irr_key]

    add_table = [[0] * k for _ in range(k)]
    mul_table = [[0] * k for _ in range(k)]

    for a in range(k):
        pa = _int_to_poly(a, p, n)
        for b in range(k):
            pb = _int_to_poly(b, p, n)

            # GF addition: coefficient-wise mod p
            sum_poly = _poly_add(pa, pb, p)
            add_table[a][b] = _poly_to_int(sum_poly[:n], p)

            # GF multiplication: polynomial multiplication mod irreducible
            prod_poly = _poly_mul_mod(pa, pb, irr, p)
            mul_table[a][b] = _poly_to_int(prod_poly, p)

    # Build human-readable irreducible polynomial string
    irr_str = _format_poly(irr, p)

    return {
        'add_table': add_table,
        'mul_table': mul_table,
        'irreducible_poly': irr_str,
        'irreducible_coeffs': irr,
    }


def _format_poly(coeffs: List[int], p: int) -> str:
    """Format polynomial coefficients as human-readable string like 'x^2 + x + 1'."""
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append("x" if c == 1 else f"{c}x")
        else:
            terms.append(f"x^{i}" if c == 1 else f"{c}x^{i}")
    return " + ".join(terms) if terms else "0"


def resolve_logic_type(k_value: int) -> Dict:
    """
    Determine the algebraic structure for the given K-value.

    Returns a dict with:
        'display':    Human-readable string, e.g. 'GF(3)', 'GF(2^2)', 'mod 6'
        'category':   'prime_field' | 'extension_field' | 'integer_ring'
        'p':          Base prime (or None for integer_ring)
        'n':          Exponent (or None for integer_ring)
        'tables':     GF tables dict (only for extension_field, else None)
    """
    pp = _factor_prime_power(k_value)

    if pp is None:
        # Composite, not a prime power → integer ring Z/kZ
        return {
            'display': f'mod {k_value}',
            'category': 'integer_ring',
            'p': None,
            'n': None,
            'tables': None,
        }

    p, n = pp
    if n == 1:
        # Prime → prime field GF(p)
        return {
            'display': f'GF({p})',
            'category': 'prime_field',
            'p': p,
            'n': 1,
            'tables': None,
        }

    # Prime power with n > 1 → extension field GF(p^n)
    try:
        tables = compute_gf_tables(p, n)
    except ValueError:
        # No irreducible polynomial available, fall back to integer ring behavior
        return {
            'display': f'GF({p}^{n})',
            'category': 'extension_field',
            'p': p,
            'n': n,
            'tables': None,
        }

    return {
        'display': f'GF({p}^{n})',
        'category': 'extension_field',
        'p': p,
        'n': n,
        'tables': tables,
    }


def format_table_for_prompt(table: List[List[int]], name: str, k: int) -> str:
    """Format a k×k table as a readable string for embedding in LLM prompts."""
    lines = [f"{name} (k={k}):"]
    # Header
    header = "    " + "  ".join(f"{j:>2}" for j in range(k))
    lines.append(f"  {'':>2}| {header.strip()}")
    lines.append(f"  {'--':>2}+{'---' * k}")
    # Rows
    for i in range(k):
        row = "  ".join(f"{table[i][j]:>2}" for j in range(k))
        lines.append(f"  {i:>2}| {row}")
    return "\n".join(lines)
