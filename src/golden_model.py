"""
MVL Golden Reference Model
==========================
Provides mathematically verified ALU results for any (k, bits, op, a, b).

This module serves as the single source of truth for benchmark validation.
All LLM-generated code outputs are compared against this golden model.

Supports:
- Prime fields GF(p): standard modular arithmetic
- Extension fields GF(p^n): digit-wise arithmetic using pre-computed tables
- Integer rings Z/kZ: standard modular arithmetic
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

try:
    from src.galois_field import resolve_logic_type
except ImportError:
    from galois_field import resolve_logic_type


# Operation codes — must match the LLM-generated code conventions
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_NEG = 3
OP_INC = 4
OP_DEC = 5

OP_NAMES = {
    OP_ADD: 'ADD', OP_SUB: 'SUB', OP_MUL: 'MUL',
    OP_NEG: 'NEG', OP_INC: 'INC', OP_DEC: 'DEC',
}
NAME_TO_OP = {v: k for k, v in OP_NAMES.items()}


@dataclass(frozen=True)
class ALUResult:
    """Result of a single ALU operation."""
    result: int
    zero: bool
    negative: bool
    carry: bool


@dataclass(frozen=True)
class TestVector:
    """A single test case with inputs and expected outputs."""
    op: int
    a: int
    b: int
    expected: ALUResult

    @property
    def op_name(self) -> str:
        return OP_NAMES.get(self.op, f'OP{self.op}')


class GoldenModel:
    """Golden reference ALU model for a specific (k, bits) configuration."""

    def __init__(self, k: int, bits: int):
        self.k = k
        self.bits = bits
        self.mod = k ** bits
        self.max_val = self.mod - 1
        self.logic_info = resolve_logic_type(k)
        self.category = self.logic_info['category']

        # Extension field setup
        self._is_extension = (
            self.category == 'extension_field'
            and self.logic_info.get('tables') is not None
        )
        if self._is_extension:
            tables = self.logic_info['tables']
            self._add_tbl = tables['add_table']
            self._mul_tbl = tables['mul_table']

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, op: int, a: int, b: int = 0) -> ALUResult:
        """Execute a single ALU operation. This is the core golden model."""
        if self._is_extension:
            return self._execute_extension(op, a, b)
        else:
            return self._execute_modular(op, a, b)

    def generate_test_vectors(
        self,
        ops: List[int] = None,
        random_count: int = 50,
        seed: int = 42,
    ) -> List[TestVector]:
        """Generate a comprehensive set of test vectors.

        Includes:
        - Edge cases: (0,0) and (max,max) for every operation
        - Small values: (1,1), (10,20) for common pitfalls
        - Random values: random_count additional random tests
        """
        if ops is None:
            ops = [OP_ADD, OP_SUB, OP_MUL, OP_NEG, OP_INC, OP_DEC]

        vectors: List[TestVector] = []

        # --- Deterministic edge cases ---
        edge_pairs = [
            (0, 0),
            (self.max_val, self.max_val),
            (0, self.max_val),
            (self.max_val, 0),
            (1, 1),
            (1, 0),
            (0, 1),
        ]
        # Small value pairs (if mod is large enough)
        if self.mod > 20:
            edge_pairs += [(10, 20), (20, 10)]
        if self.mod > 100:
            edge_pairs += [(50, 75), (99, 1)]

        for op in ops:
            for a, b in edge_pairs:
                expected = self.execute(op, a, b)
                vectors.append(TestVector(op=op, a=a, b=b, expected=expected))

        # --- Random values ---
        rng = random.Random(seed)
        for _ in range(random_count):
            op = rng.choice(ops)
            a = rng.randint(0, self.max_val)
            b = rng.randint(0, self.max_val)
            expected = self.execute(op, a, b)
            vectors.append(TestVector(op=op, a=a, b=b, expected=expected))

        return vectors

    # ------------------------------------------------------------------
    # Modular arithmetic (prime field / integer ring)
    # ------------------------------------------------------------------

    def _execute_modular(self, op: int, a: int, b: int) -> ALUResult:
        mod = self.mod
        half = mod // 2
        result = 0
        carry = False

        if op == OP_ADD:
            raw = a + b
            carry = raw >= mod
            result = raw % mod
        elif op == OP_SUB:
            carry = a < b
            result = (a - b + mod) % mod
        elif op == OP_MUL:
            result = (a * b) % mod
            carry = False
        elif op == OP_NEG:
            result = (mod - a) % mod
            carry = False
        elif op == OP_INC:
            carry = (a == self.max_val)
            result = (a + 1) % mod
        elif op == OP_DEC:
            carry = (a == 0)
            result = (a - 1 + mod) % mod
        else:
            raise ValueError(f"Unknown operation: {op}")

        return ALUResult(
            result=result,
            zero=(result == 0),
            negative=(result >= half),
            carry=carry,
        )

    # ------------------------------------------------------------------
    # Extension field digit-wise arithmetic
    # ------------------------------------------------------------------

    def _to_digits(self, value: int) -> List[int]:
        digits = []
        for _ in range(self.bits):
            digits.append(value % self.k)
            value //= self.k
        return digits

    def _from_digits(self, digits: List[int]) -> int:
        value = 0
        for i, d in enumerate(digits):
            value += d * (self.k ** i)
        return value

    def _gf_add_inv(self, d: int) -> int:
        """Find additive inverse of d in GF(k): x such that GF_ADD[x][d] == 0."""
        for x in range(self.k):
            if self._add_tbl[x][d] == 0:
                return x
        raise ValueError(f"No additive inverse for {d} in GF({self.k})")

    def _gf_add_digits(self, ad: List[int], bd: List[int]) -> List[int]:
        return [self._add_tbl[ad[i]][bd[i]] for i in range(self.bits)]

    def _gf_sub_digits(self, ad: List[int], bd: List[int]) -> List[int]:
        """SUB: for each digit, find x where GF_ADD[x][bd[i]] == ad[i]."""
        result = []
        for i in range(self.bits):
            for x in range(self.k):
                if self._add_tbl[x][bd[i]] == ad[i]:
                    result.append(x)
                    break
        return result

    def _gf_mul_digits(self, ad: List[int], bd: List[int]) -> List[int]:
        """Polynomial multiplication using GF tables, truncated to bits digits."""
        result = [0] * self.bits
        for i in range(self.bits):
            for j in range(self.bits):
                if i + j < self.bits:
                    product = self._mul_tbl[ad[i]][bd[j]]
                    result[i + j] = self._add_tbl[result[i + j]][product]
        return result

    def _execute_extension(self, op: int, a: int, b: int) -> ALUResult:
        ad = self._to_digits(a)
        bd = self._to_digits(b)

        if op == OP_ADD:
            rd = self._gf_add_digits(ad, bd)
        elif op == OP_SUB:
            rd = self._gf_sub_digits(ad, bd)
        elif op == OP_MUL:
            rd = self._gf_mul_digits(ad, bd)
        elif op == OP_NEG:
            rd = [self._gf_add_inv(d) for d in ad]
        elif op == OP_INC:
            rd = list(ad)
            rd[0] = self._add_tbl[rd[0]][1]
        elif op == OP_DEC:
            inv1 = self._gf_add_inv(1)
            rd = list(ad)
            rd[0] = self._add_tbl[rd[0]][inv1]
        else:
            raise ValueError(f"Unknown operation: {op}")

        result = self._from_digits(rd)
        # Extension fields: carry and negative are always false
        return ALUResult(
            result=result,
            zero=(result == 0),
            negative=False,
            carry=False,
        )


# ------------------------------------------------------------------
# Test vector serialization for Strategy B (stdin injection)
# ------------------------------------------------------------------

def serialize_vectors(vectors: List[TestVector]) -> str:
    """Serialize test vectors to a line-based text format for stdin injection.

    Format (one line per vector):
        OP A B EXPECTED_RESULT EXPECTED_ZERO EXPECTED_NEG EXPECTED_CARRY
    First line: count of vectors.
    """
    lines = [str(len(vectors))]
    for v in vectors:
        z = 1 if v.expected.zero else 0
        n = 1 if v.expected.negative else 0
        c = 1 if v.expected.carry else 0
        lines.append(f"{v.op} {v.a} {v.b} {v.expected.result} {z} {n} {c}")
    return '\n'.join(lines) + '\n'


def deserialize_vectors(text: str) -> List[TestVector]:
    """Deserialize test vectors from the text format produced by serialize_vectors."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    count = int(lines[0])
    vectors = []
    for line in lines[1:count + 1]:
        parts = line.split()
        op, a, b = int(parts[0]), int(parts[1]), int(parts[2])
        result, z, n, c = int(parts[3]), bool(int(parts[4])), bool(int(parts[5])), bool(int(parts[6]))
        vectors.append(TestVector(
            op=op, a=a, b=b,
            expected=ALUResult(result=result, zero=z, negative=n, carry=c),
        ))
    return vectors


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def golden_alu(k: int, bits: int, op: int, a: int, b: int = 0) -> ALUResult:
    """One-shot golden model execution. Creates a model instance each call.

    For batch operations, create a GoldenModel instance directly.
    """
    model = GoldenModel(k, bits)
    return model.execute(op, a, b)
