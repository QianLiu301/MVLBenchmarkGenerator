"""Example submission: a ternary ALU with 8 digits (benchmark alu_k3_8t).

The numbers are 0 .. 3^8 - 1 = 6560, and every operation wraps around modulo 3^8.
"""
from collections import namedtuple

K, DIGITS = 3, 8
MOD = K ** DIGITS        # 6561 values: 0 .. 6560
HALF = MOD // 2          # a result >= HALF counts as negative (flag N)

ADD, SUB, MUL, NEG, INC, DEC = range(6)
NAMES = ['ADD', 'SUB', 'MUL', 'NEG', 'INC', 'DEC']
Flags = namedtuple('Flags', 'z n c')


def alu_exec(a, b, op):
    """One operation: returns (result, Flags(z, n, c))."""
    carry = False
    if op == ADD:
        r = a + b
        carry = r >= MOD             # the sum wrapped around
    elif op == SUB:
        r = a - b
        carry = a < b                # a borrow was needed
    elif op == MUL:
        r = a * b
    elif op == NEG:
        r = -a
    elif op == INC:
        r = a + 1
        carry = a == MOD - 1
    elif op == DEC:
        r = a - 1
        carry = a == 0
    else:
        raise ValueError(f'unknown operation {op}')
    r %= MOD
    return r, Flags(z=int(r == 0), n=int(r >= HALF), c=int(carry))


if __name__ == '__main__':
    # The file's own tests: one line per test, in the format the library reads.
    pairs = [(0, 0), (MOD - 1, MOD - 1), (1, 2), (100, 6500)]
    tests = [(op, a, b) for op in range(6) for a, b in pairs]
    for i, (op, a, b) in enumerate(tests, 1):
        r, f = alu_exec(a, b, op)
        print(f'Test {i}: {NAMES[op]} A={a} B={b} -> R={r} Z={f.z} N={f.n} C={f.c}')
