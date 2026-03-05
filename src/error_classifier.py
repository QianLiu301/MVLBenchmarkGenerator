"""
Error Pattern Classifier
========================
Classifies counterexamples from validation failures into known error patterns.
Each failure is tagged with a pattern ID so the frontend can group and display them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from src.golden_model import (
        GoldenModel, ALUResult, TestVector,
        OP_ADD, OP_SUB, OP_MUL, OP_NEG, OP_INC, OP_DEC,
        OP_NAMES, NAME_TO_OP,
    )
except ImportError:
    from golden_model import (
        GoldenModel, ALUResult, TestVector,
        OP_ADD, OP_SUB, OP_MUL, OP_NEG, OP_INC, OP_DEC,
        OP_NAMES, NAME_TO_OP,
    )


# ------------------------------------------------------------------
# Pattern definitions
# ------------------------------------------------------------------

PATTERN_LABELS = {
    'standard_mod_on_gf': 'Used standard modular arithmetic on extension field',
    'missing_mod_wrap': 'Missing modular wrap (forgot % MOD)',
    'off_by_one': 'Off-by-one error',
    'flag_only': 'Result correct, flag(s) incorrect',
    'op_confused': 'Result matches a different operation',
    'systematic': 'Systematic failure on operation',
    'unknown': 'Unclassified error',
}

MAX_EXAMPLES_PER_PATTERN = 3


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ClassifiedFailure:
    """A single counterexample with its classified pattern."""
    pattern: str
    op: str
    a: int
    b: int
    got: int
    expected: int
    # For standard_mod_on_gf: what naive mod would give
    naive_mod: Optional[int] = None
    # For op_confused: which op it matched
    confused_with: Optional[str] = None
    # Flag details
    flags: Optional[Dict] = None


@dataclass
class PatternGroup:
    """A group of failures sharing the same error pattern."""
    pattern: str
    label: str
    count: int
    affected_ops: List[str]
    examples: List[Dict]  # max MAX_EXAMPLES_PER_PATTERN, serializable dicts


@dataclass
class OpSummary:
    """Per-operation pass/fail summary."""
    total: int = 0
    failed: int = 0
    patterns: List[str] = field(default_factory=list)


@dataclass
class StrategyComparison:
    """Cross-comparison of Strategy A vs B results."""
    a_only_pass: int = 0       # A passed but B failed
    b_only_pass: int = 0       # B passed but A failed
    both_pass: int = 0
    both_fail: int = 0
    selective_testing_examples: List[Dict] = field(default_factory=list)


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------

class ErrorPatternClassifier:
    """Classify validation failures into known error patterns."""

    def __init__(self, k: int, bits: int):
        self.k = k
        self.bits = bits
        self.mod = k ** bits
        self.golden = GoldenModel(k, bits)
        self._is_extension = self.golden._is_extension

    def classify_failures(self, comparisons: list) -> Dict:
        """Classify all failed comparisons and return structured counterexample data.

        Parameters
        ----------
        comparisons : List[TestComparison]
            From ValidationReport.comparisons

        Returns
        -------
        Dict ready for JSON serialization with keys: by_pattern, by_op
        """
        classified: List[ClassifiedFailure] = []

        for c in comparisons:
            op_code = NAME_TO_OP.get(c.parsed.op_name)
            if op_code is None:
                continue

            if c.passed:
                # Check flag-only errors
                flags_wrong = any(
                    f is False
                    for f in [c.zero_match, c.negative_match, c.carry_match]
                )
                if flags_wrong:
                    cf = ClassifiedFailure(
                        pattern='flag_only',
                        op=c.parsed.op_name,
                        a=c.parsed.a, b=c.parsed.b,
                        got=c.parsed.result, expected=c.expected.result,
                        flags=self._flag_detail(c),
                    )
                    classified.append(cf)
            else:
                cf = self._classify_one(c, op_code)
                classified.append(cf)

        # Build by_pattern groups
        by_pattern = self._group_by_pattern(classified)

        # Build by_op summary
        by_op = self._build_op_summary(comparisons, classified)

        # Mark systematic patterns
        self._mark_systematic(by_op, by_pattern)

        return {
            'by_pattern': [self._pattern_group_to_dict(pg) for pg in by_pattern],
            'by_op': {op: self._op_summary_to_dict(s) for op, s in by_op.items()},
        }

    def compare_strategies(
        self,
        report_a,  # ValidationReport from Strategy A
        report_b,  # ValidationReport from Strategy B
    ) -> Dict:
        """Cross-compare Strategy A and B results.

        Since A and B test different vectors, we compare by (op, a, b) keys.
        A's vectors are LLM-chosen; B's are golden-injected.
        The key insight: cases B tests but A doesn't reveal "selective self-testing".
        """
        # Build lookup: (op_name, a, b) -> passed?
        a_results = {}
        for c in report_a.comparisons:
            key = (c.parsed.op_name, c.parsed.a, c.parsed.b)
            a_results[key] = c.passed

        b_results = {}
        for c in report_b.comparisons:
            key = (c.parsed.op_name, c.parsed.a, c.parsed.b)
            b_results[key] = c.passed

        comp = StrategyComparison()
        selective_examples = []

        # Check B's vectors: which ones would A have caught?
        for key, b_pass in b_results.items():
            a_pass = a_results.get(key)  # None if A never tested this input

            if a_pass is None:
                # A never tested this input
                if not b_pass:
                    comp.a_only_pass += 1  # "selective" — A avoided a failing case
                    selective_examples.append({
                        'op': key[0], 'a': key[1], 'b': key[2],
                        'note': 'LLM self-test did not cover this input',
                    })
                else:
                    comp.both_pass += 1  # Would have passed anyway
            elif a_pass and b_pass:
                comp.both_pass += 1
            elif a_pass and not b_pass:
                comp.a_only_pass += 1
                selective_examples.append({
                    'op': key[0], 'a': key[1], 'b': key[2],
                    'note': 'LLM self-test passed but injection test failed',
                })
            elif not a_pass and b_pass:
                comp.b_only_pass += 1
            else:
                comp.both_fail += 1

        comp.selective_testing_examples = selective_examples[:MAX_EXAMPLES_PER_PATTERN]

        return {
            'a_only_pass': comp.a_only_pass,
            'b_only_pass': comp.b_only_pass,
            'both_pass': comp.both_pass,
            'both_fail': comp.both_fail,
            'selective_testing_examples': comp.selective_testing_examples,
        }

    # ------------------------------------------------------------------
    # Internal: single-failure classification
    # ------------------------------------------------------------------

    def _classify_one(self, c, op_code: int) -> ClassifiedFailure:
        """Classify a single failed comparison."""
        a, b, got, expected = c.parsed.a, c.parsed.b, c.parsed.result, c.expected.result

        # 1. Check standard_mod_on_gf (extension field only)
        if self._is_extension:
            naive = self._naive_mod_result(op_code, a, b)
            if naive is not None and got == naive and got != expected:
                return ClassifiedFailure(
                    pattern='standard_mod_on_gf',
                    op=c.parsed.op_name, a=a, b=b,
                    got=got, expected=expected,
                    naive_mod=naive,
                    flags=self._flag_detail(c),
                )

        # 2. Check missing_mod_wrap
        unwrapped = self._unwrapped_result(op_code, a, b)
        if unwrapped is not None and got == unwrapped and got != expected:
            return ClassifiedFailure(
                pattern='missing_mod_wrap',
                op=c.parsed.op_name, a=a, b=b,
                got=got, expected=expected,
                flags=self._flag_detail(c),
            )

        # 3. Check off_by_one
        if abs(got - expected) == 1:
            return ClassifiedFailure(
                pattern='off_by_one',
                op=c.parsed.op_name, a=a, b=b,
                got=got, expected=expected,
                flags=self._flag_detail(c),
            )

        # 4. Check op_confused
        confused_op = self._check_op_confusion(op_code, a, b, got)
        if confused_op is not None:
            return ClassifiedFailure(
                pattern='op_confused',
                op=c.parsed.op_name, a=a, b=b,
                got=got, expected=expected,
                confused_with=confused_op,
                flags=self._flag_detail(c),
            )

        # 5. Unknown
        return ClassifiedFailure(
            pattern='unknown',
            op=c.parsed.op_name, a=a, b=b,
            got=got, expected=expected,
            flags=self._flag_detail(c),
        )

    def _naive_mod_result(self, op: int, a: int, b: int) -> Optional[int]:
        """What standard modular arithmetic would give (wrong for extension fields)."""
        mod = self.mod
        if op == OP_ADD:
            return (a + b) % mod
        elif op == OP_SUB:
            return (a - b + mod) % mod
        elif op == OP_MUL:
            return (a * b) % mod
        elif op == OP_NEG:
            return (mod - a) % mod
        elif op == OP_INC:
            return (a + 1) % mod
        elif op == OP_DEC:
            return (a - 1 + mod) % mod
        return None

    def _unwrapped_result(self, op: int, a: int, b: int) -> Optional[int]:
        """What you get if you forget the final % mod."""
        mod = self.mod
        if op == OP_ADD:
            raw = a + b
            return raw if raw >= mod else None  # only relevant if it overflows
        elif op == OP_NEG:
            raw = mod - a  # if a==0 → raw==mod, should be 0
            return raw if raw == mod else None
        elif op == OP_INC:
            raw = a + 1
            return raw if raw >= mod else None
        return None

    def _check_op_confusion(self, actual_op: int, a: int, b: int, got: int) -> Optional[str]:
        """Check if `got` matches the golden result of a different operation."""
        for other_op in [OP_ADD, OP_SUB, OP_MUL, OP_NEG, OP_INC, OP_DEC]:
            if other_op == actual_op:
                continue
            try:
                other_result = self.golden.execute(other_op, a, b)
                if got == other_result.result:
                    return OP_NAMES[other_op]
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Internal: grouping and aggregation
    # ------------------------------------------------------------------

    def _flag_detail(self, c) -> Dict:
        """Build flag comparison dict for a TestComparison."""
        detail = {}
        for name, got_val, match in [
            ('zero', c.parsed.zero, c.zero_match),
            ('negative', c.parsed.negative, c.negative_match),
            ('carry', c.parsed.carry, c.carry_match),
        ]:
            exp_val = getattr(c.expected, name, None)
            detail[name] = {
                'got': got_val,
                'expected': exp_val,
                'match': match if match is not None else True,
            }
        return detail

    def _group_by_pattern(self, classified: List[ClassifiedFailure]) -> List[PatternGroup]:
        """Group classified failures by pattern, keeping top examples."""
        groups: Dict[str, List[ClassifiedFailure]] = {}
        for cf in classified:
            groups.setdefault(cf.pattern, []).append(cf)

        result = []
        for pattern, failures in groups.items():
            affected_ops = sorted(set(f.op for f in failures))
            examples = []
            for f in failures[:MAX_EXAMPLES_PER_PATTERN]:
                ex = {
                    'op': f.op, 'a': f.a, 'b': f.b,
                    'got': f.got, 'expected': f.expected,
                }
                if f.naive_mod is not None:
                    ex['naive_mod'] = f.naive_mod
                if f.confused_with is not None:
                    ex['confused_with'] = f.confused_with
                if f.flags:
                    ex['flags'] = f.flags
                examples.append(ex)

            result.append(PatternGroup(
                pattern=pattern,
                label=PATTERN_LABELS.get(pattern, pattern),
                count=len(failures),
                affected_ops=affected_ops,
                examples=examples,
            ))

        # Sort by count descending
        result.sort(key=lambda g: g.count, reverse=True)
        return result

    def _build_op_summary(
        self, comparisons: list, classified: List[ClassifiedFailure],
    ) -> Dict[str, OpSummary]:
        """Build per-operation summary."""
        summary: Dict[str, OpSummary] = {}
        for op_name in OP_NAMES.values():
            summary[op_name] = OpSummary()

        for c in comparisons:
            if c.parsed.op_name in summary:
                summary[c.parsed.op_name].total += 1
                if not c.passed:
                    summary[c.parsed.op_name].failed += 1

        # Attach pattern tags
        op_patterns: Dict[str, set] = {op: set() for op in OP_NAMES.values()}
        for cf in classified:
            if cf.op in op_patterns and cf.pattern != 'flag_only':
                op_patterns[cf.op].add(cf.pattern)

        for op_name, s in summary.items():
            s.patterns = sorted(op_patterns.get(op_name, set()))

        return summary

    def _mark_systematic(
        self, by_op: Dict[str, OpSummary], by_pattern: List[PatternGroup],
    ):
        """If an op has >80% failure rate, add 'systematic' pattern."""
        systematic_ops = []
        for op_name, s in by_op.items():
            if s.total > 0 and s.failed / s.total > 0.8:
                systematic_ops.append(op_name)
                if 'systematic' not in s.patterns:
                    s.patterns.append('systematic')

        if systematic_ops:
            # Check if we already have a systematic group
            existing = [g for g in by_pattern if g.pattern == 'systematic']
            if not existing:
                by_pattern.append(PatternGroup(
                    pattern='systematic',
                    label=PATTERN_LABELS['systematic'],
                    count=sum(by_op[op].failed for op in systematic_ops),
                    affected_ops=sorted(systematic_ops),
                    examples=[],
                ))

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pattern_group_to_dict(pg: PatternGroup) -> Dict:
        return {
            'pattern': pg.pattern,
            'label': pg.label,
            'count': pg.count,
            'affected_ops': pg.affected_ops,
            'examples': pg.examples,
        }

    @staticmethod
    def _op_summary_to_dict(s: OpSummary) -> Dict:
        return {
            'total': s.total,
            'failed': s.failed,
            'patterns': s.patterns,
        }
