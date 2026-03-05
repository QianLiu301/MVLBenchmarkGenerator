"""
MVL Benchmark Validator
=======================
Strategy A: Compile/run LLM-generated code, parse its self-test output,
            and compare against the golden model.

Pipeline:
  code file → compile → run → parse stdout → compare with golden → report
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from src.golden_model import GoldenModel, ALUResult, OP_NAMES, NAME_TO_OP
    from src.mvl_simulation_runner import MVLSimulationRunner
except ImportError:
    from golden_model import GoldenModel, ALUResult, OP_NAMES, NAME_TO_OP
    from mvl_simulation_runner import MVLSimulationRunner


@dataclass
class ParsedTestLine:
    """A single test result parsed from LLM-generated code output."""
    line_num: int
    op_name: str
    a: int
    b: int
    result: int
    zero: Optional[bool] = None
    negative: Optional[bool] = None
    carry: Optional[bool] = None
    raw_line: str = ''


@dataclass
class TestComparison:
    """Comparison of one parsed test against the golden model."""
    parsed: ParsedTestLine
    expected: ALUResult
    result_match: bool
    zero_match: Optional[bool]      # None if flag not parsed
    negative_match: Optional[bool]
    carry_match: Optional[bool]

    @property
    def passed(self) -> bool:
        """A test passes if result matches. Flag mismatches are warnings."""
        return self.result_match


@dataclass
class ValidationReport:
    """Full validation report for one generated code file."""
    file_path: str
    language: str
    k: int
    bits: int

    # Compilation / execution
    compile_success: bool = False
    compile_errors: str = ''
    run_success: bool = False
    run_errors: str = ''
    run_output: str = ''
    compile_time: float = 0.0
    run_time: float = 0.0

    # Parsing
    parsed_tests: List[ParsedTestLine] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)

    # Golden model comparison
    comparisons: List[TestComparison] = field(default_factory=list)

    @property
    def total_parsed(self) -> int:
        return len(self.parsed_tests)

    @property
    def total_compared(self) -> int:
        return len(self.comparisons)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.comparisons if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.comparisons if not c.passed)

    @property
    def flag_warnings(self) -> int:
        """Count tests where result matches but at least one flag doesn't."""
        count = 0
        for c in self.comparisons:
            if c.passed:
                flags = [c.zero_match, c.negative_match, c.carry_match]
                if any(f is False for f in flags):
                    count += 1
        return count

    @property
    def status(self) -> str:
        if not self.compile_success:
            return 'COMPILE_ERROR'
        if not self.run_success:
            return 'RUNTIME_ERROR'
        if self.total_parsed == 0:
            return 'NO_OUTPUT'
        if self.failed > 0:
            return 'LOGIC_ERROR'
        return 'PASS'

    def summary(self) -> Dict:
        return {
            'file': self.file_path,
            'language': self.language,
            'k': self.k,
            'bits': self.bits,
            'status': self.status,
            'compile_success': self.compile_success,
            'compile_errors': self.compile_errors,
            'run_success': self.run_success,
            'run_errors': self.run_errors,
            'compile_time': self.compile_time,
            'run_time': self.run_time,
            'total_parsed': self.total_parsed,
            'total_compared': self.total_compared,
            'passed': self.passed,
            'failed': self.failed,
            'flag_warnings': self.flag_warnings,
            'parse_warnings': self.parse_warnings,
            'failures': [
                {
                    'op': c.parsed.op_name,
                    'a': c.parsed.a,
                    'b': c.parsed.b,
                    'got': c.parsed.result,
                    'expected': c.expected.result,
                    'line': c.parsed.raw_line.strip(),
                }
                for c in self.comparisons if not c.passed
            ],
        }


class BenchmarkValidator:
    """Validates LLM-generated MVL code against the golden model."""

    def __init__(self, project_root: str = None):
        self.runner = MVLSimulationRunner(project_root=project_root)

    def validate(
        self,
        file_path: str,
        k: int,
        bits: int,
        language: str = None,
    ) -> ValidationReport:
        """Full validation pipeline: compile → run → parse → compare."""
        fp = Path(file_path)
        if language is None:
            language = {'.c': 'c', '.py': 'python', '.v': 'verilog', '.vhd': 'vhdl'
                        }.get(fp.suffix.lower(), 'unknown')

        report = ValidationReport(
            file_path=str(fp),
            language=language,
            k=k,
            bits=bits,
        )

        # Step 1: Compile and run
        sim_result = self.runner.run_simulation(str(fp), language)

        report.compile_time = sim_result.get('compile_time', 0.0)
        report.run_time = sim_result.get('run_time', 0.0)

        errors = sim_result.get('errors', [])
        if errors:
            first_error = errors[0]
            if 'Compilation failed' in first_error or 'Compilation error' in first_error:
                report.compile_success = False
                report.compile_errors = '\n'.join(errors)
                return report
            else:
                report.compile_success = True
                report.run_success = False
                report.run_errors = '\n'.join(errors)
                # Still try to parse any partial output
        else:
            report.compile_success = True

        report.run_success = sim_result.get('success', False)
        output = sim_result.get('output', '')
        report.run_output = output

        if not output.strip():
            if report.run_success:
                report.parse_warnings.append('Program ran successfully but produced no output')
            return report

        # Step 2: Parse output
        report.parsed_tests = self._parse_output(output, language, report)

        # Step 3: Compare with golden model
        if report.parsed_tests:
            golden = GoldenModel(k, bits)
            report.comparisons = self._compare(report.parsed_tests, golden)

        return report

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    # Patterns for different output formats LLMs commonly produce
    _PATTERNS = [
        # "Test  1: ADD A=0 B=0 -> R=0 Z=1 N=0 C=0"
        re.compile(
            r'Test\s*\d+\s*:\s*(?P<op>\w+)\s+'
            r'A\s*=\s*(?P<a>\d+)\s+'
            r'B\s*=\s*(?P<b>\d+)\s*'
            r'(?:->|=>|:)\s*'
            r'R\s*=\s*(?P<r>\d+)'
            r'(?:\s+Z\s*=\s*(?P<z>[01]))?'
            r'(?:\s+N\s*=\s*(?P<n>[01]))?'
            r'(?:\s+C\s*=\s*(?P<c>[01]))?',
            re.IGNORECASE,
        ),
        # "ADD(0, 0) = 0" or "ADD(0, 0) -> 0"
        re.compile(
            r'(?P<op>ADD|SUB|MUL|NEG|INC|DEC)\s*\(\s*(?P<a>\d+)\s*'
            r'(?:,\s*(?P<b>\d+)\s*)?\)\s*'
            r'(?:=|->|=>)\s*(?P<r>\d+)',
            re.IGNORECASE,
        ),
        # "op=ADD a=0 b=0 result=0" (flexible key=value)
        re.compile(
            r'op\s*=\s*(?P<op>\w+)\s+'
            r'a\s*=\s*(?P<a>\d+)\s+'
            r'b\s*=\s*(?P<b>\d+)\s+'
            r'result\s*=\s*(?P<r>\d+)'
            r'(?:\s+zero\s*=\s*(?P<z>[01]))?'
            r'(?:\s+neg(?:ative)?\s*=\s*(?P<n>[01]))?'
            r'(?:\s+carry\s*=\s*(?P<c>[01]))?',
            re.IGNORECASE,
        ),
        # Verilog $display: "Test  1: OP=0 A=  0 B=  0 R=  0 Z=1 N=0 C=0"
        re.compile(
            r'Test\s*\d+\s*:\s*OP\s*=\s*(?P<opcode>\d+)\s+'
            r'A\s*=\s*(?P<a>\d+)\s+'
            r'B\s*=\s*(?P<b>\d+)\s+'
            r'R\s*=\s*(?P<r>\d+)'
            r'(?:\s+Z\s*=\s*(?P<z>[01]))?'
            r'(?:\s+N\s*=\s*(?P<n>[01]))?'
            r'(?:\s+C\s*=\s*(?P<c>[01]))?',
            re.IGNORECASE,
        ),
        # VHDL report: "Test 1: ADD A=0 B=0 -> R=0"
        # (same as first pattern, already covered)
    ]

    _OPCODE_TO_NAME = {
        '0': 'ADD', '1': 'SUB', '2': 'MUL',
        '3': 'NEG', '4': 'INC', '5': 'DEC',
    }

    def _parse_output(
        self, output: str, language: str, report: ValidationReport,
    ) -> List[ParsedTestLine]:
        """Parse structured test output from LLM-generated code."""
        results = []
        lines = output.split('\n')

        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue

            parsed = self._try_parse_line(line)
            if parsed is not None:
                parsed.line_num = line_num
                parsed.raw_line = line
                results.append(parsed)

        if not results:
            # Try fallback: look for any numbers that might be test output
            report.parse_warnings.append(
                f'Could not parse any test lines from {len(lines)} output lines. '
                f'Expected format: "Test N: OP A=X B=Y -> R=Z Z=0 N=0 C=0"'
            )

        return results

    def _try_parse_line(self, line: str) -> Optional[ParsedTestLine]:
        """Try to parse a single output line using known patterns."""
        for pattern in self._PATTERNS:
            m = pattern.search(line)
            if m:
                groups = m.groupdict()

                # Resolve op name
                op_name = groups.get('op', '').upper()
                if not op_name and 'opcode' in groups:
                    op_name = self._OPCODE_TO_NAME.get(groups['opcode'], '')
                if op_name not in NAME_TO_OP:
                    continue

                a = int(groups['a'])
                b = int(groups.get('b') or '0')
                result = int(groups['r'])

                z = bool(int(groups['z'])) if groups.get('z') is not None else None
                n = bool(int(groups['n'])) if groups.get('n') is not None else None
                c = bool(int(groups['c'])) if groups.get('c') is not None else None

                return ParsedTestLine(
                    line_num=0, op_name=op_name,
                    a=a, b=b, result=result,
                    zero=z, negative=n, carry=c,
                )
        return None

    # ------------------------------------------------------------------
    # Golden model comparison
    # ------------------------------------------------------------------

    def _compare(
        self, parsed: List[ParsedTestLine], golden: GoldenModel,
    ) -> List[TestComparison]:
        """Compare parsed test results against the golden model."""
        comparisons = []

        for p in parsed:
            op = NAME_TO_OP.get(p.op_name)
            if op is None:
                continue

            expected = golden.execute(op, p.a, p.b)

            result_match = (p.result == expected.result)
            zero_match = (p.zero == expected.zero) if p.zero is not None else None
            neg_match = (p.negative == expected.negative) if p.negative is not None else None
            carry_match = (p.carry == expected.carry) if p.carry is not None else None

            comparisons.append(TestComparison(
                parsed=p,
                expected=expected,
                result_match=result_match,
                zero_match=zero_match,
                negative_match=neg_match,
                carry_match=carry_match,
            ))

        return comparisons


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Validate MVL benchmark code')
    parser.add_argument('file', help='Code file to validate')
    parser.add_argument('-k', type=int, required=True, help='K-value')
    parser.add_argument('-b', '--bits', type=int, required=True, help='Bitwidth')
    parser.add_argument('--lang', help='Language (auto-detect from extension)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    validator = BenchmarkValidator()
    report = validator.validate(args.file, args.k, args.bits, args.lang)

    if args.json:
        print(json.dumps(report.summary(), indent=2))
    else:
        _print_report(report)


def _print_report(report: ValidationReport):
    status_icons = {
        'PASS': '✅', 'COMPILE_ERROR': '❌', 'RUNTIME_ERROR': '💥',
        'NO_OUTPUT': '⚠️', 'LOGIC_ERROR': '🔴',
    }
    icon = status_icons.get(report.status, '❓')

    print(f"\n{'=' * 60}")
    print(f"{icon} Validation: {report.status}")
    print(f"   File: {report.file_path}")
    print(f"   Config: k={report.k}, bits={report.bits}, lang={report.language}")
    print(f"   Compile: {'OK' if report.compile_success else 'FAIL'} ({report.compile_time}s)")
    print(f"   Run: {'OK' if report.run_success else 'FAIL'} ({report.run_time}s)")

    if report.compile_errors:
        print(f"\n   Compile errors:\n   {report.compile_errors[:500]}")
    if report.run_errors:
        print(f"\n   Runtime errors:\n   {report.run_errors[:500]}")

    if report.total_parsed > 0:
        print(f"\n   Tests parsed: {report.total_parsed}")
        print(f"   Compared:     {report.total_compared}")
        print(f"   Passed:       {report.passed}")
        print(f"   Failed:       {report.failed}")
        if report.flag_warnings:
            print(f"   Flag warnings: {report.flag_warnings}")

    if report.failed > 0:
        print(f"\n   Failures:")
        for c in report.comparisons:
            if not c.passed:
                p = c.parsed
                print(f"     {p.op_name}({p.a}, {p.b}): got {p.result}, expected {c.expected.result}")

    for w in report.parse_warnings:
        print(f"   ⚠️ {w}")

    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
