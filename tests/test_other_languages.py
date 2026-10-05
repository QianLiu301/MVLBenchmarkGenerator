"""Submissions in a language without a checker.

Any language may be submitted. Files in the six checked languages keep the full
rules; a file in any other language is linted as text only, never run, and is
published marked NO_CHECKER (never counted as verified). No database is touched.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from library import submissions                                     # noqa: E402
from library.models import (LANGUAGES, NO_CHECKER, Implementation,  # noqa: E402
                            checked_language)
from library.service import filename_for                             # noqa: E402

EXAMPLE = ROOT / 'web' / 'static' / 'examples' / 'alu_k3_8t'
CHISEL = 'class MvlAlu3x8 extends Module {\n  // ...\n}\n'


def _manifest(files):
    m = json.loads((EXAMPLE / 'manifest.json').read_text(encoding='utf-8'))
    m = copy.deepcopy(m)
    m['files'] = files
    return m


def test_checked_language_names():
    assert checked_language('verilog') == 'verilog'
    assert checked_language('Verilog') == 'verilog'
    assert checked_language('SystemVerilog') == 'systemverilog'
    assert checked_language('py') == 'python'
    assert checked_language('Chisel') is None
    assert checked_language('C++') is None
    assert LANGUAGES['Chisel'] == 'Chisel'          # shown under its own name


def test_other_language_is_accepted():
    m = _manifest([{'filename': 'alu_k3_8t.scala', 'language': 'Chisel'}])
    assert submissions.validate_manifest(m, {'alu_k3_8t.scala': CHISEL}) == []
    # linted as text only: no main()/module requirement, no forbidden-construct list
    assert submissions.lint_file('alu_k3_8t.scala', CHISEL, 'Chisel') == []
    assert submissions.lint_file('x.scala', 'a\x00b', 'Chisel')


def test_checked_extension_cannot_be_declared_as_other_language():
    m = _manifest([{'filename': 'alu_k3_8t.v', 'language': 'Verilog-2005'}])
    errs = submissions.validate_manifest(m, {'alu_k3_8t.v': 'module m; endmodule'})
    assert errs and 'checked as Verilog' in errs[0]


def test_checked_language_keeps_its_extension():
    m = _manifest([{'filename': 'alu_k3_8t.txt', 'language': 'Verilog'}])
    errs = submissions.validate_manifest(m, {'alu_k3_8t.txt': 'module m; endmodule'})
    assert errs and 'expects extension .v' in errs[0]


def test_filename_and_state():
    assert filename_for('alu_k3_8t', 'Chisel', '.scala') == 'alu_k3_8t.scala'
    assert filename_for('alu_k3_8t', 'verilog', '.txt') == 'alu_k3_8t.v'
    assert Implementation(golden_status=NO_CHECKER).state == 'unchecked'
    assert Implementation(golden_status='PASS').state == 'verified'
    assert Implementation(golden_status='LOGIC_ERROR').state == 'unverified'
