"""
MVL Simulation Runner
=====================
Run simulations for MVL ALU benchmarks using gcc/python/iverilog.
"""

import subprocess
import shutil
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class MVLSimulationRunner:
    """Run MVL code simulations"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.tools = self._check_tools()

        print(f"🔧 MVL Simulation Runner initialized")
        print(f"   Project root: {self.project_root}")
        print(f"   Tools: {self.tools}")

    def _check_tools(self) -> Dict[str, bool]:
        """Check available simulation tools"""
        tools = {
            'gcc': False,
            'clang': False,
            'python': False,
            'iverilog': False,
            'vvp': False,
            'ghdl': False,
        }

        # Check C compilers (include Windows MSYS2/MinGW variants)
        # Strategy: try shutil.which first, then try running directly (Windows PATH quirks),
        # then search common Windows installation paths as fallback
        gcc_found = False
        for compiler in ['gcc', 'cc', 'mingw32-gcc', 'x86_64-w64-mingw32-gcc', 'clang']:
            if shutil.which(compiler):
                try:
                    result = subprocess.run(
                        [compiler, '--version'],
                        capture_output=True,
                        timeout=5
                    )
                    tools['gcc'] = True
                    tools['gcc_cmd'] = compiler
                    gcc_found = True
                    break
                except:
                    pass

        # Fallback: on Windows, shutil.which may not find gcc even when it's in PATH
        # Try running gcc directly via subprocess (shell=True uses CMD's PATH resolution)
        if not gcc_found and os.name == 'nt':
            for compiler in ['gcc', 'cc', 'clang']:
                try:
                    result = subprocess.run(
                        f'{compiler} --version',
                        capture_output=True,
                        timeout=5,
                        shell=True
                    )
                    if result.returncode == 0 and result.stdout:
                        tools['gcc'] = True
                        tools['gcc_cmd'] = compiler
                        gcc_found = True
                        break
                except:
                    pass

        # Fallback: search common Windows MSYS2/MinGW installation directories
        # Scan all available drive letters, not just C:
        if not gcc_found and os.name == 'nt':
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f'{letter}:\\'
                if os.path.exists(drive):
                    drives.append(letter)

            # Relative paths under each drive to check for gcc
            gcc_relative_paths = [
                r'msys64\ucrt64\bin\gcc.exe',
                r'msys64\mingw64\bin\gcc.exe',
                r'msys64\mingw32\bin\gcc.exe',
                r'msys64\usr\bin\gcc.exe',
                r'msys2\ucrt64\bin\gcc.exe',
                r'msys2\mingw64\bin\gcc.exe',
                r'msys2\mingw32\bin\gcc.exe',
                r'msys2\usr\bin\gcc.exe',
                r'MinGW\bin\gcc.exe',
                r'mingw64\bin\gcc.exe',
                r'Program Files\mingw-w64\bin\gcc.exe',
            ]
            # Also search common subdirectories (like D:\dws\msys2\...)
            search_roots = []
            for d in drives:
                drive_path = f'{d}:\\'
                search_roots.append(drive_path)
                # Also check one level of subdirectories for msys2/msys64 installs
                try:
                    for entry in os.scandir(drive_path):
                        if entry.is_dir():
                            search_roots.append(entry.path)
                except (PermissionError, OSError):
                    pass

            for root in search_roots:
                if gcc_found:
                    break
                for rel_path in gcc_relative_paths:
                    gcc_path = os.path.join(root, rel_path)
                    if os.path.isfile(gcc_path):
                        try:
                            result = subprocess.run(
                                [gcc_path, '--version'],
                                capture_output=True,
                                timeout=5
                            )
                            tools['gcc'] = True
                            tools['gcc_cmd'] = gcc_path
                            gcc_found = True
                            break
                        except:
                            pass

        # Check Python
        for python_cmd in ['python3', 'python']:
            if shutil.which(python_cmd):
                try:
                    result = subprocess.run(
                        [python_cmd, '--version'],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        tools['python'] = True
                        tools['python_cmd'] = python_cmd
                        break
                except:
                    pass

        # Check Verilog tools
        if shutil.which('iverilog'):
            try:
                result = subprocess.run(
                    ['iverilog', '-V'],
                    capture_output=True,
                    timeout=5
                )
                tools['iverilog'] = True
            except:
                pass

        if shutil.which('vvp'):
            try:
                result = subprocess.run(
                    ['vvp', '-V'],
                    capture_output=True,
                    timeout=5
                )
                tools['vvp'] = True
            except:
                pass

        # Check GHDL
        if shutil.which('ghdl'):
            try:
                result = subprocess.run(
                    ['ghdl', '--version'],
                    capture_output=True,
                    timeout=5
                )
                # Some GHDL versions return non-zero for --version; just check it runs
                tools['ghdl'] = True
            except:
                pass

        return tools

    def refresh_tools(self):
        """Re-detect available tools (useful after installing new tools)"""
        self.tools = self._detect_tools()
        return self.get_tools_status()

    def get_tools_status(self) -> Dict:
        """Get tools status for API"""
        return {
            'c_available': self.tools.get('gcc') or self.tools.get('clang'),
            'python_available': self.tools.get('python'),
            'verilog_available': self.tools.get('iverilog') and self.tools.get('vvp'),
            'vhdl_available': self.tools.get('ghdl'),
            'tools': self.tools
        }

    def can_run(self, language: str) -> bool:
        """Check if can run simulation for given language"""
        lang = language.lower()
        if lang == 'c':
            return self.tools.get('gcc') or self.tools.get('clang')
        elif lang == 'python':
            return self.tools.get('python')
        elif lang == 'verilog':
            return self.tools.get('iverilog') and self.tools.get('vvp')
        elif lang == 'vhdl':
            return self.tools.get('ghdl')
        return False

    def run_simulation(self, file_path: str, language: str = None,
                        stdin_data: str = None, vector_file: str = None) -> Dict:
        """
        Run simulation for MVL code.

        Args:
            file_path: Path to the code file
            language: Code language (auto-detect if not provided)
            stdin_data: Optional text to feed via stdin (Strategy B for C/Python/Verilog)
            vector_file: Optional path to a test-vector file (Strategy B for VHDL)

        Returns:
            Dict with simulation results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {'success': False, 'error': f'File not found: {file_path}'}

        # Auto-detect language
        if language is None:
            ext = file_path.suffix.lower()
            language = {
                '.c': 'c',
                '.py': 'python',
                '.v': 'verilog',
                '.vhd': 'vhdl'
            }.get(ext, 'unknown')

        if not self.can_run(language):
            tool_hints = {
                'c': 'Install gcc or clang',
                'python': 'Install python3',
                'verilog': 'Install iverilog and vvp (Icarus Verilog)',
                'vhdl': 'Install ghdl (GHDL VHDL simulator)'
            }
            hint = tool_hints.get(language, f'Install tools for {language}')
            return {
                'success': False,
                'language': language,
                'file': file_path.name,
                'compile_time': 0,
                'run_time': 0,
                'output': '',
                'errors': [f'No tools available for {language}. {hint}.'],
                'test_results': {'total': 0, 'passed': 0, 'failed': 0},
                'tools': self.get_tools_status()
            }

        # Run based on language
        if language == 'c':
            return self._run_c(file_path, stdin_data=stdin_data)
        elif language == 'python':
            return self._run_python(file_path, stdin_data=stdin_data)
        elif language == 'verilog':
            return self._run_verilog(file_path, stdin_data=stdin_data)
        elif language == 'vhdl':
            return self._run_vhdl(file_path, vector_file=vector_file)
        else:
            return {
                'success': False,
                'language': language,
                'file': file_path.name,
                'compile_time': 0,
                'run_time': 0,
                'output': '',
                'errors': [f'Unsupported language: {language}'],
                'test_results': {'total': 0, 'passed': 0, 'failed': 0}
            }

    def _run_c(self, file_path: Path, stdin_data: str = None) -> Dict:
        """Compile and run C code"""
        import time

        # Setup paths
        results_dir = self.project_root / 'output' / 'mvl_results'
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exe_name = f"{file_path.stem}_{timestamp}"

        # Windows vs Unix executable
        if os.name == 'nt':
            exe_file = results_dir / f"{exe_name}.exe"
        else:
            exe_file = results_dir / exe_name

        log_file = results_dir / f"{exe_name}.log"

        result = {
            'success': False,
            'language': 'c',
            'file': file_path.name,
            'compile_time': 0,
            'run_time': 0,
            'output': '',
            'errors': [],
            'test_results': {
                'total': 0,
                'passed': 0,
                'failed': 0
            }
        }

        # Select compiler (use detected command name for MSYS2/MinGW compatibility)
        compiler = self.tools.get('gcc_cmd', 'gcc')

        # Step 1: Compile
        try:
            start = time.time()

            compile_cmd = [
                compiler,
                '-o', str(exe_file),
                str(file_path),
                '-lm',  # Math library
                '-Wall'  # Warnings
            ]

            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            result['compile_time'] = round(time.time() - start, 2)

            if compile_result.returncode != 0:
                result['errors'].append(f'Compilation failed: {compile_result.stderr}')
                return result

            print(f"✅ Compiled in {result['compile_time']}s")

        except subprocess.TimeoutExpired:
            result['errors'].append('Compilation timeout (30s)')
            return result
        except Exception as e:
            result['errors'].append(f'Compilation error: {str(e)}')
            return result

        # Step 2: Run
        try:
            start = time.time()

            run_result = subprocess.run(
                [str(exe_file)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(results_dir),
                input=stdin_data,
                encoding='utf-8',
                errors='replace'
            )

            result['run_time'] = round(time.time() - start, 2)
            result['output'] = run_result.stdout
            result['success'] = True

            # Save log
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(run_result.stdout)
                if run_result.stderr:
                    f.write('\n--- STDERR ---\n')
                    f.write(run_result.stderr)

            result['log_file'] = str(log_file.relative_to(self.project_root)).replace('\\', '/')

            # Parse test results
            self._parse_test_output(result, run_result.stdout)

            print(f"✅ Simulation completed in {result['run_time']}s")

        except subprocess.TimeoutExpired:
            result['errors'].append('Simulation timeout (60s)')
        except Exception as e:
            result['errors'].append(f'Simulation error: {str(e)}')

        # Cleanup executable
        try:
            if exe_file.exists():
                exe_file.unlink()
        except:
            pass

        return result

    def _run_python(self, file_path: Path, stdin_data: str = None) -> Dict:
        """Run Python code"""
        import time

        # Setup paths
        results_dir = self.project_root / 'output' / 'mvl_results'
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = results_dir / f"{file_path.stem}_{timestamp}.log"

        result = {
            'success': False,
            'language': 'python',
            'file': file_path.name,
            'compile_time': 0,
            'run_time': 0,
            'output': '',
            'errors': [],
            'test_results': {
                'total': 0,
                'passed': 0,
                'failed': 0
            }
        }

        python_cmd = self.tools.get('python_cmd', 'python')

        try:
            start = time.time()

            run_result = subprocess.run(
                [python_cmd, str(file_path)],
                capture_output=True,
                text=True,
                timeout=60,
                input=stdin_data,
                encoding='utf-8',
                errors='replace'
            )

            result['run_time'] = round(time.time() - start, 2)
            result['output'] = run_result.stdout
            result['success'] = (run_result.returncode == 0)

            if run_result.stderr:
                result['errors'].append(run_result.stderr)

            # Save log
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(run_result.stdout)
                if run_result.stderr:
                    f.write('\n--- STDERR ---\n')
                    f.write(run_result.stderr)

            result['log_file'] = str(log_file.relative_to(self.project_root)).replace('\\', '/')

            # Parse test results
            self._parse_test_output(result, run_result.stdout)

            print(f"✅ Python simulation completed in {result['run_time']}s")

        except subprocess.TimeoutExpired:
            result['errors'].append('Simulation timeout (60s)')
        except Exception as e:
            result['errors'].append(f'Simulation error: {str(e)}')

        return result

    def _run_verilog(self, file_path: Path, stdin_data: str = None) -> Dict:
        """Compile and run Verilog code"""
        import time

        # Setup paths
        results_dir = self.project_root / 'output' / 'mvl_results'
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vvp_file = results_dir / f"{file_path.stem}_{timestamp}.vvp"
        log_file = results_dir / f"{file_path.stem}_{timestamp}.log"

        result = {
            'success': False,
            'language': 'verilog',
            'file': file_path.name,
            'compile_time': 0,
            'run_time': 0,
            'output': '',
            'errors': [],
            'test_results': {
                'total': 0,
                'passed': 0,
                'failed': 0
            }
        }

        # Step 1: Compile with iverilog
        try:
            start = time.time()

            compile_cmd = [
                'iverilog',
                '-g2012',
                '-o', str(vvp_file),
                str(file_path)
            ]

            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )

            result['compile_time'] = round(time.time() - start, 2)

            if compile_result.returncode != 0:
                result['errors'].append(f'Compilation failed: {compile_result.stderr}')
                return result

        except subprocess.TimeoutExpired:
            result['errors'].append('Compilation timeout (10s)')
            return result
        except Exception as e:
            result['errors'].append(f'Compilation error: {str(e)}')
            return result

        # Step 2: Run with vvp
        try:
            start = time.time()

            run_result = subprocess.run(
                ['vvp', str(vvp_file)],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(results_dir),
                input=stdin_data,
                encoding='utf-8',
                errors='replace'
            )

            result['run_time'] = round(time.time() - start, 2)
            result['output'] = run_result.stdout
            result['success'] = True

            # Save log
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(run_result.stdout)

            result['log_file'] = str(log_file.relative_to(self.project_root)).replace('\\', '/')

            # Parse test results
            self._parse_test_output(result, run_result.stdout)

        except subprocess.TimeoutExpired:
            result['errors'].append('Simulation timeout (15s) — testbench may be missing $finish')
            result['success'] = False
        except Exception as e:
            result['errors'].append(f'Simulation error: {str(e)}')

        # Cleanup
        try:
            if vvp_file.exists():
                vvp_file.unlink()
        except:
            pass

        return result

    @staticmethod
    def _find_vhdl_entity(file_path: Path) -> str:
        """Parse the VHDL file to find the top-level entity name.

        Prefers entities whose name contains 'tb' or 'test' (testbench).
        Falls back to the last entity declared in the file.
        Returns None if no entity is found.
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return None

        # Match "entity <name> is" (case-insensitive)
        entities = re.findall(
            r'(?i)\bentity\s+(\w+)\s+is\b', content
        )
        if not entities:
            return None

        # Prefer testbench entity (contains 'tb' or 'test')
        for ent in entities:
            if 'tb' in ent.lower() or 'test' in ent.lower():
                return ent

        # Fallback: last entity (often the testbench wrapping earlier entities)
        return entities[-1]

    @staticmethod
    def _count_vhdl_asserts(file_path: Path) -> int:
        """Count assert statements in a VHDL testbench file.

        Used to determine test count when GHDL produces no output
        (meaning all assertions passed).
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return 0

        # Match "assert <condition> report ..." lines (test assertions)
        # Exclude "assert false" which is used for stopping simulation
        asserts = re.findall(
            r'(?i)\bassert\s+(?!false\b)\S+.*\breport\b', content
        )
        return len(asserts)

    def _run_vhdl(self, file_path: Path, vector_file: str = None) -> Dict:
        """Compile and run VHDL code using GHDL"""
        import time

        results_dir = self.project_root / 'output' / 'mvl_results'
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        work_dir = results_dir / f"ghdl_work_{timestamp}"
        work_dir.mkdir(parents=True, exist_ok=True)
        log_file = results_dir / f"{file_path.stem}_{timestamp}.log"

        result = {
            'success': False,
            'language': 'vhdl',
            'file': file_path.name,
            'compile_time': 0,
            'run_time': 0,
            'output': '',
            'errors': [],
            'test_results': {
                'total': 0,
                'passed': 0,
                'failed': 0
            }
        }

        # Step 1: Analyze (syntax check + parse)
        try:
            start = time.time()
            analyze_cmd = [
                'ghdl', '-a',
                '--workdir=' + str(work_dir),
                str(file_path)
            ]
            analyze_result = subprocess.run(
                analyze_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )
            if analyze_result.returncode != 0:
                result['errors'].append(f'Compilation failed: {analyze_result.stderr}')
                return result
        except subprocess.TimeoutExpired:
            result['errors'].append('Compilation timeout (30s)')
            return result
        except Exception as e:
            result['errors'].append(f'Compilation error: {str(e)}')
            return result

        # Step 2: Elaborate
        # Parse actual entity name from VHDL source (file stem may not match)
        entity_name = self._find_vhdl_entity(file_path) or file_path.stem.replace('-', '_')
        try:
            elab_cmd = [
                'ghdl', '-e',
                '--workdir=' + str(work_dir),
                entity_name
            ]
            elab_result = subprocess.run(
                elab_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(work_dir),
                encoding='utf-8',
                errors='replace'
            )
            result['compile_time'] = round(time.time() - start, 2)

            if elab_result.returncode != 0:
                result['errors'].append(f'Elaboration failed: {elab_result.stderr}')
                return result
        except subprocess.TimeoutExpired:
            result['errors'].append('Elaboration timeout (30s)')
            return result
        except Exception as e:
            result['errors'].append(f'Elaboration error: {str(e)}')
            return result

        # Step 3: Run
        # If a vector file is provided (Strategy B), copy it into the work dir
        if vector_file:
            import shutil as _shutil2
            _shutil2.copy2(vector_file, str(work_dir / 'test_vectors.txt'))

        try:
            start = time.time()
            run_cmd = [
                'ghdl', '-r',
                '--workdir=' + str(work_dir),
                entity_name,
                '--stop-time=10ms'
            ]
            run_result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(work_dir),
                encoding='utf-8',
                errors='replace'
            )
            result['run_time'] = round(time.time() - start, 2)
            # GHDL outputs report messages to stderr
            result['output'] = run_result.stdout + run_result.stderr
            result['success'] = True

            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(result['output'])

            result['log_file'] = str(log_file.relative_to(self.project_root)).replace('\\', '/')
            self._parse_test_output(result, result['output'])

            # VHDL special case: assert ... severity error only prints on FAILURE.
            # If simulation succeeded with 0 tests detected and no assertion errors
            # in the output, count assert statements in the source as passed tests.
            if (result['test_results']['total'] == 0
                    and result['success']
                    and 'severity' not in result['output'].lower()):
                assert_count = self._count_vhdl_asserts(file_path)
                if assert_count > 0:
                    result['test_results']['total'] = assert_count
                    result['test_results']['passed'] = assert_count
                    result['test_results']['failed'] = 0

        except subprocess.TimeoutExpired:
            result['errors'].append('Simulation timeout (60s)')
        except Exception as e:
            result['errors'].append(f'Simulation error: {str(e)}')

        # Cleanup work dir
        try:
            import shutil as _shutil
            _shutil.rmtree(work_dir, ignore_errors=True)
        except:
            pass

        return result

    def _parse_test_output(self, result: Dict, output: str):
        """Parse test output for statistics.

        Handles many output formats that LLMs commonly produce in $display/printf/print:
          - "Test  1: ADD A=0 B=0 -> R=0 Z=1 N=0 C=0"
          - "ADD: 0 + 0 = 0"  /  "SUB: 5 - 3 = 2"
          - "OP=ADD A=0 B=0 Result=0"   (Verilog $display)
          - "op=0 a=0 b=0 result=0"     (numeric op codes)
          - "[  10] op=0 a=0 ..."        (timestamped Verilog)
          - "  0: A= 0 B= 0 OP= 0 => Result= 0"  (tabular)
          - "a=5, b=3, result=8"          (simple key=value)
          - "Time 10: a=5 b=3 opcode=0 result=8"
          - "PASS: ADD 5+3=8" / "FAIL: SUB 5-3!=1"
          - "Test ADD: a=0, b=0, expected=0, got=0"
        """
        lines = output.split('\n')

        _OP_NAMES = r'(?:ADD|SUB|MUL|NEG|INC|DEC|AND|OR|XOR|NOT|SHL|SHR|MOD|DIV)'
        _TEST_PATTERNS = [
            # Explicit test numbering: "Test 1:", "Test #1", "test_1", "Test case 1"
            re.compile(r'Test\s*(?:case\s*)?[#_]?\s*\d+', re.IGNORECASE),
            # Operation name at start: "ADD: ...", "SUB(...)", "ADD :"
            re.compile(rf'^{_OP_NAMES}\s*[:(]', re.IGNORECASE),
            # OP= or opcode=: "OP=ADD", "OP=0", "opcode=3"
            re.compile(r'(?:OP|opcode)\s*=\s*', re.IGNORECASE),
            # Timestamped Verilog: "[  10] ..." with test-like content
            re.compile(rf'^\[\s*\d+\]\s*.*(?:op|{_OP_NAMES}|result|a\s*=|b\s*=)', re.IGNORECASE),
            # Tabular: "  0: A= 0 B= 0"
            re.compile(r'^\s*\d+\s*:\s*[A-Za-z]\s*=', re.IGNORECASE),
            # Key=value with result: "a=5, b=3, result=8" or "a = 5 b = 3 result = 8"
            re.compile(r'(?:^|[\s,])a\s*=\s*\d+.*(?:result|out)', re.IGNORECASE),
            # "result=" or "Result:" with a number
            re.compile(r'result\s*[=:]\s*\d+', re.IGNORECASE),
            # PASS/FAIL markers (also counted separately below)
            re.compile(r'^\s*(?:PASS|FAIL)\s*[:\-]', re.IGNORECASE),
            # "Expected X, Got Y" pattern
            re.compile(r'expected\s*[=:]\s*\d+.*got\s*[=:]\s*\d+', re.IGNORECASE),
            # Time-prefixed: "Time 10ns: ..." or "@ 10:" with test content
            re.compile(rf'(?:Time|@)\s*\d+\s*(?:ns|ps|us)?\s*:?\s*.*(?:{_OP_NAMES}|result|a\s*=)', re.IGNORECASE),
            # Numeric op at start of line: "op=0 a=0 b=0"
            re.compile(r'^op\s*=\s*\d', re.IGNORECASE),
            # Operation name followed by operands: "ADD 5 + 3 = 8", "MUL(5, 3) = 15"
            re.compile(rf'{_OP_NAMES}\s*[\(:]?\s*\d+', re.IGNORECASE),
        ]

        # Lines to exclude (not real test output)
        _EXCLUDE_PATTERNS = [
            re.compile(r'^\s*(?:module|endmodule|wire|reg|input|output|assign|always|initial)\b'),
            re.compile(r'^\s*(?://|/\*|\*)', re.IGNORECASE),
            re.compile(r'^\s*(?:VCD|WARNING|ERROR|Loading|Compiling)', re.IGNORECASE),
            re.compile(r'TIMEOUT.*auto-terminated', re.IGNORECASE),
        ]

        test_count = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip non-test lines
            if any(ep.search(stripped) for ep in _EXCLUDE_PATTERNS):
                continue
            for pat in _TEST_PATTERNS:
                if pat.search(stripped):
                    test_count += 1
                    break

        # Fallback: if no specific patterns matched, count lines that look like
        # structured test output (contain multiple = signs with numeric values,
        # e.g. "alu_exec(0, 0, 0) = (0, Flags(z=1, n=0, c=0))")
        if test_count == 0:
            _FALLBACK_PATTERN = re.compile(
                r'(?:'
                r'=\s*\d+.*=\s*\d+'           # two or more "=number" on same line
                r'|'
                r'alu_exec\s*\('               # alu_exec(...) call output
                r'|'
                r'\bFlags?\s*\('               # Flags(...) or Flag(...) output
                r'|'
                r'\(\s*\d+\s*,\s*\d+\s*\)'    # tuple-like (N, M) output
                r'|'
                r'\d+.*(?:->|→|=>).*\d+'       # "N ... -> ... M" (input -> output)
                r')',
                re.IGNORECASE
            )
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if any(ep.search(stripped) for ep in _EXCLUDE_PATTERNS):
                    continue
                if _FALLBACK_PATTERN.search(stripped):
                    test_count += 1

        result['test_results']['total'] = test_count
        result['test_results']['passed'] = test_count
        result['test_results']['failed'] = 0

        # Look for explicit pass/fail markers
        passed = len(re.findall(r'PASS|✓|✅', output, re.IGNORECASE))
        failed = len(re.findall(r'FAIL|✗|❌', output, re.IGNORECASE))

        if passed > 0 or failed > 0:
            result['test_results']['passed'] = passed
            result['test_results']['failed'] = failed
            result['test_results']['total'] = passed + failed


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run MVL Simulations')
    parser.add_argument('file', help='Code file to simulate')
    parser.add_argument('--lang', help='Language (auto-detect if not specified)')

    args = parser.parse_args()

    runner = MVLSimulationRunner()

    print(f"\n🚀 Running simulation: {args.file}")

    result = runner.run_simulation(args.file, args.lang)

    if result['success']:
        print(f"\n✅ Simulation successful!")
        print(f"   Tests: {result['test_results']['total']}")
        print(f"\n--- Output ---")
        print(result['output'][:2000])
    else:
        print(f"\n❌ Simulation failed!")
        for error in result.get('errors', []):
            print(f"   Error: {error}")


if __name__ == '__main__':
    main()