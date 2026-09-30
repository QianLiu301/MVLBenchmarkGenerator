"""
MVL Simulation Runner
=====================
Run simulations for MVL ALU benchmarks using gcc/python/iverilog.
"""

import subprocess
import shutil
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class MVLSimulationRunner:
    """Run MVL code simulations"""

    # Wall-clock limits for the VHDL steps, in seconds. A well-formed run ends
    # long before any of them: the testbench stops its clock after the last
    # vector. They are ceilings for large files (GF tables make analysis slow)
    # and for a slower server, so they err on the generous side. The injected
    # run gets the most because an exhausted operand space is 393,216 vectors.
    VHDL_ANALYSE_TIMEOUT = 60
    VHDL_ELABORATE_TIMEOUT = 60
    VHDL_RUN_TIMEOUT = 120
    VHDL_RUN_TIMEOUT_INJECTED = 300

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.tools = self._check_tools()

        print(f"🔧 MVL Simulation Runner initialized")
        print(f"   Project root: {self.project_root}")
        # self.tools carries *_env entries holding the whole process environment,
        # which on the server means every API key, the secret key and the
        # database password. Printing the dict wrote all of them into the
        # application log, so only the availability flags are reported here.
        available = sorted(name for name, value in self.tools.items()
                           if value is True)
        print(f"   Tools available: {', '.join(available) if available else 'none'}")

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

        # Check C compilers with multiple strategies
        gcc_found = False

        # Strategy 1: shutil.which (standard, works on Linux/Mac and sometimes Windows)
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
                    print(f"   GCC found via shutil.which: {compiler}")
                    break
                except:
                    pass

        # Strategy 2 (Windows): use 'where gcc' to find the exact path
        if not gcc_found and os.name == 'nt':
            try:
                where_result = subprocess.run(
                    ['where', 'gcc'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=True
                )
                if where_result.returncode == 0 and where_result.stdout.strip():
                    gcc_path = where_result.stdout.strip().splitlines()[0].strip()
                    if os.path.isfile(gcc_path):
                        tools['gcc'] = True
                        tools['gcc_cmd'] = gcc_path
                        gcc_found = True
                        print(f"   GCC found via 'where': {gcc_path}")
            except:
                pass

        # Strategy 3 (Windows): try running gcc directly with shell=True
        if not gcc_found and os.name == 'nt':
            try:
                result = subprocess.run(
                    'gcc --version',
                    capture_output=True,
                    timeout=5,
                    shell=True
                )
                if result.returncode == 0 and result.stdout:
                    tools['gcc'] = True
                    tools['gcc_cmd'] = 'gcc'
                    gcc_found = True
                    print("   GCC found via shell=True subprocess")
            except:
                pass

        # Strategy 4 (Windows): scan all drives for common MSYS2/MinGW paths
        if not gcc_found and os.name == 'nt':
            import string
            gcc_relative_paths = [
                os.path.join('msys64', 'ucrt64', 'bin', 'gcc.exe'),
                os.path.join('msys64', 'mingw64', 'bin', 'gcc.exe'),
                os.path.join('msys64', 'mingw32', 'bin', 'gcc.exe'),
                os.path.join('msys2', 'ucrt64', 'bin', 'gcc.exe'),
                os.path.join('msys2', 'mingw64', 'bin', 'gcc.exe'),
                os.path.join('msys2', 'mingw32', 'bin', 'gcc.exe'),
                os.path.join('MinGW', 'bin', 'gcc.exe'),
                os.path.join('mingw64', 'bin', 'gcc.exe'),
            ]
            # Collect search roots: drive roots + one level of subdirectories
            search_roots = []
            for letter in string.ascii_uppercase:
                drive_path = f'{letter}:\\'
                if os.path.exists(drive_path):
                    search_roots.append(drive_path)
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
                            print(f"   GCC found via path scan: {gcc_path}")
                            break
                        except:
                            pass

        if not gcc_found:
            print("   GCC: not found (tried shutil.which, where, shell, path scan)")

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

        # Check Verilog tools (iverilog, vvp) and GHDL
        for tool_name, version_flag, tool_key in [
            ('iverilog', '-V', 'iverilog'),
            ('vvp', '-V', 'vvp'),
            ('ghdl', '--version', 'ghdl'),
        ]:
            # Collect ALL candidate paths (order: shutil.which, where, path scan)
            candidates = []

            # 1) shutil.which
            which_path = shutil.which(tool_name)
            if which_path:
                candidates.append(('shutil.which', which_path))

            # 2) Windows 'where' — returns ALL installations, try every one
            if os.name == 'nt':
                try:
                    where_result = subprocess.run(
                        f'where {tool_name}',
                        capture_output=True, text=True, timeout=5, shell=True
                    )
                    if where_result.returncode == 0:
                        for line in where_result.stdout.strip().splitlines():
                            p = line.strip()
                            if p and os.path.isfile(p) and p not in [c[1] for c in candidates]:
                                candidates.append(('where', p))
                except Exception:
                    pass

            # 3) Windows path scan (Program Files + all drives)
            if os.name == 'nt':
                import string
                scan_patterns = [
                    os.path.join('iverilog', 'bin'),
                    os.path.join('Icarus Verilog', 'bin'),
                    os.path.join('IcarusVerilog', 'bin'),
                    os.path.join('oss-cad-suite', 'bin'),
                    os.path.join('msys64', 'ucrt64', 'bin'),
                    os.path.join('msys64', 'mingw64', 'bin'),
                    os.path.join('msys2', 'ucrt64', 'bin'),
                    os.path.join('msys2', 'mingw64', 'bin'),
                    os.path.join('GHDL', 'bin'),
                    os.path.join('ghdl', 'bin'),
                ]
                # Check Program Files first (fast)
                for pf in [os.environ.get('ProgramFiles', ''),
                           os.environ.get('ProgramFiles(x86)', '')]:
                    if pf:
                        for pat in scan_patterns:
                            candidate = os.path.join(pf, pat, f'{tool_name}.exe')
                            if os.path.isfile(candidate) and candidate not in [c[1] for c in candidates]:
                                candidates.append(('ProgramFiles', candidate))

                # Scan drive roots + one level of subdirectories
                search_roots = []
                for letter in string.ascii_uppercase:
                    drive = f'{letter}:\\'
                    if os.path.exists(drive):
                        search_roots.append(drive)
                        try:
                            for entry in os.scandir(drive):
                                if entry.is_dir():
                                    search_roots.append(entry.path)
                        except (PermissionError, OSError):
                            pass
                for root in search_roots:
                    for pat in scan_patterns:
                        candidate = os.path.join(root, pat, f'{tool_name}.exe')
                        if os.path.isfile(candidate) and candidate not in [c[1] for c in candidates]:
                            candidates.append(('scan', candidate))

            # Try each candidate with proper environment setup
            for source, cpath in candidates:
                if tools.get(tool_key):
                    break
                env = self._build_tool_env(cpath)
                print(f"   Trying {tool_name}: {cpath} (via {source})")

                # For iverilog: -V only tests the front-end, but the backend (ivl.exe)
                # may have broken DLLs. We must test actual compilation.
                for use_shell in ([False, True] if os.name == 'nt' else [False]):
                    try:
                        # Quick version check first
                        result = subprocess.run(
                            [cpath, version_flag] if not use_shell else f'"{cpath}" {version_flag}',
                            capture_output=True, timeout=5,
                            env=env, shell=use_shell
                        )
                        if result.returncode != 0:
                            continue

                        # For iverilog: also test actual compilation (backend ivl.exe)
                        if tool_name == 'iverilog':
                            if not self._verify_iverilog_compile(cpath, env, use_shell):
                                print(f"   ✗ {tool_name} -V OK but compile test FAILED (backend broken)")
                                continue

                        tools[tool_key] = True
                        tools[f'{tool_key}_cmd'] = cpath
                        tools[f'{tool_key}_env'] = env
                        if use_shell:
                            tools[f'{tool_key}_needs_shell'] = True
                        shell_note = " (shell)" if use_shell else ""
                        print(f"   ✓ {tool_name} fully verified{shell_note}: {cpath}")
                        break
                    except Exception:
                        pass
                else:
                    print(f"   ✗ {tool_name} at {cpath} failed")

            # Last resort: shell=True with just the tool name
            if not tools.get(tool_key) and os.name == 'nt':
                try:
                    result = subprocess.run(
                        f'{tool_name} {version_flag}',
                        capture_output=True, timeout=5, shell=True
                    )
                    if result.returncode == 0:
                        tools[tool_key] = True
                        tools[f'{tool_key}_cmd'] = tool_name
                        tools[f'{tool_key}_needs_shell'] = True
                        print(f"   ✓ {tool_name} found via shell (name only)")
                except Exception:
                    pass

            if not tools.get(tool_key):
                if candidates:
                    paths_str = ', '.join(c[1] for c in candidates)
                    tools[f'{tool_key}_candidates'] = [c[1] for c in candidates]
                    print(f"   ✗ {tool_name}: found at [{paths_str}] but ALL failed to run (DLL missing)")
                    print(f"     Fix: open cmd.exe and run '{candidates[0][1]} {version_flag}'")
                    print(f"     If it fails, reinstall Icarus Verilog or add its DLL directory to PATH")
                else:
                    print(f"   ✗ {tool_name}: not found anywhere")

        return tools

    @staticmethod
    def _verify_iverilog_compile(iverilog_path: str, env: dict, use_shell: bool = False) -> bool:
        """Test that iverilog can actually compile a minimal file.

        iverilog -V only tests the front-end (iverilog.exe).
        The backend compiler (ivl.exe) may have broken DLLs
        (e.g., oss-cad-suite with mismatched C++ ABI).
        This test catches that by doing actual compilation.
        """
        import tempfile
        test_v = None
        test_out = None
        try:
            # Create minimal Verilog file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v',
                                              delete=False, dir=tempfile.gettempdir()) as f:
                f.write('module test; initial $display("ok"); endmodule\n')
                test_v = f.name
            test_out = test_v.replace('.v', '.vvp')

            cmd = [iverilog_path, '-o', test_out, test_v]
            result = subprocess.run(
                cmd if not use_shell else f'"{iverilog_path}" -o "{test_out}" "{test_v}"',
                capture_output=True, timeout=10,
                env=env, shell=use_shell
            )
            return result.returncode == 0
        except Exception:
            return False
        finally:
            # Cleanup temp files
            for f in [test_v, test_out]:
                if f:
                    try:
                        os.unlink(f)
                    except Exception:
                        pass

    @staticmethod
    def _build_tool_env(tool_path: str) -> dict:
        """Build environment dict with proper PATH for a tool's DLL dependencies.

        For oss-cad-suite: adds both bin/ and lib/ directories.
        For standalone Icarus Verilog: adds bin/ and lib/ivl/.
        For MSYS2: adds the MSYS2 bin directory.
        """
        env = os.environ.copy()
        if not os.path.exists(tool_path):
            return env

        tool_dir = os.path.dirname(os.path.abspath(tool_path))
        parent_dir = os.path.dirname(tool_dir)
        extra_paths = [tool_dir]

        # Detect oss-cad-suite: parent directory is named 'oss-cad-suite'
        # or grandparent contains oss-cad-suite
        oss_root = None
        for ancestor in [parent_dir, os.path.dirname(parent_dir)]:
            if 'oss-cad-suite' in os.path.basename(ancestor).lower():
                oss_root = ancestor
                break

        if oss_root:
            # oss-cad-suite needs both bin and lib in PATH
            oss_bin = os.path.join(oss_root, 'bin')
            oss_lib = os.path.join(oss_root, 'lib')
            if os.path.isdir(oss_bin):
                extra_paths.append(oss_bin)
            if os.path.isdir(oss_lib):
                extra_paths.append(oss_lib)
            # Also check for environment script to discover more paths
            env_bat = os.path.join(oss_root, 'environment.bat')
            if os.path.isfile(env_bat):
                try:
                    content = open(env_bat, 'r', encoding='utf-8', errors='replace').read()
                    import re as _re
                    for m in _re.finditer(r'set\s+PATH=([^;%\n]+)', content, _re.IGNORECASE):
                        p = m.group(1).strip()
                        # Resolve relative paths against oss_root
                        if not os.path.isabs(p):
                            p = os.path.join(oss_root, p)
                        if os.path.isdir(p) and p not in extra_paths:
                            extra_paths.append(p)
                except Exception:
                    pass

        # Standalone Icarus Verilog: check for lib/ivl
        lib_ivl = os.path.join(parent_dir, 'lib', 'ivl')
        if os.path.isdir(lib_ivl):
            extra_paths.append(lib_ivl)

        # MSYS2: if tool is in ucrt64/bin, add that
        if 'msys' in tool_dir.lower() or 'ucrt64' in tool_dir.lower() or 'mingw' in tool_dir.lower():
            extra_paths.append(tool_dir)

        # Prepend all extra paths to PATH
        current_path = env.get('PATH', '')
        new_paths = [p for p in extra_paths if p not in current_path]
        if new_paths:
            env['PATH'] = os.pathsep.join(new_paths) + os.pathsep + current_path

        return env

    def refresh_tools(self):
        """Re-detect available tools (useful after installing new tools)"""
        self.tools = self._detect_tools()
        MVLSimulationRunner._systemc_probe = None   # probe SystemC again as well
        return self.get_tools_status()

    def get_tools_status(self) -> Dict:
        """Get tools status for API"""
        status = {
            'c_available': self.tools.get('gcc') or self.tools.get('clang'),
            'python_available': self.tools.get('python'),
            'verilog_available': bool(self.tools.get('iverilog') and self.tools.get('vvp')),
            'vhdl_available': bool(self.tools.get('ghdl')),
            'systemc_available': bool(self._systemc_status()['ok']),
            'tools': {k: v for k, v in self.tools.items()
                      if not isinstance(v, dict)}  # exclude env dicts (too large)
        }
        # Add diagnostic info for unavailable tools
        if not status['systemc_available']:
            status['systemc_diagnostic'] = self._systemc_status()['reason']
        if not status['verilog_available']:
            candidates = self.tools.get('iverilog_candidates', [])
            if candidates:
                status['verilog_diagnostic'] = (
                    f'iverilog found at {candidates[0]} but DLL dependencies missing. '
                    f'Try reinstalling from https://bleyer.org/icarus/ '
                    f'(check "Add to PATH" during install).'
                )
        return status

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
        elif lang == 'systemc':
            return self._systemc_status()['ok']
        return False

    def _env_with_tool_dir(self, command: str) -> Dict[str, str]:
        """Environment with the tool's own directory first on PATH.

        A MSYS2 gcc loads its runtime DLLs from the directory it sits in, and
        without that directory on PATH it exits 1 and prints nothing at all, so
        a perfectly good file looks like a compilation failure. The detected
        command is usually a bare name, which os.path.exists cannot resolve —
        hence shutil.which before taking the directory.
        """
        env = os.environ.copy()
        resolved = shutil.which(command) if command else None
        if resolved:
            # Prepended unconditionally, even when the directory is already on
            # PATH: what matters is that it comes first. Here D:\soft\Git\
            # mingw64\bin precedes the MSYS2 directory and offers its own
            # incompatible runtime DLLs, so gcc picked those up and died
            # silently while sitting on a PATH that looked correct.
            tool_dir = os.path.dirname(os.path.abspath(resolved))
            if tool_dir:
                env['PATH'] = tool_dir + os.pathsep + env.get('PATH', '')
        return env

    def check_syntax(self, file_path: str, language: str) -> Dict:
        """Compile or analyse a file without running it.

        Returns {'ok': bool, 'errors': str, 'checked': bool}. 'checked' is False
        when the toolchain for that language is not installed, so a caller can
        tell "nothing was wrong" apart from "nothing was examined".

        This reports only whether the file is well formed. It says nothing about
        whether it computes the right function, which matters because its output
        is fed back to the model: a compiler cannot leak the expected results,
        and the reference model is never consulted here.
        """
        path = Path(file_path)
        lang = (language or '').lower()
        out = {'ok': False, 'errors': '', 'checked': True}

        if not self.can_run(lang):
            return {'ok': True, 'errors': '', 'checked': False}

        try:
            if lang == 'c':
                cc = self.tools.get('gcc_cmd') or self.tools.get('clang_cmd') or 'gcc'
                proc = subprocess.run([cc, '-fsyntax-only', str(path)],
                                      capture_output=True, timeout=30,
                                      env=self._env_with_tool_dir(cc))
            elif lang == 'python':
                py = self.tools.get('python_cmd') or sys.executable
                cmd = [py, '-m', 'py_compile', str(path)]
                proc = subprocess.run(cmd, capture_output=True, timeout=30)
            elif lang == 'verilog':
                work_dir = path.parent / ('_syntax_' + path.stem)
                work_dir.mkdir(parents=True, exist_ok=True)
                cmd = [self.tools.get('iverilog_cmd', 'iverilog'),
                       '-o', str(work_dir / 'a.out'), str(path)]
                proc = subprocess.run(cmd, capture_output=True, timeout=30,
                                      env=self.tools.get('iverilog_env', os.environ.copy()))
            elif lang == 'systemc':
                gxx = self._systemc_status()['cmd'] or 'g++'
                proc = subprocess.run([gxx, *self._systemc_flags(), '-fsyntax-only', str(path)],
                                      capture_output=True, timeout=self.SYSTEMC_COMPILE_TIMEOUT,
                                      env=self._systemc_env(self._env_with_tool_dir(gxx)))
            else:  # vhdl
                work_dir = path.parent / ('_syntax_' + path.stem)
                work_dir.mkdir(parents=True, exist_ok=True)
                cmd = [self.tools.get('ghdl_cmd', 'ghdl'), '-a', '--std=08',
                       '--workdir=' + str(work_dir), str(path)]
                proc = subprocess.run(cmd, capture_output=True, timeout=30,
                                      shell=self.tools.get('ghdl_needs_shell', False),
                                      env=self.tools.get('ghdl_env', os.environ.copy()))

            out['ok'] = proc.returncode == 0
            if not out['ok']:
                out['errors'] = (self._decode_output(proc.stderr)
                                 or self._decode_output(proc.stdout)
                                 or f'returncode={proc.returncode}')
        except subprocess.TimeoutExpired:
            out['errors'] = f'{lang} syntax check timed out after 30s'
        except Exception as e:
            out['errors'] = f'{lang} syntax check could not run: {e}'

        return out

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
                '.vhd': 'vhdl',
                '.cpp': 'systemc',
            }.get(ext, 'unknown')

        if not self.can_run(language):
            # Build diagnostic message
            if language == 'verilog':
                candidates = self.tools.get('iverilog_candidates', [])
                if candidates:
                    hint = (
                        f'iverilog found at {candidates[0]} but cannot run (DLL error). '
                        f'Fix: open cmd.exe and run: "{candidates[0]}" -V\n'
                        f'If it fails, try reinstalling Icarus Verilog from '
                        f'https://bleyer.org/icarus/ (choose "Add to PATH" during install).\n'
                        f'Or add the iverilog bin directory to your system PATH.'
                    )
                else:
                    hint = (
                        'Install Icarus Verilog from https://bleyer.org/icarus/\n'
                        'During installation, check "Add to PATH".'
                    )
            else:
                tool_hints = {
                    'c': 'Install gcc or clang',
                    'python': 'Install python3',
                    'vhdl': 'Install ghdl (GHDL VHDL simulator)',
                    'systemc': ('Install a C++ compiler and the SystemC library '
                                '(Debian: libsystemc-dev; MSYS2: mingw-w64-ucrt-x86_64-systemc). '
                                'Probe said: ' + (self._systemc_status()['reason'] or 'unavailable')),
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
        elif language == 'systemc':
            return self._run_systemc(file_path, stdin_data=stdin_data)
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

    # ------------------------------------------------------------------
    # SystemC: a C++ class library, so it needs g++ and the library itself.
    # Whether both are usable is found out by building and running a small
    # model (_SYSTEMC_PROBE). That takes a few seconds and runners are created
    # often, so it happens on first use and the answer is shared by all.
    # ------------------------------------------------------------------
    _systemc_probe: Optional[Dict] = None
    SYSTEMC_COMPILE_TIMEOUT = 120      # systemc.h is heavy; a slow server needs the room
    # SystemC refuses to link code compiled under another C++ standard than the
    # library was: the library exports sc_api_version_<ver>_cxx<standard>, and
    # every translation unit references the one matching its own -std. The
    # MSYS2 package is built as C++20, Debian's as C++17, so the probe tries
    # each and every later compile uses the one that worked.
    SYSTEMC_STD_CANDIDATES = ['-std=c++17', '-std=c++20', '-std=c++14']
    SYSTEMC_LIBS = ['-lsystemc', '-lpthread']
    # The probe has to use what real models use. A program that only includes
    # systemc.h linked fine against the MSYS2 archive, while any model using
    # sc_signal<bool> failed with "multiple definition": that libsystemc.a holds
    # both static objects and DLL import stubs, and both define the same
    # template instances. The probe therefore instantiates a module with the
    # port types of an entry, simulates it and checks the result.
    _SYSTEMC_PROBE = """#include <systemc.h>
SC_MODULE(probe_m) {
    sc_in<bool> clk; sc_in<sc_uint<13>> a; sc_out<sc_uint<13>> r; sc_out<bool> z;
    void f() { r.write(a.read()); z.write(a.read() == 0); }
    SC_CTOR(probe_m) { SC_METHOD(f); sensitive << clk.pos(); }
};
int sc_main(int, char*[]) {
    sc_clock clk("clk", 10, SC_NS);
    sc_signal<sc_uint<13>> a, r; sc_signal<bool> z;
    probe_m m("m"); m.clk(clk); m.a(a); m.r(r); m.z(z);
    a.write(5); sc_start(20, SC_NS);
    return (r.read() == 5 && !z.read()) ? 0 : 1;
}
"""

    # Linking the MSYS2 DLL directly avoids its mixed archive, but that DLL
    # exports neither main() nor sc_elab_and_sim(), which live only in the
    # archive. This entry supplies main(): it calls sc_main as the library's own
    # main does and, as sc_elab_and_sim does, turns an escaped SystemC error into
    # a message on stderr rather than a silent abort. It is compiled once per
    # C++ standard and linked in; the model's code is not touched.
    _SYSTEMC_ENTRY = """#include <systemc.h>
#include <cstdio>
#include <exception>
int main(int argc, char* argv[]) {
    try {
        return sc_main(argc, argv);
    } catch (const sc_core::sc_report& e) {
        std::fprintf(stderr, "SystemC error: %s\\n", e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "error: unknown exception\\n");
    }
    return 1;
}
"""

    def _systemc_flags(self) -> list:
        return [self._systemc_status().get('std_flag') or self.SYSTEMC_STD_CANDIDATES[0]]

    def _systemc_libs(self) -> list:
        """What to put after the source file: the entry object if this
        installation needs one, then the library."""
        status = self._systemc_status()
        libs = status.get('libs') or list(self.SYSTEMC_LIBS)
        if status.get('needs_entry'):
            return [self._systemc_entry_object(status), *libs]
        return libs

    def _systemc_entry_object(self, status: Dict, out_dir: Path = None) -> str:
        """Compile _SYSTEMC_ENTRY once per C++ standard and return the object."""
        out_dir = out_dir or (self.project_root / 'output' / 'mvl_results')
        out_dir.mkdir(parents=True, exist_ok=True)
        std = status['std_flag']
        tag = std.replace('-std=', '').replace('+', 'x')
        src = out_dir / 'sc_entry.cpp'
        obj = out_dir / f'sc_entry_{tag}.o'
        if not src.exists() or src.read_text(encoding='utf-8') != self._SYSTEMC_ENTRY:
            src.write_text(self._SYSTEMC_ENTRY, encoding='utf-8')
            if obj.exists():
                obj.unlink()
        if not obj.exists():
            proc = subprocess.run(
                [status['cmd'], std, '-c', str(src), '-o', str(obj)],
                capture_output=True, timeout=self.SYSTEMC_COMPILE_TIMEOUT,
                env=self._systemc_env(self._env_with_tool_dir(status['cmd'])))
            if proc.returncode != 0:
                raise RuntimeError('could not build the SystemC entry point: '
                                   + (self._decode_output(proc.stderr) or '')[:300])
        return str(obj)

    def _systemc_link_candidates(self, gxx: str) -> list:
        """(link arguments, needs our entry) to try, in order: -lsystemc, then on
        Windows the DLL itself, which MinGW's ld links against directly and
        which leaves the mixed MSYS2 archive out entirely."""
        candidates = [(list(self.SYSTEMC_LIBS), False)]
        if os.name == 'nt':
            import glob
            tool_dir = os.path.dirname(os.path.abspath(gxx))
            for dll in sorted(glob.glob(os.path.join(tool_dir, 'libsystemc*.dll'))):
                candidates.append(([dll, '-lpthread'], True))
        return candidates

    @staticmethod
    def _systemc_env(base: Dict[str, str]) -> Dict[str, str]:
        env = dict(base)
        env['SC_COPYRIGHT_MESSAGE'] = 'DISABLE'   # keep the banner out of the parsed output
        return env

    def _systemc_status(self) -> Dict:
        if MVLSimulationRunner._systemc_probe is None:
            MVLSimulationRunner._systemc_probe = self._probe_systemc()
        return MVLSimulationRunner._systemc_probe

    def _probe_systemc(self) -> Dict:
        import tempfile
        gxx = shutil.which('g++') or shutil.which('clang++')
        if not gxx:
            return {'ok': False, 'cmd': None, 'reason': 'no C++ compiler (g++ or clang++) found'}
        work = Path(tempfile.mkdtemp(prefix='mvl_sc_probe_'))
        try:
            src = work / 'probe.cpp'
            src.write_text(self._SYSTEMC_PROBE, encoding='utf-8')
            exe = work / ('probe.exe' if os.name == 'nt' else 'probe')
            env = self._systemc_env(self._env_with_tool_dir(gxx))
            # every attempt's outcome is kept: the last one alone is misleading
            # (C++14 always fails on SystemC 3 headers and would hide the cause)
            attempts = []
            for std in self.SYSTEMC_STD_CANDIDATES:
                for libs, needs_entry in self._systemc_link_candidates(gxx):
                    label = f'{std} {"+entry " if needs_entry else ""}{" ".join(libs)}'
                    try:
                        extra = ([self._systemc_entry_object({'cmd': gxx, 'std_flag': std}, work)]
                                 if needs_entry else [])
                    except RuntimeError as e:
                        attempts.append(f'{label}: {str(e)[:120]}')
                        continue
                    build = subprocess.run(
                        [gxx, std, str(src), '-o', str(exe), *extra, *libs],
                        capture_output=True, timeout=self.SYSTEMC_COMPILE_TIMEOUT, env=env)
                    if build.returncode != 0:
                        err = self._decode_output(build.stderr) or ''
                        key = [l for l in err.splitlines()
                               if 'error' in l.lower() or 'multiple definition' in l
                               or 'undefined reference' in l]
                        attempts.append(f'{label}: {(key or err.splitlines() or ["build failed"])[0][-160:]}')
                        continue
                    run = subprocess.run([str(exe)], capture_output=True, timeout=60, env=env)
                    if run.returncode == 0:
                        return {'ok': True, 'cmd': gxx, 'std_flag': std, 'libs': libs,
                                'needs_entry': needs_entry, 'reason': ''}
                    attempts.append(f'{label}: probe model ran but returned {run.returncode}')
            return {'ok': False, 'cmd': gxx, 'reason': ' | '.join(attempts)[:1500]}
        except Exception as e:
            return {'ok': False, 'cmd': gxx, 'reason': str(e)[:200]}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _run_c(self, file_path: Path, stdin_data: str = None) -> Dict:
        """Compile and run C code"""
        compiler = self.tools.get('gcc_cmd', 'gcc')
        return self._compile_and_run(
            file_path, stdin_data, language='c', label='C', compiler=compiler,
            compile_args=['-lm', '-Wall'], env=self._env_with_tool_dir(compiler),
            compile_timeout=30)

    def _run_systemc(self, file_path: Path, stdin_data: str = None) -> Dict:
        """Compile a SystemC model with its sc_main testbench, and run it."""
        compiler = self._systemc_status()['cmd'] or 'g++'
        return self._compile_and_run(
            file_path, stdin_data, language='systemc', label='SystemC', compiler=compiler,
            compile_args=[*self._systemc_flags(), *self._systemc_libs()],
            env=self._systemc_env(self._env_with_tool_dir(compiler)),
            compile_timeout=self.SYSTEMC_COMPILE_TIMEOUT)

    def _compile_and_run(self, file_path: Path, stdin_data: Optional[str], *, language: str,
                         label: str, compiler: str, compile_args: list, env: Dict[str, str],
                         compile_timeout: int) -> Dict:
        """Compile a native program, run it, and parse the test lines it prints.

        Shared by C and SystemC, which differ only in the compiler, its
        arguments, the environment and how long the compiler may take.
        """
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
            'language': language,
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

        c_env = env

        # Step 1: Compile. Libraries go after the source file: GNU ld resolves
        # them left to right, so "-lsystemc file.cpp" would leave sc_main's
        # references unresolved.
        try:
            start = time.time()

            compile_cmd = [compiler, '-o', str(exe_file), str(file_path), *compile_args]

            print(f"   {label} compile cmd: {' '.join(compile_cmd)}")

            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                timeout=compile_timeout,
                env=c_env
            )

            result['compile_time'] = round(time.time() - start, 2)

            if compile_result.returncode != 0:
                stderr_text = self._decode_output(compile_result.stderr)
                stdout_text = self._decode_output(compile_result.stdout)
                error_msg = stderr_text or stdout_text or f'returncode={compile_result.returncode}'
                print(f"   {label} compile failed: {error_msg}")
                result['errors'].append(f'Compilation failed: {error_msg}')
                return result

            print(f"   {label} compiled OK in {result['compile_time']}s")

        except subprocess.TimeoutExpired:
            result['errors'].append(f'Compilation timeout ({compile_timeout}s)')
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
                errors='replace',
                env=c_env
            )

            result['run_time'] = round(time.time() - start, 2)
            result['output'] = run_result.stdout
            result['success'] = run_result.returncode == 0
            if not result['success']:
                result['errors'].append(
                    f"Simulation aborted (exit code {run_result.returncode}): "
                    + (run_result.stderr or '').strip().splitlines()[-1:][0] if (run_result.stderr or '').strip() else
                    f"Simulation aborted (exit code {run_result.returncode})")

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

        # Use the verified environment from detection
        use_shell = self.tools.get('iverilog_needs_shell', False)
        run_env = self.tools.get('iverilog_env', os.environ.copy())

        # Step 1: Compile with iverilog
        try:
            start = time.time()

            iverilog_cmd = self.tools.get('iverilog_cmd', 'iverilog')
            compile_cmd = [
                iverilog_cmd,
                '-g2012',
                '-o', str(vvp_file),
                str(file_path)
            ]

            print(f"   Verilog compile cmd: {' '.join(compile_cmd)}")

            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                timeout=10,
                shell=use_shell,
                env=run_env
            )

            result['compile_time'] = round(time.time() - start, 2)

            if compile_result.returncode != 0:
                stderr_text = self._decode_output(compile_result.stderr)
                stdout_text = self._decode_output(compile_result.stdout)
                error_msg = stderr_text or stdout_text or f'returncode={compile_result.returncode}'
                print(f"   Verilog compile failed: {error_msg}")
                result['errors'].append(f'Compilation failed: {error_msg}')
                return result

            print(f"   Verilog compiled OK in {result['compile_time']}s")

        except subprocess.TimeoutExpired:
            result['errors'].append('Compilation timeout (10s)')
            return result
        except Exception as e:
            result['errors'].append(f'Compilation error: {str(e)}')
            return result

        # Step 2: Run with vvp
        try:
            start = time.time()

            vvp_cmd = self.tools.get('vvp_cmd', 'vvp')
            vvp_shell = self.tools.get('vvp_needs_shell', use_shell)
            vvp_env = self.tools.get('vvp_env', run_env)

            run_result = subprocess.run(
                [vvp_cmd, str(vvp_file)],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(results_dir),
                input=stdin_data,
                encoding='utf-8',
                errors='replace',
                shell=vvp_shell,
                env=vvp_env
            )

            result['run_time'] = round(time.time() - start, 2)
            result['output'] = run_result.stdout
            result['success'] = run_result.returncode == 0
            if not result['success']:
                result['errors'].append(
                    f"Simulation aborted (exit code {run_result.returncode}): "
                    + (run_result.stderr or '').strip().splitlines()[-1:][0] if (run_result.stderr or '').strip() else
                    f"Simulation aborted (exit code {run_result.returncode})")

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

        # Use the verified environment from detection
        ghdl_cmd = self.tools.get('ghdl_cmd', 'ghdl')
        use_shell = self.tools.get('ghdl_needs_shell', False)
        ghdl_env = self.tools.get('ghdl_env', os.environ.copy())

        # Step 1: Analyze (syntax check + parse)
        try:
            start = time.time()
            analyze_cmd = [
                ghdl_cmd, '-a',
                '--std=08',
                '--workdir=' + str(work_dir),
                str(file_path)
            ]
            print(f"   VHDL analyze cmd: {' '.join(analyze_cmd)}")
            analyze_result = subprocess.run(
                analyze_cmd,
                capture_output=True,
                timeout=self.VHDL_ANALYSE_TIMEOUT,
                shell=use_shell,
                env=ghdl_env
            )
            if analyze_result.returncode != 0:
                stderr_text = self._decode_output(analyze_result.stderr)
                stdout_text = self._decode_output(analyze_result.stdout)
                error_msg = stderr_text or stdout_text or f'returncode={analyze_result.returncode}'
                print(f"   VHDL analyze failed: {error_msg}")
                result['errors'].append(f'Compilation failed: {error_msg}')
                return result
            print(f"   VHDL analyze OK")
        except subprocess.TimeoutExpired:
            result['errors'].append(f'Analysis timeout ({self.VHDL_ANALYSE_TIMEOUT}s)')
            return result
        except Exception as e:
            result['errors'].append(f'Compilation error: {str(e)}')
            return result

        # Step 2: Elaborate
        # Parse actual entity name from VHDL source (file stem may not match)
        entity_name = self._find_vhdl_entity(file_path) or file_path.stem.replace('-', '_')
        try:
            elab_cmd = [
                ghdl_cmd, '-e',
                '--std=08',
                '--workdir=' + str(work_dir),
                entity_name
            ]
            print(f"   VHDL elaborate cmd: {' '.join(elab_cmd)} (entity={entity_name})")
            elab_result = subprocess.run(
                elab_cmd,
                capture_output=True,
                timeout=self.VHDL_ELABORATE_TIMEOUT,
                cwd=str(work_dir),
                shell=use_shell,
                env=ghdl_env
            )
            result['compile_time'] = round(time.time() - start, 2)

            if elab_result.returncode != 0:
                stderr_text = self._decode_output(elab_result.stderr)
                stdout_text = self._decode_output(elab_result.stdout)
                error_msg = stderr_text or stdout_text or f'returncode={elab_result.returncode}'
                print(f"   VHDL elaborate failed: {error_msg}")
                result['errors'].append(f'Elaboration failed: {error_msg}')
                return result
            print(f"   VHDL elaborate OK in {result['compile_time']}s")
        except subprocess.TimeoutExpired:
            result['errors'].append(f'Elaboration timeout ({self.VHDL_ELABORATE_TIMEOUT}s)')
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
            # Both testbenches that reach this point stop their own clock after
            # the last vector, so a well-formed run ends by itself and the stop
            # time is only a ceiling for files that still carry a free-running
            # clock (every VHDL entry generated before the template was fixed).
            #
            # It is set from the work requested: 20 ns per vector, doubled for
            # margin, never below 1 ms. The file's own testbench applies about
            # twenty vectors (under 1 us); the injected one can apply 393,216
            # when the operand space is exhausted (7.9 ms), which a flat 1 ms
            # would have cut off silently.
            #
            # It used to scale with the data width instead, which is backwards:
            # width makes each cycle slower to evaluate but does not add cycles,
            # so the widest designs got the most cycles to spin through after
            # their last vector -- "Simulation timeout (90s, stop-time=50ms)".
            n_vectors = 0
            if vector_file:
                try:
                    with open(vector_file, 'r', encoding='utf-8', errors='replace') as fh:
                        n_vectors = int(fh.readline().split()[0])
                except (OSError, ValueError, IndexError):
                    n_vectors = 400_000          # the exhaustive maximum
            stop_ns = max((n_vectors * 20 + 100) * 2, 1_000_000)
            stop_time = f'{stop_ns}ns'
            proc_timeout = (self.VHDL_RUN_TIMEOUT_INJECTED if vector_file
                            else self.VHDL_RUN_TIMEOUT)

            run_cmd = [
                ghdl_cmd, '-r',
                '--std=08',
                '--workdir=' + str(work_dir),
                entity_name,
                f'--stop-time={stop_time}'
            ]
            run_result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=proc_timeout,
                cwd=str(work_dir),
                encoding='utf-8',
                errors='replace',
                shell=use_shell,
                env=ghdl_env
            )
            result['run_time'] = round(time.time() - start, 2)
            # GHDL outputs report messages to stderr
            result['output'] = run_result.stdout + run_result.stderr
            # A non-zero exit or a "ghdl:error:" line (bound check failure, overflow,
            # "simulation failed") means the run aborted; lines printed before the
            # crash must not be graded as a passing run.
            crashed = run_result.returncode != 0 or 'ghdl:error:' in result['output']
            result['success'] = not crashed
            if crashed:
                tail = [l for l in result['output'].splitlines() if 'ghdl:error' in l][:3]
                result['errors'].append(
                    f"Simulation aborted (exit code {run_result.returncode}): " + ' | '.join(tail))

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
            # The wall clock ran out before the simulation reached its stop
            # time, so the simulator was still busy, not merely idling to the
            # end. With a clock that stops after the last vector that leaves the
            # design itself: an operation that is very slow to evaluate, or a
            # combinational loop that keeps it cycling through delta steps.
            result['errors'].append(
                f'Simulation did not finish within {proc_timeout}s of wall-clock time '
                f'(simulated-time limit {stop_time}). Either the clock keeps running '
                f'after the last vector, or the design is very slow to evaluate each '
                f'cycle, for example because of a combinational loop.')
        except Exception as e:
            result['errors'].append(f'Simulation error: {str(e)}')

        # Cleanup work dir
        try:
            import shutil as _shutil
            _shutil.rmtree(work_dir, ignore_errors=True)
        except:
            pass

        return result

    @staticmethod
    def _decode_output(raw_bytes) -> str:
        """Decode subprocess output bytes trying multiple encodings.

        On Chinese Windows, compiler output may be GBK/CP936 encoded.
        """
        if not raw_bytes:
            return ''
        if isinstance(raw_bytes, str):
            return raw_bytes
        for enc in ('utf-8', 'gbk', 'cp936', 'latin-1'):
            try:
                return raw_bytes.decode(enc)
            except (UnicodeDecodeError, AttributeError):
                continue
        return str(raw_bytes)

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