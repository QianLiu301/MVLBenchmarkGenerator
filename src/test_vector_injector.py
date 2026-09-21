"""
Test Vector Injector — Strategy B
==================================
Generates per-language harness code that:
  1. Reads golden test vectors from stdin (serialized by golden_model.serialize_vectors)
  2. Calls the LLM-generated alu_exec() function for each vector
  3. Prints results in a standardised format the validator can parse

The harness replaces the LLM's own main/testbench so we test the ALU logic
with externally-controlled, golden-model-verified inputs.
"""

import re
import textwrap
from typing import Optional


# ====================================================================
# Public API
# ====================================================================

def generate_harness(language: str, k: int, bits: int, llm_code: str) -> str:
    """Return *complete* source that embeds the LLM ALU + a stdin-driven harness.

    Parameters
    ----------
    language : 'c', 'python', 'verilog', 'vhdl'
    k, bits  : ALU configuration (needed for module names in HDL)
    llm_code : Raw LLM-generated source code

    Returns
    -------
    Combined source ready for compilation/execution.
    """
    lang = language.lower()
    if lang == 'c':
        return _harness_c(llm_code, k, bits)
    elif lang == 'python':
        return _harness_python(llm_code, k, bits)
    elif lang == 'verilog':
        return _harness_verilog(llm_code, k, bits)
    elif lang == 'vhdl':
        return _harness_vhdl(llm_code, k, bits)
    else:
        raise ValueError(f"Unsupported language: {language}")


# ====================================================================
# C harness
# ====================================================================

def _strip_c_main(code: str) -> str:
    """Remove the LLM's main() function so we can substitute our own."""
    # Find 'int main' and remove everything from there to the matching closing brace
    pattern = re.compile(
        r'^[ \t]*int\s+main\s*\([^)]*\)\s*\{',
        re.MULTILINE,
    )
    m = pattern.search(code)
    if not m:
        return code

    start = m.start()
    # Find matching closing brace
    depth = 0
    i = m.end() - 1  # points at the opening '{'
    while i < len(code):
        if code[i] == '{':
            depth += 1
        elif code[i] == '}':
            depth -= 1
            if depth == 0:
                return code[:start] + code[i + 1:]
        i += 1
    # If we can't find the match, just truncate
    return code[:start]


_C_FLAG_NAMES = {
    'z': ('z', 'zero', 'zf', 'zero_flag', 'is_zero'),
    'n': ('n', 'neg', 'negative', 'nf', 'negative_flag', 'sign'),
    'c': ('c', 'carry', 'cf', 'carry_flag', 'cout', 'overflow'),
}
_C_RESULT_NAMES = ('result', 'value', 'res', 'val', 'r', 'out', 'y')


def _c_struct_fields(code: str, type_name: str):
    """Field list [(type, name)] of `typedef struct {...} type_name;` or `struct type_name {...}`."""
    # body must not contain braces, otherwise the pattern would span neighbouring structs
    m = (re.search(r'typedef\s+struct\s*\w*\s*\{([^{}]*)\}\s*' + re.escape(type_name) + r'\s*;', code, re.S)
         or re.search(r'struct\s+' + re.escape(type_name) + r'\s*\{([^{}]*)\}', code, re.S))
    if not m:
        return None
    body = re.sub(r'/\*.*?\*/', '', m.group(1), flags=re.S)
    body = re.sub(r'//[^\n]*', '', body)
    fields = []
    for decl in body.split(';'):
        decl = decl.strip()
        if not decl:
            continue
        parts = decl.replace('*', ' ').split()
        if len(parts) >= 2:
            fields.append((' '.join(parts[:-1]), parts[-1]))
    return fields


def _c_interface(code: str):
    """Work out how to read result and flags from what alu_exec() returns.

    The generator prompt only fixes the *contents* of the struct (result + flags
    z, n, c); models choose the type name and whether the flags are nested. Returns
    (return_type, result_expr, z_expr, n_expr, c_expr) with `r` as the variable, or
    None when the interface cannot be determined (reported as a harness error).
    """
    sig = re.search(r'\b(\w+)\s+alu_exec\s*\(', code)
    if not sig:
        return None
    rtype = sig.group(1)
    fields = _c_struct_fields(code, rtype)
    if not fields:
        return None

    def pick(names, flds, prefix):
        for _, fname in flds:
            if fname.lower() in names:
                return prefix + fname
        return None

    res = pick(_C_RESULT_NAMES, fields, 'r.')
    flag = {}
    for key, names in _C_FLAG_NAMES.items():
        flag[key] = pick(names, fields, 'r.')
    # flags nested in a sub-struct member (e.g. `Flags flags;`)
    if not all(flag.values()):
        for ftype, fname in fields:
            sub = _c_struct_fields(code, ftype.replace('struct', '').strip())
            if sub:
                for key, names in _C_FLAG_NAMES.items():
                    flag[key] = flag[key] or pick(names, sub, f'r.{fname}.')
    if res is None:
        # a lone integer field that is not a flag is the result
        others = [f for t, f in fields if f.lower() not in sum(_C_FLAG_NAMES.values(), ())]
        res = f'r.{others[0]}' if len(others) == 1 else None
    if res is None or not all(flag.values()):
        return None
    return rtype, res, flag['z'], flag['n'], flag['c']


def _harness_c(llm_code: str, k: int, bits: int) -> str:
    alu_only = _strip_c_main(llm_code)
    iface = _c_interface(alu_only)
    if iface is None:
        raise ValueError('C harness: could not determine the alu_exec() result/flag interface '
                         '(expected a struct with a result field and flags z/n/c, flat or nested)')
    rtype, res, z, n, c = iface

    # Ensure essential headers are present
    headers = ''
    for h in ['stdio.h', 'stdlib.h', 'stdint.h', 'stdbool.h']:
        if h not in alu_only:
            headers += f'#include <{h}>\n'

    harness_main = textwrap.dedent(r"""
        /* ---- Strategy-B stdin-driven harness ---- */
        int main(void) {
            int count;
            if (scanf("%d", &count) != 1) return 1;

            for (int i = 0; i < count; i++) {
                int op;
                unsigned long long a_val, b_val;
                unsigned long long exp_r;
                int exp_z, exp_n, exp_c;
                if (scanf("%d %llu %llu %llu %d %d %d",
                          &op, &a_val, &b_val, &exp_r, &exp_z, &exp_n, &exp_c) != 7) {
                    fprintf(stderr, "Parse error on vector %d\n", i);
                    return 1;
                }

                /* Call the LLM ALU (op is passed as int; enums accept it) */
                RTYPE r = alu_exec((uint64_t)a_val, (uint64_t)b_val, op);

                const char* names[] = {"ADD","SUB","MUL","NEG","INC","DEC"};
                const char* op_str = (op >= 0 && op <= 5) ? names[op] : "???";
                printf("Test %2d: %s A=%llu B=%llu -> R=%llu Z=%d N=%d C=%d\n",
                       i + 1, op_str,
                       (unsigned long long)a_val,
                       (unsigned long long)b_val,
                       (unsigned long long)(RES),
                       (int)(ZF), (int)(NF), (int)(CF));
            }
            return 0;
        }
    """)
    harness_main = (harness_main.replace('RTYPE', rtype).replace('(RES)', f'({res})')
                    .replace('(ZF)', f'({z})').replace('(NF)', f'({n})').replace('(CF)', f'({c})'))
    return headers + '\n' + alu_only.rstrip() + '\n' + harness_main


# ====================================================================
# Python harness
# ====================================================================

def _strip_python_main(code: str) -> str:
    """Remove 'if __name__' block from LLM Python code."""
    pattern = re.compile(
        r'^if\s+__name__\s*==\s*["\']__main__["\']\s*:',
        re.MULTILINE,
    )
    m = pattern.search(code)
    if not m:
        return code

    start = m.start()
    # Remove from the if __name__ line to the end, or until the next
    # top-level (non-indented, non-empty) line
    lines = code[m.end():].split('\n')
    end_offset = len(code)
    cumulative = m.end()
    for idx, line in enumerate(lines):
        cumulative += len(line) + 1
        if idx == 0:
            continue  # skip the rest of the 'if' line
        if line.strip() == '':
            continue
        if not line[0].isspace():
            # Found a new top-level definition — stop removing
            end_offset = cumulative - len(line) - 1
            break

    return code[:start] + code[end_offset:]


def _harness_python(llm_code: str, k: int, bits: int) -> str:
    alu_only = _strip_python_main(llm_code)

    harness = textwrap.dedent("""\
        # ---- Strategy-B stdin-driven harness ----
        import sys as _sys

        _OP_NAMES = {0: 'ADD', 1: 'SUB', 2: 'MUL', 3: 'NEG', 4: 'INC', 5: 'DEC'}

        def _run_harness():
            lines = _sys.stdin.read().strip().split('\\n')
            count = int(lines[0])
            for i in range(count):
                parts = lines[1 + i].split()
                op, a, b = int(parts[0]), int(parts[1]), int(parts[2])
                result, flags = alu_exec(a, b, op)
                op_str = _OP_NAMES.get(op, '???')
                z = int(flags.z) if hasattr(flags, 'z') else (int(flags.zero) if hasattr(flags, 'zero') else 0)
                n = int(flags.n) if hasattr(flags, 'n') else (int(flags.negative) if hasattr(flags, 'negative') else 0)
                c = int(flags.c) if hasattr(flags, 'c') else (int(flags.carry) if hasattr(flags, 'carry') else 0)
                print(f"Test {i+1:2d}: {op_str} A={a} B={b} -> R={result} Z={z} N={n} C={c}")

        if __name__ == '__main__':
            _run_harness()
    """)

    return alu_only.rstrip() + '\n\n' + harness


# ====================================================================
# Verilog harness
# ====================================================================

def _strip_verilog_testbench(code: str) -> str:
    """Remove existing testbench module(s) from Verilog code."""
    # Remove any module whose name ends with '_tb'
    result = code
    pattern = re.compile(
        r'^[ \t]*module\s+\w+_tb\b[^;]*;',
        re.MULTILINE,
    )
    while True:
        m = pattern.search(result)
        if not m:
            break
        # Find matching endmodule
        end_pat = re.compile(r'^[ \t]*endmodule\b', re.MULTILINE)
        end_m = end_pat.search(result, m.end())
        if end_m:
            result = result[:m.start()] + result[end_m.end():]
        else:
            result = result[:m.start()]
    return result


def _harness_verilog(llm_code: str, k: int, bits: int) -> str:
    import math
    mod = k ** bits
    data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
    module_name = f"mvl_alu_{k}_{bits}bit"

    alu_only = _strip_verilog_testbench(llm_code)

    harness_tb = textwrap.dedent(f"""\

        /* ---- Strategy-B stdin-driven testbench ---- */
        module {module_name}_tb;
            reg clk, rst;
            reg [{data_width - 1}:0] a, b;
            reg [3:0] opcode;
            wire [{data_width - 1}:0] result;
            wire zero, negative, carry;

            {module_name} uut (
                .clk(clk), .rst(rst),
                .a(a), .b(b), .opcode(opcode),
                .result(result), .zero(zero),
                .negative(negative), .carry(carry)
            );

            // Clock
            always #5 clk = ~clk;

            integer fd, count, i, op_val, status;
            integer a_val, b_val, exp_r, exp_z, exp_n, exp_c;

            initial begin
                clk = 0; rst = 1;
                #20 rst = 0;
                #10;

                // Read vector count from stdin
                status = $fscanf('h8000_0000, "%d", count);

                for (i = 0; i < count; i = i + 1) begin
                    status = $fscanf('h8000_0000, "%d %d %d %d %d %d %d",
                                     op_val, a_val, b_val, exp_r, exp_z, exp_n, exp_c);

                    a = a_val;
                    b = b_val;
                    opcode = op_val;
                    @(posedge clk);  // wait one clock cycle
                    @(posedge clk);  // extra cycle for propagation
                    #1;              // small delta for output to settle

                    $display("Test %0d: OP=%0d A=%0d B=%0d R=%0d Z=%0d N=%0d C=%0d",
                             i + 1, op_val, a_val, b_val, result, zero, negative, carry);
                end

                $finish;
            end
        endmodule
    """)

    return alu_only.rstrip() + '\n' + harness_tb


# ====================================================================
# VHDL harness
# ====================================================================

def _strip_vhdl_testbench(code: str) -> str:
    """Remove existing testbench entity/architecture from VHDL code."""
    # Remove entity + architecture for *_tb
    result = code
    # Find entity foo_tb is ... end entity
    pattern = re.compile(
        r'(^[ \t]*library\s+IEEE;[^;]*;[^;]*;\s*)?'
        r'^[ \t]*entity\s+\w+_tb\s+is\b',
        re.MULTILINE | re.IGNORECASE,
    )
    m = pattern.search(result)
    if m:
        # Remove from the matched entity to end of file (TB is always last)
        result = result[:m.start()]
    return result


def _harness_vhdl(llm_code: str, k: int, bits: int) -> str:
    import math
    mod = k ** bits
    data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
    module_name = f"mvl_alu_{k}_{bits}bit"

    alu_only = _strip_vhdl_testbench(llm_code)

    # VHDL testbench that reads from a file (stdin not directly supported in VHDL)
    # We'll use a text file approach — the validator writes vectors to a temp file
    harness_tb = textwrap.dedent(f"""\

        -- ---- Strategy-B file-driven testbench ----
        library IEEE;
        use IEEE.STD_LOGIC_1164.ALL;
        use IEEE.NUMERIC_STD.ALL;
        use STD.TEXTIO.ALL;

        entity {module_name}_tb is
        end entity {module_name}_tb;

        architecture test of {module_name}_tb is
            signal clk_sig     : std_logic := '0';
            signal rst_sig     : std_logic := '1';
            signal a_sig       : std_logic_vector({data_width - 1} downto 0) := (others => '0');
            signal b_sig       : std_logic_vector({data_width - 1} downto 0) := (others => '0');
            signal opcode_sig  : std_logic_vector(3 downto 0) := (others => '0');
            signal result_sig  : std_logic_vector({data_width - 1} downto 0);
            signal zero_sig    : std_logic;
            signal negative_sig: std_logic;
            signal carry_sig   : std_logic;
            -- Set by stim_proc when the vector file is exhausted; stops the clock so
            -- the simulation ends on its own instead of running to --stop-time
            -- (a free-running clock kept the DUT evaluating the last opcode for
            -- 10^7 cycles, which for a k=4 MUL exceeded the process timeout).
            signal sim_done    : boolean := false;

            constant CLK_PERIOD : time := 10 ns;
        begin
            uut: entity work.{module_name}
                port map (
                    clk      => clk_sig,
                    rst      => rst_sig,
                    a        => a_sig,
                    b        => b_sig,
                    opcode   => opcode_sig,
                    result   => result_sig,
                    zero     => zero_sig,
                    negative => negative_sig,
                    carry    => carry_sig
                );

            clk_proc: process
            begin
                while not sim_done loop
                    clk_sig <= '0'; wait for CLK_PERIOD / 2;
                    clk_sig <= '1'; wait for CLK_PERIOD / 2;
                end loop;
                wait;
            end process;

            stim_proc: process
                file vector_file : text open read_mode is "test_vectors.txt";
                variable v_line  : line;
                variable v_count : integer;
                variable v_op    : integer;
                variable v_a     : integer;
                variable v_b     : integer;
                variable v_exp_r : integer;
                variable v_exp_z : integer;
                variable v_exp_n : integer;
                variable v_exp_c : integer;
                variable v_space : character;
            begin
                -- Reset
                rst_sig <= '1';
                wait for CLK_PERIOD * 2;
                rst_sig <= '0';
                wait for CLK_PERIOD;

                -- Read vector count
                readline(vector_file, v_line);
                read(v_line, v_count);

                for i in 1 to v_count loop
                    readline(vector_file, v_line);
                    read(v_line, v_op);    read(v_line, v_space);
                    read(v_line, v_a);     read(v_line, v_space);
                    read(v_line, v_b);     read(v_line, v_space);
                    read(v_line, v_exp_r); read(v_line, v_space);
                    read(v_line, v_exp_z); read(v_line, v_space);
                    read(v_line, v_exp_n); read(v_line, v_space);
                    read(v_line, v_exp_c);

                    a_sig      <= std_logic_vector(to_unsigned(v_a, {data_width}));
                    b_sig      <= std_logic_vector(to_unsigned(v_b, {data_width}));
                    opcode_sig <= std_logic_vector(to_unsigned(v_op, 4));
                    wait for CLK_PERIOD * 2;

                    report "Test " & integer'image(i) &
                           ": OP=" & integer'image(v_op) &
                           " A=" & integer'image(v_a) &
                           " B=" & integer'image(v_b) &
                           " R=" & integer'image(to_integer(unsigned(result_sig))) &
                           " Z=" & std_logic'image(zero_sig) &
                           " N=" & std_logic'image(negative_sig) &
                           " C=" & std_logic'image(carry_sig);
                end loop;

                report "STRATEGY_B_DONE" severity note;
                sim_done <= true;
                wait;
            end process;
        end architecture test;
    """)

    return alu_only.rstrip() + '\n' + harness_tb
