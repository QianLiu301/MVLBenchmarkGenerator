"""
MVL Benchmark Generator
=======================
Generate Multi-Valued Logic ALU benchmarks using LLMs.

Supports:
- K-values: 2, 3, 4, 5, 6, 7, 8
- Bitwidths: 8, 10, 12, 14
- Languages: C, Python, Verilog, VHDL
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from src.galois_field import resolve_logic_type, format_table_for_prompt
except ImportError:
    from galois_field import resolve_logic_type, format_table_for_prompt


class MVLGenerator:
    """Generate MVL ALU code using LLM"""

    def __init__(
            self,
            llm_provider: str = 'groq',
            model: Optional[str] = None,
            output_dir: Optional[str] = None,
            project_root: Optional[str] = None
    ):
        self.llm_provider_name = llm_provider.lower()
        self.model = model
        self.llm = self._setup_llm()
        self.output_dir = self._setup_output_dir(output_dir, project_root)
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Show actual provider and model
        actual_model = getattr(self.llm, 'model', None) if self.llm else None
        actual_class = type(self.llm).__name__ if self.llm else 'None'
        print(f"\n🔢 MVL Generator initialized")
        print(f"   Provider: {self.llm_provider_name} ({actual_class})")
        print(f"   Model: {actual_model or 'N/A'}")
        print(f"   Output directory: {self.output_dir}")

    def _load_config_api_key(self, provider_name: str) -> Optional[str]:
        """Load API key from llm_config.json for the given provider"""
        import json
        project_root = Path(__file__).parent.parent
        config_locations = [
            project_root / 'config' / 'llm_config.json',      # config/
            project_root / 'llm_config.json',                  # project root
            Path(__file__).parent / 'llm_config.json',         # src/
            Path.cwd() / 'llm_config.json',                    # CWD
            Path.cwd() / 'config' / 'llm_config.json',        # CWD/config/
        ]

        print(f"   🔍 Searching for llm_config.json (provider: {provider_name})...")
        for config_path in config_locations:
            try:
                resolved = config_path.resolve()
                exists = config_path.exists()
                print(f"      Checking: {resolved} -> {'FOUND' if exists else 'not found'}")
                if exists:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    # Print top-level keys to understand structure
                    print(f"      Config top-level keys: {list(config.keys())}")

                    # Try direct lookup: config["together"]
                    provider_config = config.get(provider_name, {})

                    # If not found directly, search common wrapper keys
                    if not provider_config:
                        for wrapper_key in ['providers', 'llm_providers', 'llm', 'models']:
                            wrapper = config.get(wrapper_key, {})
                            if isinstance(wrapper, dict) and provider_name in wrapper:
                                provider_config = wrapper[provider_name]
                                print(f"      Found under '{wrapper_key}.{provider_name}'")
                                break

                    # Also handle list-style configs where providers are in a list
                    if not provider_config and isinstance(config.get('providers'), list):
                        for item in config['providers']:
                            if isinstance(item, dict) and item.get('name') == provider_name:
                                provider_config = item
                                print(f"      Found in providers list by name")
                                break

                    api_key = provider_config.get('api_key', '') if isinstance(provider_config, dict) else ''
                    if api_key:
                        print(f"   📄 Config loaded from: {resolved}")
                        print(f"      API key found for '{provider_name}': {api_key[:4]}***{api_key[-4:]}")
                        return api_key
                    else:
                        print(f"      ⚠️ Config found but no api_key for '{provider_name}'")
                        print(f"      provider_config = {provider_config}")

                    # Also check for model override from config
                    if isinstance(provider_config, dict):
                        model_from_config = provider_config.get('model', '')
                        if model_from_config and not self.model:
                            self.model = model_from_config
            except Exception as e:
                print(f"      ❌ Error reading {config_path}: {e}")
                continue
        print(f"   ⚠️ No API key found for '{provider_name}' in any config file")
        return None

    def _setup_llm(self):
        """Setup LLM provider"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from llm_providers import (
                GeminiProvider,
                OpenAIProvider,
                ClaudeProvider,
                GroqProvider,
                DeepSeekProvider,
                QwenProvider,
                MistralProvider,
                TogetherProvider,
                GrokProvider,
                LocalLLMProvider,
            )

            providers = {
                'gemini': GeminiProvider,
                'openai': OpenAIProvider,
                'gpt': OpenAIProvider,
                'claude': ClaudeProvider,
                'groq': GroqProvider,
                'deepseek': DeepSeekProvider,
                'qwen': QwenProvider,
                'mistral': MistralProvider,
                'together': TogetherProvider,
                'grok': GrokProvider,
                'local': LocalLLMProvider,
            }

            if self.llm_provider_name not in providers:
                print(f"⚠️  Unknown LLM provider: {self.llm_provider_name}")
                print(f"   Available: {', '.join(providers.keys())}")
                self.llm_provider_name = 'groq'

            provider_class = providers[self.llm_provider_name]

            # Load API key from llm_config.json if not set in environment
            kwargs = {}
            config_api_key = self._load_config_api_key(self.llm_provider_name)
            if config_api_key:
                kwargs['api_key'] = config_api_key

            if self.model:
                kwargs['model'] = self.model

            # LocalLLMProvider doesn't accept kwargs
            if self.llm_provider_name == 'local':
                llm = provider_class()
            else:
                llm = provider_class(**kwargs)

            # Log the actual provider and model
            actual_model = getattr(llm, 'model', 'N/A')
            print(f"✅ LLM provider loaded: {provider_class.__name__}")
            print(f"   Provider: {self.llm_provider_name}")
            print(f"   Model: {actual_model}")

            return llm

        except ImportError as e:
            print(f"❌ Failed to import LLM providers: {e}")
            return None
        except ValueError as e:
            print(f"❌ Failed to initialize provider '{self.llm_provider_name}': {e}")
            print(f"   Please set the API key in llm_config.json or as an environment variable")
            return None
        except Exception as e:
            print(f"❌ Unexpected error setting up provider '{self.llm_provider_name}': {e}")
            return None

    def _setup_output_dir(self, output_dir: Optional[str], project_root: Optional[str]) -> Path:
        """Setup output directory"""
        if output_dir:
            base_dir = Path(output_dir)
        elif project_root:
            base_dir = Path(project_root) / "output" / "mvl_code"
        else:
            base_dir = Path.cwd() / "output" / "mvl_code"

        llm_dir = base_dir / self.llm_provider_name
        llm_dir.mkdir(parents=True, exist_ok=True)

        return llm_dir

    @staticmethod
    def _parse_natural_input(text: str) -> Dict[str, Optional[int]]:
        """Parse k_value and bitwidth from natural language input.

        Examples:
            '10-trit ternary calculator' -> {'k_value': 3, 'bitwidth': 10}
            'GF(5) 12-trit ALU'         -> {'k_value': 5, 'bitwidth': 12}
            'quaternary 8-bit unit'      -> {'k_value': 4, 'bitwidth': 8}
        """
        result: Dict[str, Optional[int]] = {'k_value': None, 'bitwidth': None}
        lower = text.lower()

        # --- Parse k_value ---
        # Pattern: GF(n) or gf(n)
        m = re.search(r'gf\s*\(\s*(\d+)\s*\)', lower)
        if m:
            result['k_value'] = int(m.group(1))

        # Named radixes (use word boundary to avoid 'ternary' matching inside 'quaternary')
        if result['k_value'] is None:
            radix_map = {
                'binary': 2, 'ternary': 3, 'quaternary': 4,
                'quinary': 5, 'senary': 6, 'septenary': 7, 'octal': 8,
            }
            for name, k in radix_map.items():
                if re.search(r'\b' + name + r'\b', lower):
                    result['k_value'] = k
                    break

        # Pattern: k=N or k-value N or k value N
        if result['k_value'] is None:
            m = re.search(r'k\s*[=\-:]\s*(?:value\s*)?(\d+)', lower)
            if m:
                result['k_value'] = int(m.group(1))

        # --- Parse bitwidth ---
        # Pattern: N-trit or N trit
        m = re.search(r'(\d+)\s*-?\s*trit', lower)
        if m:
            result['bitwidth'] = int(m.group(1))

        # Pattern: N-bit or N bit
        if result['bitwidth'] is None:
            m = re.search(r'(\d+)\s*-?\s*bit', lower)
            if m:
                result['bitwidth'] = int(m.group(1))

        # Pattern: bitwidth=N or bitwidth N
        if result['bitwidth'] is None:
            m = re.search(r'bitwidth\s*[=:]\s*(\d+)', lower)
            if m:
                result['bitwidth'] = int(m.group(1))

        return result

    @staticmethod
    def _resolve_logic_type(k_value: int) -> Dict:
        """Determine the algebraic structure for the given K-value.

        Returns a dict with 'display', 'category', 'p', 'n', 'tables'.
        See galois_field.resolve_logic_type() for details.
        """
        return resolve_logic_type(k_value)

    @staticmethod
    def _compute_gf_test_values(k: int, bits: int, logic_info: Dict) -> Dict:
        """Compute expected ALU test results using GF(p^n) digit-wise operations.

        Returns a dict with pre-computed expected values for edge-case and
        small-value tests, suitable for embedding in LLM prompts and testbenches.
        """
        tables = logic_info['tables']
        add_tbl = tables['add_table']
        mul_tbl = tables['mul_table']
        mod = k ** bits
        max_val = mod - 1

        def to_digits(value):
            digits = []
            for _ in range(bits):
                digits.append(value % k)
                value //= k
            return digits

        def from_digits(digits):
            value = 0
            for i, d in enumerate(digits):
                value += d * (k ** i)
            return value

        def gf_add(a, b):
            ad, bd = to_digits(a), to_digits(b)
            return from_digits([add_tbl[ad[i]][bd[i]] for i in range(bits)])

        def gf_sub(a, b):
            ad, bd = to_digits(a), to_digits(b)
            result = []
            for i in range(bits):
                for x in range(k):
                    if add_tbl[x][bd[i]] == ad[i]:
                        result.append(x)
                        break
            return from_digits(result)

        def gf_mul(a, b):
            ad, bd = to_digits(a), to_digits(b)
            result = [0] * bits
            for i in range(bits):
                for j in range(bits):
                    if i + j < bits:
                        result[i + j] = add_tbl[result[i + j]][mul_tbl[ad[i]][bd[j]]]
            return from_digits(result)

        def gf_neg(a):
            ad = to_digits(a)
            result = []
            for d in ad:
                for x in range(k):
                    if add_tbl[x][d] == 0:
                        result.append(x)
                        break
            return from_digits(result)

        def gf_inc(a):
            ad = to_digits(a)
            ad[0] = add_tbl[ad[0]][1]
            return from_digits(ad)

        def gf_dec(a):
            ad = to_digits(a)
            inv1 = 0
            for x in range(k):
                if add_tbl[x][1] == 0:
                    inv1 = x
                    break
            ad[0] = add_tbl[ad[0]][inv1]
            return from_digits(ad)

        return {
            'add_0_0': gf_add(0, 0),
            'sub_0_0': gf_sub(0, 0),
            'mul_0_0': gf_mul(0, 0),
            'neg_0': gf_neg(0),
            'inc_0': gf_inc(0),
            'dec_0': gf_dec(0),
            'add_max': gf_add(max_val, max_val),
            'sub_max': gf_sub(max_val, max_val),
            'mul_max': gf_mul(max_val, max_val),
            'neg_max': gf_neg(max_val),
            'inc_max': gf_inc(max_val),
            'dec_max': gf_dec(max_val),
            'add_10_20': gf_add(10, 20),
            'sub_20_10': gf_sub(20, 10),
            'sub_10_20': gf_sub(10, 20),
            'mul_10_20': gf_mul(10, 20),
            'neg_10': gf_neg(10),
            'neg_1': gf_neg(1),
            'inc_10': gf_inc(10),
            'dec_10': gf_dec(10),
        }

    @staticmethod
    def _vhdl_unsigned_literal(value: int, width: int) -> str:
        """Generate a VHDL unsigned literal that avoids integer overflow.

        For values <= 2^31-1, returns: to_unsigned(value, width)
        For values > 2^31-1, returns hex-based literal to avoid VHDL integer overflow.
        """
        if value <= 2147483647:
            return f"to_unsigned({value}, {width})"

        # Build hex literal for large values
        hex_str = format(value, 'X')
        hex_bits = len(hex_str) * 4
        if hex_bits < width:
            # Need leading binary bits to reach exact width
            leading_bits = width - hex_bits
            leading_val = value >> hex_bits
            leading_bin = format(leading_val, f'0{leading_bits}b')
            return f'"{leading_bin}" & x"{hex_str[-hex_bits//4:]}"'
        elif hex_bits == width:
            return f'x"{hex_str}"'
        else:
            # hex_bits > width: trim leading hex, use binary prefix
            full_bin = format(value, f'0{width}b')
            # Split into leading binary + hex portion
            hex_chars = width // 4
            remaining_bits = width - hex_chars * 4
            hex_portion = format(value & ((1 << (hex_chars * 4)) - 1), f'0{hex_chars}X')
            if remaining_bits > 0:
                leading_val = value >> (hex_chars * 4)
                leading_bin = format(leading_val, f'0{remaining_bits}b')
                return f'"{leading_bin}" & x"{hex_portion}"'
            else:
                return f'x"{hex_portion}"'

    @staticmethod
    def _vhdl_slv_literal(value: int, width: int) -> str:
        """Generate std_logic_vector literal for VHDL testbench assertions.

        For values <= 2^31-1: std_logic_vector(to_unsigned(value, width))
        For values > 2^31-1: uses hex-based unsigned literal
        """
        if value <= 2147483647:
            return f"std_logic_vector(to_unsigned({value}, {width}))"
        else:
            lit = MVLGenerator._vhdl_unsigned_literal(value, width)
            return f"std_logic_vector({lit})"

    @staticmethod
    def _build_algebra_section(k: int, bits: int, mod: int, logic_info: Dict, language: str = 'c') -> str:
        """Build the algebra-specific operation description for LLM prompts.

        Returns a prompt section that describes how operations should be implemented
        based on the algebraic structure (prime field, extension field, or integer ring).
        """
        import math
        category = logic_info['category']
        display = logic_info['display']
        max_val = mod - 1
        mul_product_max = max_val * max_val
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1

        # Determine required intermediate width for MUL
        if mul_product_max > 0:
            mul_bits_needed = math.ceil(math.log2(mul_product_max + 1))
        else:
            mul_bits_needed = 1

        if mul_product_max <= 2**32 - 1:
            c_mul_type = "uint32_t"
            c_mul_note = f"uint32_t is sufficient ({max_val} * {max_val} = {mul_product_max} < 2^32)"
        elif mul_product_max <= 2**64 - 1:
            c_mul_type = "uint64_t"
            c_mul_note = f"uint64_t is sufficient ({max_val} * {max_val} = {mul_product_max} < 2^64)"
        else:
            c_mul_type = "__uint128_t"
            c_mul_note = (f"⚠️ uint64_t is NOT sufficient! {max_val} * {max_val} = {mul_product_max} > 2^64. "
                          f"MUST use __uint128_t (GCC/Clang) or split into high/low 64-bit words")

        mul_overflow_warning = ""
        if mul_product_max > 2**64 - 1:
            mul_overflow_warning = f"""
  ⚠️ CRITICAL OVERFLOW WARNING FOR MUL:
  Max product = {max_val} × {max_val} = {mul_product_max}
  This EXCEEDS uint64_t max (18446744073709551615).
  In C: use {c_mul_type} for the intermediate product.
  In Verilog: use [{mul_bits_needed - 1}:0] for the temp register.
  In VHDL: use unsigned({mul_bits_needed - 1} downto 0) for the intermediate."""

        if category == 'extension_field' and logic_info.get('tables'):
            # GF(p^n) extension field — digit-wise operations using pre-computed tables
            p = logic_info['p']
            n = logic_info['n']
            tables = logic_info['tables']
            add_tbl = tables['add_table']
            mul_tbl = tables['mul_table']
            irr_poly = tables['irreducible_poly']

            # Format tables as 2D array literals
            add_literal = "[\n" + ",\n".join(
                "    [" + ", ".join(str(v) for v in row) + "]" for row in add_tbl
            ) + "\n  ]"
            mul_literal = "[\n" + ",\n".join(
                "    [" + ", ".join(str(v) for v in row) + "]" for row in mul_tbl
            ) + "\n  ]"

            # Format readable tables
            add_readable = format_table_for_prompt(add_tbl, 'GF_ADD', k)
            mul_readable = format_table_for_prompt(mul_tbl, 'GF_MUL', k)

            section = f"""ALGEBRAIC STRUCTURE: {display} — Extension Field
  Irreducible polynomial: {irr_poly} over GF({p})
  ⚠️  This is NOT simple modular arithmetic. GF({k}) ≠ Z/{k}Z.

  Each operand is a {bits}-digit number in base {k}. Each digit is an element of GF({k}).
  Operations are performed DIGIT-BY-DIGIT using the field tables below — NO carry propagation between digits.

  GF({k}) Addition Table (pre-computed, mathematically verified):
  {add_readable}

  GF({k}) Multiplication Table (pre-computed, mathematically verified):
  {mul_readable}

  The tables as arrays (copy directly into your code):
  GF_ADD = {add_literal}
  GF_MUL = {mul_literal}

  OPERATION DEFINITIONS (all digit-wise, using the tables above):
  - ADD: for each digit position i: result_digit[i] = GF_ADD[a_digit[i]][b_digit[i]]
  - SUB: for each digit position i: find x such that GF_ADD[x][b_digit[i]] == a_digit[i] (additive inverse then add)
  - MUL: convolution of digits using GF_MUL, then reduce each digit mod the field.
         For simplicity, implement as schoolbook polynomial multiplication:
         for i in range({bits}): for j in range({bits}):
           if i+j < {bits}: result[i+j] = GF_ADD[result[i+j]][GF_MUL[a[i]][b[j]]]
  - NEG: for each digit position i: find x such that GF_ADD[x][a_digit[i]] == 0
         (additive inverse in GF({k}), using the addition table)
         In GF({p}^{n}), characteristic is {p}. {"Since char=2, every element is its own additive inverse, so NEG(a) = a (identity)." if p == 2 else f"For each digit d, negate each polynomial coefficient mod {p}."}
  - INC: add 1 to the least significant digit using GF_ADD: result_digit[0] = GF_ADD[a_digit[0]][1], rest unchanged
  - DEC: subtract 1 from the least significant digit: find additive inverse of 1, then GF_ADD

  HELPER FUNCTIONS NEEDED:
  1. to_digits(value, k={k}, width={bits}): convert integer to array of {bits} base-{k} digits
  2. from_digits(digits, k={k}): convert digit array back to integer
  3. gf_add_digits(a_digits, b_digits): digit-wise addition using GF_ADD table
  4. gf_mul_digits(a_digits, b_digits): polynomial multiplication using GF_MUL table

  FLAGS:
  - Z (zero): result == 0
  - N (negative): always false (field elements are non-negative)
  - C (carry): not applicable for field arithmetic, always false"""

        elif category == 'prime_field':
            # GF(p) — standard modular integer arithmetic
            p = logic_info['p']
            section = f"""ALGEBRAIC STRUCTURE: {display} — Prime Field
  All operations use standard modular integer arithmetic mod {mod} (= {k}^{bits}).
  This is equivalent to base-{k} integer arithmetic with carry propagation.
  Operand range: 0 to {max_val}. Binary bit width: {data_width} bits.
{mul_overflow_warning}
  OPERATION DEFINITIONS (all results mod {mod}):
  - ADD: (a + b) % {mod}
  - SUB: (a - b + {mod}) % {mod}
  - MUL: ({c_mul_type})(a) * b) % {mod}  — {c_mul_note}
  - NEG: ({mod} - a) % {mod}   ⚠️ The % {mod} is MANDATORY! NEG(0) must be 0, not {mod}.
  - INC: (a + 1) % {mod}   — adds exactly 1, NOT {k} or any other value
  - DEC: (a - 1 + {mod}) % {mod}   — subtracts exactly 1

  CARRY/FLAG DETECTION (must be checked BEFORE taking mod):
  - ADD carry: (a + b) >= {mod}
  - SUB borrow: a < b
  - MUL carry: always 0 (no carry for multiplication)
  - NEG carry: always 0
  - INC carry: a == {max_val}
  - DEC borrow: a == 0
  - Z (zero): result == 0
  - N (negative): result >= {mod // 2}"""

        else:
            # Z/kZ — integer ring, same as prime field arithmetic
            section = f"""ALGEBRAIC STRUCTURE: Z/{k}Z — Integer Ring (modular arithmetic)
  All operations use standard modular integer arithmetic mod {mod} (= {k}^{bits}).
  Note: not all elements have multiplicative inverses (non-field).
  Operand range: 0 to {max_val}. Binary bit width: {data_width} bits.
{mul_overflow_warning}
  OPERATION DEFINITIONS (all results mod {mod}):
  - ADD: (a + b) % {mod}
  - SUB: (a - b + {mod}) % {mod}
  - MUL: ({c_mul_type})(a) * b) % {mod}  — {c_mul_note}
  - NEG: ({mod} - a) % {mod}   ⚠️ The % {mod} is MANDATORY! NEG(0) must be 0, not {mod}.
  - INC: (a + 1) % {mod}   — adds exactly 1, NOT {k} or any other value
  - DEC: (a - 1 + {mod}) % {mod}   — subtracts exactly 1

  CARRY/FLAG DETECTION (must be checked BEFORE taking mod):
  - ADD carry: (a + b) >= {mod}
  - SUB borrow: a < b
  - MUL carry: always 0 (no carry for multiplication)
  - NEG carry: always 0
  - INC carry: a == {max_val}
  - DEC borrow: a == 0
  - Z (zero): result == 0
  - N (negative): result >= {mod // 2}"""

        return section

    def generate(
            self,
            k_value: int = 3,
            bitwidth: int = 8,
            language: str = 'c',
            operations: List[str] = None,
            natural_input: str = '',
            module_type: str = 'alu',
            register_count: int = None,
            pipeline_stages: int = None
    ) -> Dict:
        """
        Generate MVL code for the specified module type.

        Args:
            k_value: Number of logic values (2-8)
            bitwidth: Number of digits (8, 10, 12, 14)
            language: Output language ('c', 'python', 'verilog', 'vhdl')
            operations: List of operations to include
            natural_input: Natural language description (takes priority for prompt/naming when provided)
            module_type: Hardware module type ('alu', 'counter', 'register', 'cpu-risc-v')
            register_count: Number of registers for register file module (4, 8, 16, 32)
            pipeline_stages: Number of pipeline stages for CPU module (3, 5, 7)

        Returns:
            Dict with generation results
        """
        if operations is None:
            operations = ['ADD', 'SUB', 'MUL', 'NEG', 'INC', 'DEC']

        # When natural_input is provided, parse k_value/bitwidth from it (overrides dropdowns)
        if natural_input:
            parsed = self._parse_natural_input(natural_input)
            if parsed['k_value'] is not None:
                k_value = parsed['k_value']
            if parsed['bitwidth'] is not None:
                bitwidth = parsed['bitwidth']

        logic_info = self._resolve_logic_type(k_value)
        resolved_logic = logic_info['display']
        module_label = module_type.upper().replace('-', ' ')

        actual_model = getattr(self.llm, 'model', 'N/A') if self.llm else 'N/A'
        actual_class = type(self.llm).__name__ if self.llm else 'None'
        print(f"\n{'=' * 60}")
        print(f"🔢 Generating MVL {module_label}")
        if natural_input:
            print(f"   Natural input: {natural_input}")
        print(f"   Module type: {module_label}")
        print(f"   K-value: {k_value} ({resolved_logic})")
        print(f"   Logic type: {logic_info['category']}")
        print(f"   Bitwidth: {bitwidth}-trit")
        print(f"   Language: {language.upper()}")
        print(f"   Operations: {', '.join(operations)}")
        if register_count is not None:
            print(f"   Register count: {register_count}")
        if pipeline_stages is not None:
            print(f"   Pipeline stages: {pipeline_stages}")
        print(f"   LLM Provider: {self.llm_provider_name} ({actual_class})")
        print(f"   LLM Model: {actual_model}")
        print(f"{'=' * 60}")

        # Calculate MOD value
        mod_value = k_value ** bitwidth

        # Create prompt based on module type
        prompt = self._create_prompt_for_module(
            module_type, k_value, bitwidth, mod_value, operations,
            language, logic_info, natural_input,
            register_count=register_count, pipeline_stages=pipeline_stages
        )
        if prompt is None:
            return {'success': False, 'error': f'Unsupported language: {language}'}

        # Call LLM
        if self.llm is None:
            return {
                'success': False,
                'error': f'LLM provider "{self.llm_provider_name}" failed to initialize. '
                         f'Please check your API key in llm_config.json or set the corresponding environment variable.'
            }

        actual_model = getattr(self.llm, 'model', 'N/A')
        print(f"\n🤖 Calling {self.llm_provider_name.upper()} API (model: {actual_model})...")

        try:
            response = self.llm._call_api(
                prompt,
                max_tokens=16384,
                system_prompt="You are an expert programmer. Generate clean, compilable code without any explanations."
            )

            print(f"✅ LLM response received ({len(response)} chars)")

            # Extract code
            code = self._extract_code(response, language)

            if not code:
                return {'success': False, 'error': 'Failed to extract code from response'}

            # Fix common issues
            code = self._fix_code(code, language)

            # Auto-enhance test coverage if too low
            code = self._enhance_test_coverage(code, language, k_value, bitwidth, mod_value)

            # Validate generated code quality
            validation_warnings = self._validate_code(code, language, k_value, bitwidth, mod_value, logic_info=logic_info)

            # Save file
            file_path = self._save_code(code, k_value, bitwidth, language,
                                        natural_input=natural_input, module_type=module_type)

            print(f"✅ Code saved: {file_path}")

            return {
                'success': True,
                'code': code,
                'file_path': str(file_path),
                'filename': file_path.name,
                'k_value': k_value,
                'bitwidth': bitwidth,
                'language': language,
                'mod_value': mod_value,
                'module_type': module_type,
                'logic_type': resolved_logic,
                'llm': self.llm_provider_name,
                'validation_warnings': validation_warnings
            }

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def generate_stream(
            self,
            k_value: int = 3,
            bitwidth: int = 8,
            language: str = 'c',
            operations: List[str] = None,
            natural_input: str = '',
            module_type: str = 'alu',
            register_count: int = None,
            pipeline_stages: int = None
    ):
        """
        Generate MVL code with streaming output.
        Yields (event_type, data) tuples:
          - ("chunk", text_chunk)
          - ("done", result_dict)
          - ("error", error_message)
        """
        import json

        if operations is None:
            operations = ['ADD', 'SUB', 'MUL', 'NEG', 'INC', 'DEC']

        # When natural_input is provided, parse k_value/bitwidth from it (overrides dropdowns)
        if natural_input:
            parsed = self._parse_natural_input(natural_input)
            if parsed['k_value'] is not None:
                k_value = parsed['k_value']
            if parsed['bitwidth'] is not None:
                bitwidth = parsed['bitwidth']

        logic_info = self._resolve_logic_type(k_value)
        resolved_logic = logic_info['display']
        mod_value = k_value ** bitwidth

        # Create prompt based on module type
        prompt = self._create_prompt_for_module(
            module_type, k_value, bitwidth, mod_value, operations,
            language, logic_info, natural_input,
            register_count=register_count, pipeline_stages=pipeline_stages
        )
        if prompt is None:
            yield ("error", f'Unsupported language: {language}')
            return

        if self.llm is None:
            yield ("error", f'LLM provider "{self.llm_provider_name}" failed to initialize.')
            return

        # Check if provider supports streaming
        has_stream = hasattr(self.llm, '_call_api_stream')
        if not has_stream:
            # Fallback to non-streaming
            result = self.generate(k_value=k_value, bitwidth=bitwidth, language=language,
                                   operations=operations, natural_input=natural_input,
                                   module_type=module_type,
                                   register_count=register_count,
                                   pipeline_stages=pipeline_stages)
            if result.get('success') and result.get('code'):
                yield ("chunk", result['code'])
            yield ("done", result)
            return

        try:
            full_response = ""
            system_prompt = "You are an expert programmer. Generate clean, compilable code without any explanations."

            for chunk in self.llm._call_api_stream(prompt, max_tokens=16384, system_prompt=system_prompt):
                full_response += chunk
                yield ("chunk", chunk)

            print(f"✅ LLM streaming response received ({len(full_response)} chars)")

            # Process the full response
            code = self._extract_code(full_response, language)
            if not code:
                yield ("error", 'Failed to extract code from response')
                return

            code = self._fix_code(code, language)

            # Auto-enhance test coverage if too low
            code = self._enhance_test_coverage(code, language, k_value, bitwidth, mod_value)

            # Validate generated code quality
            validation_warnings = self._validate_code(code, language, k_value, bitwidth, mod_value)

            file_path = self._save_code(code, k_value, bitwidth, language,
                                        natural_input=natural_input, module_type=module_type)

            print(f"✅ Code saved: {file_path}")

            yield ("done", {
                'success': True,
                'code': code,
                'file_path': str(file_path),
                'filename': file_path.name,
                'k_value': k_value,
                'bitwidth': bitwidth,
                'language': language,
                'mod_value': mod_value,
                'module_type': module_type,
                'logic_type': resolved_logic,
                'llm': self.llm_provider_name,
                'validation_warnings': validation_warnings
            })

        except Exception as e:
            print(f"❌ Streaming generation failed: {e}")
            import traceback
            traceback.print_exc()
            yield ("error", str(e))

    def _create_prompt_for_module(self, module_type: str, k: int, bits: int, mod: int,
                                    operations: List[str], language: str, logic_info: Dict,
                                    natural_input: str = '',
                                    register_count: int = None,
                                    pipeline_stages: int = None) -> Optional[str]:
        """Dispatch prompt creation based on module type.

        Returns the prompt string, or None if the language is unsupported.
        """
        # Natural input always takes priority regardless of module type
        if natural_input:
            return self._create_natural_prompt(natural_input, k, bits, mod, operations, language, logic_info)

        # Module-specific prompt dispatch
        if module_type == 'register':
            return self._create_register_prompt(k, bits, mod, operations, language, logic_info,
                                                register_count=register_count or 8)
        elif module_type == 'cpu-risc-v':
            return self._create_cpu_prompt(k, bits, mod, operations, language, logic_info,
                                           pipeline_stages=pipeline_stages or 5)
        elif module_type == 'counter':
            return self._create_counter_prompt(k, bits, mod, operations, language, logic_info)
        else:
            # Default: ALU prompts (existing behavior)
            lang = language.lower()
            if lang == 'c':
                return self._create_c_prompt(k, bits, mod, operations, logic_info)
            elif lang == 'python':
                return self._create_python_prompt(k, bits, mod, operations, logic_info)
            elif lang == 'verilog':
                return self._create_verilog_prompt(k, bits, mod, operations, logic_info)
            elif lang == 'vhdl':
                return self._create_vhdl_prompt(k, bits, mod, operations, logic_info)
            else:
                return None

    def _create_counter_prompt(self, k: int, bits: int, mod: int,
                                operations: List[str], language: str,
                                logic_info: Dict = None) -> str:
        """Create prompt for MVL Counter module generation."""
        import math
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)
        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, language)

        lang = language.lower()
        if lang == 'c':
            lang_upper = 'C'
            compile_note = 'Must be complete and compilable with gcc.'
        elif lang == 'python':
            lang_upper = 'Python'
            compile_note = 'Must be complete and runnable with python3.'
        elif lang == 'verilog':
            lang_upper = 'Verilog'
            compile_note = 'Must be synthesizable and simulatable with iverilog.'
        elif lang == 'vhdl':
            lang_upper = 'VHDL'
            compile_note = 'Must be synthesizable and simulatable with GHDL.'
        else:
            lang_upper = language.upper()
            compile_note = ''

        prompt = f"""Generate a complete {lang_upper} implementation of a {bits}-trit MVL counter operating in base-{k}.
{compile_note}

CRITICAL RULES:
1. Output ONLY {lang_upper} code, no markdown, no explanations
2. Must include AT LEAST 20 test vectors

SPECIFICATIONS:
- K-value: {k} (base-{k} system, each digit has {k} possible values: 0 to {k-1})
- Trit width: {bits} (number of base-{k} digits per counter value)
- MOD value: {mod} ({k}^{bits}) — counter wraps around at this value
- Binary bit width needed: {data_width} bits (ceil(log2({mod})))
- Counter range: 0 to {mod - 1}
- Maximum value: {mod - 1} (NOT {2**data_width - 1} — do NOT rely on register bit-width overflow!)

{algebra_section}

MODULE NAMING:
- Module name MUST be: mvl_counter_{k}_{bits}bit
- Testbench name MUST be: mvl_counter_{k}_{bits}bit_tb

COUNTER OPERATIONS (2-bit opcode):
- 00 = COUNT_UP: count = (count + 1) % {mod}. Sets overflow when count was {mod - 1}.
- 01 = COUNT_DOWN: count = (count - 1 + {mod}) % {mod}. Sets underflow when count was 0.
- 10 = RESET: count = 0
- 11 = LOAD: count = load_value (must be in range 0 to {mod - 1})

FLAG RULES:
- Overflow flag: set to 1 ONLY when COUNT_UP wraps (count was {mod - 1}), else 0
- Underflow flag: set to 1 ONLY when COUNT_DOWN wraps (count was 0), else 0
- Zero flag: set to 1 when the NEW counter value equals 0, else 0
  ⚠️ The zero flag must reflect the RESULT, not be hardcoded!
  Example: COUNT_DOWN from 1 → result is 0 → zero MUST be 1
  Example: COUNT_UP from {mod - 1} → result is 0 (overflow) → zero MUST be 1
  Example: LOAD(0) → zero MUST be 1

⚠️ CRITICAL — DO NOT RELY ON REGISTER OVERFLOW:
The counter MUST explicitly compare against {mod - 1} (not {2**data_width - 1}) and wrap using modulo {mod}.
{"This matters because MOD=" + str(mod) + " ≠ 2^" + str(data_width) + "=" + str(2**data_width) + ". Using register overflow would give WRONG results!" if mod != 2**data_width else "For this specific case MOD=" + str(mod) + " = 2^" + str(data_width) + ", but still use explicit MOD constant for clarity."}

REQUIRED STRUCTURE:
- Define MOD constant = {mod}
- Counter register: {data_width} bits wide, holds values 0 to {mod - 1}
- Synchronous logic on clock edge, asynchronous reset
- Test section with AT LEAST 20 test vectors, each with $display/printf/print showing:
  operation, count value, overflow, underflow, zero flags
- Test cases MUST include:
  - COUNT_UP from 0 (result=1, zero=0)
  - COUNT_UP from {mod - 2} (result={mod - 1}, zero=0)
  - COUNT_UP from {mod - 1} (result=0, overflow=1, zero=1)
  - COUNT_DOWN from {mod - 1} (result={mod - 2}, zero=0)
  - COUNT_DOWN from 1 (result=0, zero=1)
  - COUNT_DOWN from 0 (result={mod - 1}, underflow=1, zero=0)
  - RESET (result=0, zero=1)
  - LOAD with values: 0, 1, {mod // 2}, {mod - 1}

Generate the complete {lang_upper} code now:
"""
        return prompt

    def _create_register_prompt(self, k: int, bits: int, mod: int,
                                 operations: List[str], language: str,
                                 logic_info: Dict = None,
                                 register_count: int = 8) -> str:
        """Create prompt for MVL Register File module generation."""
        import math
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
        addr_width = math.ceil(math.log2(register_count)) if register_count > 1 else 1
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)
        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, language)

        lang = language.lower()
        if lang == 'c':
            lang_upper = 'C'
            compile_note = 'Must be complete and compilable with gcc.'
        elif lang == 'python':
            lang_upper = 'Python'
            compile_note = 'Must be complete and runnable with python3.'
        elif lang == 'verilog':
            lang_upper = 'Verilog'
            compile_note = 'Must be synthesizable and simulatable with iverilog.'
        elif lang == 'vhdl':
            lang_upper = 'VHDL'
            compile_note = 'Must be synthesizable and simulatable with GHDL.'
        else:
            lang_upper = language.upper()
            compile_note = ''

        prompt = f"""Generate a complete {lang_upper} implementation of a {bits}-trit MVL register file with {register_count} registers, operating in base-{k}.
{compile_note}

CRITICAL RULES:
1. Output ONLY {lang_upper} code, no markdown, no explanations
2. Must include AT LEAST 20 test vectors
3. The code must compile and simulate WITHOUT errors

SPECIFICATIONS:
- K-value: {k} (base-{k} system)
- Trit width: {bits} (number of base-{k} digits per register)
- MOD value: {mod} ({k}^{bits}) — each register holds values 0 to {mod - 1}
- Binary data width: {data_width} bits (ceil(log2({mod})))
- Register count: {register_count}
- Address width: {addr_width} bits (ceil(log2({register_count})))

{algebra_section}

MODULE NAMING:
- DUT module name MUST be: mvl_regfile_{k}_{bits}bit
- Testbench module name MUST be: mvl_regfile_{k}_{bits}bit_tb

⚠️ ARCHITECTURE — DUT AND TESTBENCH MUST BE SEPARATE MODULES:
The DUT (design under test) and the testbench MUST be two separate modules.
- The DUT module contains ONLY the register file logic (no initial blocks, no $display).
- The testbench module instantiates the DUT, generates the clock, drives all inputs, and reads outputs.
- NEVER put test logic (initial blocks, $display) inside the DUT module.
- NEVER drive input ports from inside the module that declares them — this is ILLEGAL in Verilog/VHDL.

REGISTER FILE DUT INTERFACE:
- clk: clock input
- rst: asynchronous reset input (use "always @(posedge clk or posedge rst)")
- read_addr1 [{addr_width - 1}:0]: read port 1 address
- read_addr2 [{addr_width - 1}:0]: read port 2 address
- read_data1 [{data_width - 1}:0]: read port 1 data output (combinational)
- read_data2 [{data_width - 1}:0]: read port 2 data output (combinational)
- write_enable: write enable (active high)
- write_addr [{addr_width - 1}:0]: write port address
- write_data [{data_width - 1}:0]: write port data input
- {register_count} registers, each {data_width} bits wide (values 0 to {mod - 1})
- Register 0 is hardwired to 0 (writes to register 0 are ignored)
- Reads are combinational (assign), writes are synchronous (posedge clk)
- Reset: all registers set to 0

TESTBENCH REQUIREMENTS:
- Generate clock: initial clk = 0; forever #5 clk = ~clk;
- Apply reset at start: rst=1, wait 2 clock cycles, rst=0
- For EVERY test: set read_addr1/read_addr2 BEFORE reading read_data1/read_data2
  ⚠️ You MUST explicitly assign read_addr1 or read_addr2 to the register you want to read!
  The read address does NOT auto-follow the write address — they are independent ports.
- Wait for posedge clk after setting write signals (use @(posedge clk) or sufficient #delay)
- Use $display for EVERY test showing: test number, operation, address, written value, read value
- At least 20 test vectors covering:
  1. Write to R1, set read_addr1=1, verify read_data1 (basic write-read)
  2. Write to R2, set read_addr2=2, verify read_data2 (second read port)
  3. Write to R0, set read_addr1=0, verify read_data1=0 (R0 always 0)
  4. Read R1 and R2 simultaneously (both read ports at once)
  5. Write with write_enable=0, verify data NOT written
  6. Write edge values: 0, 1, {mod - 1}
  7. Reset test: write values, assert rst, verify all registers = 0
  8. Write to multiple registers, read them all back

Generate the complete {lang_upper} code now:
"""
        return prompt

    def _create_cpu_prompt(self, k: int, bits: int, mod: int,
                            operations: List[str], language: str,
                            logic_info: Dict = None,
                            pipeline_stages: int = 5) -> str:
        """Create prompt for MVL CPU (RISC-V style) module generation."""
        import math
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)
        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, language)

        lang = language.lower()
        if lang == 'c':
            lang_upper = 'C'
            compile_note = 'Must be complete and compilable with gcc.'
        elif lang == 'python':
            lang_upper = 'Python'
            compile_note = 'Must be complete and runnable with python3.'
        elif lang == 'verilog':
            lang_upper = 'Verilog'
            compile_note = 'Must be synthesizable and simulatable with iverilog.'
        elif lang == 'vhdl':
            lang_upper = 'VHDL'
            compile_note = 'Must be synthesizable and simulatable with GHDL.'
        else:
            lang_upper = language.upper()
            compile_note = ''

        # Pipeline stage descriptions
        if pipeline_stages == 3:
            pipeline_desc = "3-stage pipeline: IF (Instruction Fetch), EX (Execute), WB (Write Back)"
            stage_names = "IF, EX, WB"
        elif pipeline_stages == 7:
            pipeline_desc = "7-stage pipeline: IF (Instruction Fetch), ID (Instruction Decode), IS (Issue), EX (Execute), MEM (Memory Access), WB (Write Back), CM (Commit)"
            stage_names = "IF, ID, IS, EX, MEM, WB, CM"
        else:
            pipeline_desc = "5-stage pipeline: IF (Instruction Fetch), ID (Instruction Decode), EX (Execute), MEM (Memory Access), WB (Write Back)"
            stage_names = "IF, ID, EX, MEM, WB"

        prompt = f"""Generate a complete {lang_upper} implementation of a simplified {bits}-trit MVL RISC-V-style CPU with a {pipeline_stages}-stage pipeline, operating in base-{k}.
{compile_note}

CRITICAL RULES:
1. Output ONLY {lang_upper} code, no markdown, no explanations
2. Must include AT LEAST 20 test instructions in the test program
3. The code must compile and simulate WITHOUT errors

SPECIFICATIONS:
- K-value: {k} (base-{k} system)
- Trit width: {bits} (number of base-{k} digits per data word)
- MOD value: {mod} ({k}^{bits}) — all arithmetic wraps at this value
- Binary data width: {data_width} bits (ceil(log2({mod})))
- Data range: 0 to {mod - 1}
- Pipeline: {pipeline_desc}

{algebra_section}

MODULE NAMING:
- CPU module: mvl_cpu_{k}_{bits}bit
- Testbench module: mvl_cpu_{k}_{bits}bit_tb

⚠️ DUT AND TESTBENCH MUST BE SEPARATE MODULES:
- The CPU module has ONLY clk and rst as inputs. It fetches instructions from internal instruction memory.
- The testbench module instantiates the CPU, generates clock, drives reset, and monitors outputs.
- NEVER put test stimulus (initial blocks that drive signals) inside the CPU module.
- For Verilog: use "integer i;" declared before always blocks. Do NOT use "for (int i = ...)" — that is SystemVerilog, not Verilog.

CPU ARCHITECTURE:
- {pipeline_stages}-stage pipeline ({stage_names})
- 8 general-purpose registers (R0-R7), each {data_width} bits wide
  - ⚠️ R0 is HARDWIRED to 0: every write to R0 must be IGNORED, reads from R0 always return 0
- Program counter (PC): {data_width} bits, starts at 0, increments by 1 each cycle
- Instruction memory: array of instruction records (at least 32 slots), pre-loaded with test program
- Data memory: array of {data_width}-bit words (at least 16 slots)

INSTRUCTION SET (all arithmetic mod {mod}):
- Opcode 0 = ADD   rd, rs1, rs2  : R[rd] = (R[rs1] + R[rs2]) % {mod}
- Opcode 1 = SUB   rd, rs1, rs2  : R[rd] = (R[rs1] - R[rs2] + {mod}) % {mod}
- Opcode 2 = MUL   rd, rs1, rs2  : R[rd] = (R[rs1] * R[rs2]) % {mod}
- Opcode 3 = ADDI  rd, rs1, imm  : R[rd] = (R[rs1] + imm) % {mod}
- Opcode 4 = LOAD  rd, imm(rs1)  : R[rd] = DataMem[(R[rs1] + imm) % 16]
- Opcode 5 = STORE rs2, imm(rs1) : DataMem[(R[rs1] + imm) % 16] = R[rs2]  (does NOT write register file)
- Opcode 6 = BEQ   rs1, rs2, off : if R[rs1] == R[rs2] then PC = PC + off, else PC = PC + 1
- Opcode 7 = NOP                 : no operation (pipeline bubble)

INSTRUCTION ENCODING:
- Each instruction is a record/struct with fields: opcode (3 bits), rd (3 bits), rs1 (3 bits), rs2 (3 bits), imm ({data_width} bits)
- Instructions are stored in an instruction memory array, indexed by PC

⚠️ PIPELINE CRITICAL RULES:
1. INSTRUCTION FETCH (IF): Read instruction from instruction_mem[PC]. Latch into IF/ID pipeline register.
2. Each pipeline stage passes its results to the NEXT stage via pipeline registers.
   Pipeline registers store: opcode, rd, rs1_value, rs2_value, imm, alu_result, etc.
3. WB stage writes ONLY the result computed in EX stage (stored in pipeline register).
   ⚠️ WB must write "result_wb" (the value stored in the pipeline register), NOT re-read reg_file[rs1_ex].
   Re-reading the register file in WB is WRONG because the register may have been overwritten by a later instruction.
4. R0 protection: if rd == 0, do NOT write to register file (skip the write).
5. STORE does NOT write to the register file — it only writes to data memory.
6. BEQ does NOT write to the register file — it only affects PC.
7. NOP does nothing — no register write, no memory write.
8. ALL combinational always blocks must assign outputs for ALL opcode cases (use default/else to prevent latches).
9. Insert NOP between dependent instructions to avoid data hazards (simplest approach).

TEST PROGRAM:
Pre-load instruction memory with at least 20 instructions. Include NOP between dependent instructions.
Example sequence:
  ADDI R1, R0, 10    // R1 = 10
  NOP                // avoid hazard
  ADDI R2, R0, 20    // R2 = 20
  NOP
  ADD  R3, R1, R2    // R3 = (10+20)%{mod} = 30
  NOP
  SUB  R4, R2, R1    // R4 = (20-10+{mod})%{mod} = 10
  NOP
  MUL  R5, R1, R2    // R5 = (10*20)%{mod} = 200
  NOP
  STORE R1, 0(R0)    // DataMem[0] = R1 = 10
  NOP
  LOAD R6, 0(R0)     // R6 = DataMem[0] = 10
  NOP
  BEQ R1, R6, +2     // R1==R6, should branch (skip next)
  ADDI R7, R0, 999   // skipped by branch
  NOP
  ... (continue to reach 20+ instructions)

TESTBENCH:
- Generate clock: initial clk = 0; forever #5 clk = ~clk;
- Apply reset: rst=1 for 2 cycles, then rst=0
- Run for enough cycles to execute all instructions (e.g., #500)
- After execution, use $display to print ALL registers R0-R7 and DataMem[0..15]
- Use $display to show expected vs actual values for key registers

Generate the complete {lang_upper} code now:
"""
        return prompt

    def _create_natural_prompt(self, natural_input: str, k: int, bits: int, mod: int,
                                 operations: List[str], language: str, logic_info: Dict = None) -> str:
        """Create prompt based on natural language input, supplemented with structured parameters."""
        import math
        ops_str = ', '.join(operations)
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)

        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, language)

        # Determine C intermediate type for MUL
        mul_product_max = (mod - 1) * (mod - 1)
        if mul_product_max <= 2**64 - 1:
            c_mul_type = "uint64_t"
        else:
            c_mul_type = "__uint128_t"

        # Pre-compute VHDL-safe literals for large values
        mod_val_lit = self._vhdl_unsigned_literal(mod, data_width + 1)

        lang_instructions = {
            'c': f"""Output ONLY C code. Must be complete and compilable with gcc -lm.
REQUIRED STRUCTURE:
- #include <stdio.h>, <stdlib.h>, <stdint.h>, <stdbool.h>, <time.h>, <math.h>
- For MUL: max product = {mod-1} * {mod-1} = {mul_product_max}. Use {c_mul_type} for the intermediate.
- #define K_VALUE {k}
- #define TRIT_WIDTH {bits}
- #define MOD_VALUE {mod}
- Operation enum: OP_ADD=0, OP_SUB=1, OP_MUL=2, OP_NEG=3, OP_INC=4, OP_DEC=5
- Result struct containing BOTH the uint64_t result value AND Flags (z, n, c)
- alu_exec(a, b, op) function that RETURNS a struct with BOTH result AND flags
  ⚠️ The function MUST return the computed result — do NOT discard it as a local variable!
- ⚠️ NEG MUST use (MOD - a) % MOD, NOT just (MOD - a). When a=0, (MOD-0)=MOD which is out of range!
- main() with srand(time(NULL)) and at least 20 random test vectors
- Use the result from the returned struct in printf, NOT a separately computed value
- printf format: "Test %2d: %-3s A=%llu B=%llu -> R=%llu Z=%d N=%d C=%d\\n" """,

            'python': f"""Output ONLY Python code. Must be complete and runnable with python3.
REQUIRED STRUCTURE:
- import random
- MOD = {mod}
- class or dataclass for Flags(z, n, c)
- alu_exec(a, b, op) function returning (result, flags) tuple
- if __name__ == "__main__": block with at least 20 random test vectors
COMMON MISTAKES TO AVOID:
- INC adds exactly 1, not '{k}' or '{k}^{bits-1}' or any other value
- DEC subtracts exactly 1, not '{k}' or '{k}^{bits-1}' or any other value""",

            'verilog': f"""Output ONLY Verilog code (NOT VHDL). Must be synthesizable and simulatable with iverilog.
CRITICAL BIT WIDTH: Each operand needs {data_width} bits to represent values 0 to {mod-1}.
  Formula: ceil(log2({mod})) = {data_width} bits. Do NOT use {bits} bits — that is the trit count, not the bit count!
MODULE INTERFACE (use this EXACT interface):
module mvl_alu_{k}_{bits}bit (
    input wire clk,
    input wire rst,
    input wire [{data_width-1}:0] a,
    input wire [{data_width-1}:0] b,
    input wire [3:0] opcode,
    output reg [{data_width-1}:0] result,
    output reg zero,
    output reg negative,
    output reg carry
);
ARCHITECTURE REQUIREMENTS:
- Use ONLY ONE always block — do NOT split into combinational + sequential
- ALL reg declarations MUST be at module level, NOT inside always blocks or case statements
- For MUL: declare temp as reg [{data_width * 2 - 1}:0] at MODULE LEVEL
- Declare temp_result and temp_carry as reg at module level for intermediate calculations
- Use BLOCKING assignment (=) for temp_result/temp_carry inside always block,
  then NON-BLOCKING (<=) for ALL output ports (result, zero, negative, carry).
  Do NOT mix: carry = ... (blocking) alongside result <= ... (non-blocking).
- MUL carry = 0, NEG carry = 0 (these operations do not generate carry)
TESTBENCH: Include a testbench module mvl_alu_{k}_{bits}bit_tb with at least 20 test vectors using $display.""",

            'vhdl': f"""Output ONLY VHDL code (NOT Verilog). Must be synthesizable and simulatable with GHDL.
CRITICAL: Do NOT output Verilog syntax (module, wire, reg, always). Use ONLY VHDL syntax (library, entity, architecture, signal, process).
CRITICAL BIT WIDTH: Each operand needs {data_width} bits to represent values 0 to {mod-1}.
  Formula: ceil(log2({mod})) = {data_width} bits. Do NOT use {bits} bits — that is the trit count, not the bit count!
ENTITY INTERFACE (use this EXACT interface):
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
entity mvl_alu_{k}_{bits}bit is
    port (
        clk     : in  std_logic;
        rst     : in  std_logic;
        a       : in  std_logic_vector({data_width-1} downto 0);
        b       : in  std_logic_vector({data_width-1} downto 0);
        opcode  : in  std_logic_vector(3 downto 0);
        result  : out std_logic_vector({data_width-1} downto 0);
        zero    : out std_logic;
        negative: out std_logic;
        carry   : out std_logic
    );
end entity;
VHDL REQUIREMENTS:
{"⚠️ MOD value " + str(mod) + " EXCEEDS VHDL integer range (max 2^31-1)! MUST use hex literal constant:" + chr(10) + "   constant MOD_VAL : unsigned(" + str(data_width) + " downto 0) := " + mod_val_lit + ";" + chr(10) + "   Do NOT use to_unsigned() or integer type for MOD!" if mod > 2147483647 else "- MOD constant MUST be set to exactly " + str(mod) + " — do NOT use (others => '1') or any other value"}
- Use unsigned type for ALL arithmetic, convert with unsigned() and std_logic_vector()
- Use variables (not signals) inside process for intermediate computations
- ⚠️ ALL variable declarations MUST be in the process declarative region (between "process" and "begin").
  Do NOT declare variables inside case/when branches — this is ILLEGAL in VHDL and will NOT compile!
- ⚠️ Do NOT use "when...else" for ANY variable assignment inside a process (VHDL-93 does not support it).
  This applies to carry, zero, AND negative flags too!
  WRONG: v_carry := '1' when cond else '0';  WRONG: v_zero := '1' when v_result = 0 else '0';
  CORRECT: if cond then v_carry := '1'; else v_carry := '0'; end if;
- NEVER compare unsigned with < 0 — unsigned is ALWAYS >= 0, the comparison is always false!
  For SUB: compare operands FIRST (if unsigned(a) < unsigned(b) then ...) before subtraction.
  For DEC(0): check if a = 0 FIRST, then assign MOD-1 directly instead of subtracting.
- ⚠️ Do NOT use (others => '0') in comparisons — VHDL cannot infer its length. Use "= 0" instead.
- ⚠️ MOD_VAL is {data_width+1} bits, v_result is {data_width} bits. Always use resize() when assigning MOD_VAL expressions to v_result.
- ADD/SUB: resize operands to {data_width+1} bits BEFORE adding (to detect carry)
- ⚠️ MUL: use unsigned(a) * unsigned(b) DIRECTLY — do NOT resize before multiplying!
  VHDL "*" returns length = left'length + right'length, so {data_width}*{data_width} = {data_width*2} bits automatically.
  resize before multiply would give {data_width*4} bits causing a length mismatch error.
- MUL carry = '0'. NEG carry = '0'.
- NEG(0) must equal 0, NOT {mod}
- Include BOTH entity/architecture AND a testbench entity mvl_alu_{k}_{bits}bit_tb
- Testbench must have at least 20 test vectors with assert/report statements
- ⚠️ Each test MUST explicitly assign BOTH a AND b — even if same values as previous test.
  Changing only the opcode uses stale a/b and causes wrong results.
- Testbench expected values (verified — use these EXACT values, do NOT recompute):
  ADD({mod-1},{mod-1})={2*(mod-1)%mod}, NEG(0)=0, NEG({mod-1})=1, NEG(10)={(mod-10)%mod}, NEG(1)={(mod-1)%mod}
  ⚠️ NEG(10) = {mod} - 10 = {(mod-10)%mod}. Do NOT miscalculate this!"""
        }

        prompt = f"""Based on the following description, generate a complete implementation:

DESCRIPTION: {natural_input}

CONTEXT PARAMETERS (these are MANDATORY constraints):
- K-value: {k} (base-{k} system)
- Trit width: {bits} (number of base-{k} digits per operand)
- MOD value: {mod} ({k}^{bits})
- Binary bit width needed: {data_width} bits (ceil(log2({mod})))
- Operand range: 0 to {mod - 1}
- Operations: {ops_str}
- Output language: {language.upper()}

{algebra_section}

LANGUAGE-SPECIFIC INSTRUCTIONS:
{lang_instructions.get(language.lower(), lang_instructions['c'])}

CRITICAL RULES:
1. No markdown formatting, no explanations — output ONLY {language.upper()} code
2. The MOD value {mod} and bit width {data_width} are mathematically derived and MUST be used exactly
3. All 6 operations must be implemented

⚠️ MANDATORY TEST REQUIREMENT (DO NOT SKIP):
You MUST include AT LEAST 20 test vectors in the test/main section. This is a HARD REQUIREMENT — code with fewer than 20 tests will be REJECTED.
Test vectors must cover ALL 6 operations with these categories:
- Edge cases: a=0, b=0 (at least 2 tests)
- Max value: a={mod-1}, b={mod-1} (at least 2 tests)
- Overflow/underflow: values that wrap around mod {mod} (at least 4 tests)
- Small values: a=1, b=1 (at least 2 tests)
- Random values: random a, b for each operation (at least 10 tests)

Generate the complete {language.upper()} code now:
"""
        return prompt

    def _create_c_prompt(self, k: int, bits: int, mod: int, operations: List[str], logic_info: Dict = None) -> str:
        """Create prompt for C code generation"""
        import math
        ops_str = ', '.join(operations)
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)

        mul_product_max = (mod - 1) * (mod - 1)
        if mul_product_max <= 2**64 - 1:
            c_mul_type = "uint64_t"
        else:
            c_mul_type = "__uint128_t"

        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, 'c')
        is_extension = (logic_info['category'] == 'extension_field' and logic_info.get('tables'))

        if is_extension:
            # Extension field: digit-wise operations, no standard mod arithmetic
            arith_instructions = f"""3. GF({k}) addition and multiplication lookup tables (copy from ALGEBRAIC STRUCTURE section)
4. Result struct containing BOTH uint64_t result AND Flags (z, n, c)
   ⚠️ alu_exec() MUST return a struct with both the result value AND the flags!
   Do NOT return only flags while discarding the result as a local variable.
5. Helper functions: to_digits(), from_digits(), gf_add_digits(), gf_mul_digits()
6. ⚠️ ALL operations MUST use digit-wise GF({k}) table lookups as described in ALGEBRAIC STRUCTURE.
   Do NOT use standard modular arithmetic ((a+b)%MOD, (MOD-a)%MOD, etc.) — these give WRONG results!"""
        else:
            # Prime field / integer ring: standard mod arithmetic
            arith_instructions = f"""3. #define MOD {mod}ULL
4. Result struct containing BOTH uint64_t result AND Flags (z, n, c)
   ⚠️ alu_exec() MUST return a struct with both the result value AND the flags!
   Do NOT return only flags while discarding the result as a local variable.
5. For MUL: use {c_mul_type} for intermediate product ({mod-1} * {mod-1} = {mul_product_max})
6. ⚠️ NEG: MUST use (MOD - a) % MOD, NOT just (MOD - a). When a=0, (MOD-0)=MOD which is NOT in range!
   NEG(0) must equal 0. The % MOD is mandatory."""

        prompt = f"""Generate a complete, compilable C program for a {bits}-trit ALU operating in base-{k}.

CRITICAL RULES:
1. Output ONLY C code, no markdown, no explanations
2. Must be complete and compilable with gcc
3. main() MUST contain AT LEAST 20 test vectors — this is a HARD REQUIREMENT

SPECIFICATIONS:
- K-value: {k}
- Bitwidth: {bits} trits
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}
- Each operand range: 0 to {mod - 1}

{algebra_section}

REQUIRED STRUCTURE:
1. #include statements (stdio.h, stdlib.h, stdint.h, stdbool.h, time.h)
2. Operation enum (OP_ADD=0, OP_SUB=1, OP_MUL=2, OP_NEG=3, OP_INC=4, OP_DEC=5)
{arith_instructions}
7. alu_exec(a, b, op) function with switch-case
8. op_name() function to get operation name string
9. main() with srand() and AT LEAST 20 printf test lines — use the result FROM alu_exec()

⚠️ MANDATORY TEST REQUIREMENT:
main() MUST include AT LEAST 20 test vectors using printf. Code with fewer than 20 tests will be REJECTED.
Include:
- 6 edge-case tests (one per operation with a=0, b=0)
- 6 max-value tests (one per operation with a={mod-1}, b={mod-1})
- 8+ random tests with srand(time(NULL)) and rand() % {mod}

EXAMPLE OUTPUT FORMAT IN main():
printf("Test %2d: %-3s A=%llu B=%llu -> R=%llu Z=%d N=%d C=%d\\n", ...);

Generate the complete C code now:
"""
        return prompt

    def _create_python_prompt(self, k: int, bits: int, mod: int, operations: List[str], logic_info: Dict = None) -> str:
        """Create prompt for Python code generation"""
        ops_str = ', '.join(operations)
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)

        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, 'python')
        is_extension = (logic_info['category'] == 'extension_field' and logic_info.get('tables'))

        if is_extension:
            structure_section = f"""REQUIRED STRUCTURE:
1. imports (random, dataclasses or namedtuple)
2. GF({k}) addition and multiplication lookup tables (copy from ALGEBRAIC STRUCTURE section)
3. Operation constants (OP_ADD=0, OP_SUB=1, etc.)
4. Flags dataclass or namedtuple (z, n, c)
5. Helper functions: to_digits(), from_digits(), gf_add_digits(), gf_mul_digits()
6. alu_exec(a, b, op) function returning (result, flags)
   ⚠️ ALL operations MUST use digit-wise GF({k}) table lookups as described in ALGEBRAIC STRUCTURE.
   Do NOT use standard modular arithmetic ((a+b)%MOD, (MOD-a)%MOD, etc.) — these give WRONG results!
   ⚠️ SUB: In GF({k}), subtraction a-b means finding x where GF_ADD[x][b_digit] == a_digit for each digit.
   Do NOT compute SUB as a + (b+b) — in characteristic {logic_info['p']}, b+b=0, so that returns a, not a-b!
7. op_name(op) function
8. if __name__ == "__main__": block with AT LEAST 20 print test lines"""
        else:
            structure_section = f"""REQUIRED STRUCTURE:
1. imports (random, dataclasses or namedtuple)
2. MOD = {mod}
3. Operation constants (OP_ADD=0, OP_SUB=1, etc.)
4. Flags dataclass or namedtuple (z, n, c)
5. alu_exec(a, b, op) function returning (result, flags)
6. op_name(op) function
7. if __name__ == "__main__": block with AT LEAST 20 print test lines"""

        prompt = f"""Generate a complete Python program for a {bits}-trit ALU operating in base-{k}.

CRITICAL RULES:
1. Output ONLY Python code, no markdown, no explanations
2. Must be complete and runnable with python3
3. if __name__ == "__main__" block MUST contain AT LEAST 20 test vectors — HARD REQUIREMENT

SPECIFICATIONS:
- K-value: {k}
- Bitwidth: {bits} trits
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}
- Each operand range: 0 to {mod - 1}

{algebra_section}

{structure_section}

⚠️ MANDATORY TEST REQUIREMENT:
The main block MUST include AT LEAST 20 test vectors using print(). Code with fewer than 20 tests will be REJECTED.
Include:
- 6 edge-case tests (one per operation with a=0, b=0)
- 6 max-value tests (one per operation with a={mod-1}, b={mod-1})
- 8+ random tests with random.randint(0, {mod-1})
Use this EXACT print format for every test:
  print(f"Test {{i}}: {{op_name(op)}} a={{a}} b={{b}} result={{result}} z={{flags.z}} n={{flags.n}} c={{flags.c}}")

Generate the complete Python code now:
"""
        return prompt

    def _create_verilog_prompt(self, k: int, bits: int, mod: int, operations: List[str], logic_info: Dict = None) -> str:
        """Create prompt for Verilog code generation"""
        ops_str = ', '.join(operations)
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)

        # Calculate bit width needed to represent mod value
        import math
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
        bits_per_digit = math.ceil(math.log2(k)) if k > 1 else 1

        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, 'verilog')
        is_extension = (logic_info['category'] == 'extension_field' and logic_info.get('tables'))

        if is_extension:
            arch_section = f"""ARCHITECTURE REQUIREMENTS:
⚠️ This is GF({k}) digit-wise arithmetic — NOT standard modular arithmetic!
  Do NOT use (a + b) % {mod} or (a * b) % {mod}. These give WRONG results for extension fields.

- Use ONLY ONE always block — do NOT split outputs into combinational + sequential blocks!

⚠️ CRITICAL VERILOG SYNTAX RULES (violations cause compilation failure):
- ALL reg declarations MUST be at module level. Do NOT declare reg inside always blocks or case statements!
  WRONG:  always @(*) begin reg [1:0] temp; ...     ← ILLEGAL!
  CORRECT: reg [1:0] temp;  // at module level, before always block
- Do NOT declare or initialize variables inside always blocks:
  WRONG:  reg [{data_width-1}:0] temp_result = 0;   ← ILLEGAL inside always!
  CORRECT: reg [{data_width-1}:0] temp_result;       // at module level
           always @(*) begin temp_result = 0; ...   // assign inside always
- `integer i, j;` MUST be declared at module level, NOT inside always blocks.
- Do NOT use `break` or `continue` — these are NOT valid in Verilog!
  If you need to exit a loop early, use a flag variable or restructure the logic.
  WRONG:  for (j=0; j<4; j=j+1) begin if (cond) break; end    ← ILLEGAL!
  CORRECT: for (j=0; j<4; j=j+1) begin if (!found) begin ... found=1; end end

- Each {data_width}-bit operand encodes {bits} digits of {bits_per_digit} bits each.
  Extract digits using indexed part-select: a_digits[i] = a[i*{bits_per_digit} +: {bits_per_digit}]
  ⚠️ Do NOT use a[(i*{bits_per_digit})+{bits_per_digit-1}: i*{bits_per_digit}] — variable part-select is ILLEGAL in Verilog!
  ⚠️ Use a[i*{bits_per_digit} +: {bits_per_digit}] (indexed part-select) instead.
- Implement GF_ADD and GF_MUL as Verilog functions with [{bits_per_digit-1}:0] inputs (NOT [3:0]!)
  Use case({{a, b}}) with {bits_per_digit*2}-bit patterns matching the tables from ALGEBRAIC STRUCTURE.
- Compute results into temp regs using BLOCKING assignment (=), then assign outputs with NON-BLOCKING (<=).
- Carry and negative flags are always 0 for GF field arithmetic.
- Zero flag: check if temp_result == 0."""
        else:
            arch_section = f"""ARCHITECTURE REQUIREMENTS:
⚠️ Use ONLY ONE always block — do NOT split outputs into combinational + sequential blocks!
  Having result/zero/negative/carry driven by two always blocks creates a MULTI-DRIVER conflict.
  Use a single always @(posedge clk) block that handles reset and computation.
- ALL reg declarations MUST be at module level. Do NOT declare reg inside always blocks or case statements!
  Example: declare "reg [{data_width * 2 - 1}:0] temp;" at module level, NOT inside a case branch.
- For MUL: declare temp as reg [{data_width * 2 - 1}:0] at module level to hold the full product.
- ⚠️ NON-BLOCKING ASSIGNMENT TIMING: Inside a clocked always block, non-blocking assignments (<=)
  update at the END of the time step. Reading a signal on the RHS after assigning it with <= gives
  the OLD value. To fix this, compute the result into a temporary reg using BLOCKING assignment (=),
  then assign ALL outputs from those temporaries:
    reg [{data_width - 1}:0] temp_result;
    reg temp_carry;
    always @(posedge clk) begin
      temp_result = (a + b) % {mod};  // BLOCKING: immediate update
      temp_carry = (a + b) >= {mod};  // BLOCKING: immediate update
      result <= temp_result;           // NON-BLOCKING: output port
      zero <= (temp_result == 0);      // reads NEW value correctly
      negative <= (temp_result >= {mod // 2});
      carry <= temp_carry;             // NON-BLOCKING: output port
    end
  ⚠️ ALL output ports (result, zero, negative, carry) MUST use non-blocking (<=).
  Use blocking (=) ONLY for temp_result/temp_carry intermediate calculations.
  Do NOT mix: carry = ... (blocking) alongside result <= ... (non-blocking).
- ⚠️ MUL CARRY: MUL does NOT generate carry. Set carry = 0 for MUL opcode.
  NEG also does NOT generate carry. Set carry = 0 for NEG opcode.
- Use 'output reg' for result/flags since they are assigned in always block"""

        prompt = f"""Generate a complete Verilog module for a {bits}-trit ALU operating in base-{k}.

CRITICAL RULES:
1. Output ONLY Verilog code, no markdown, no explanations
2. Must be synthesizable and simulatable with iverilog

SPECIFICATIONS:
- K-value: {k}
- Bitwidth: {bits} trits (need {data_width} bits to represent 0 to {mod - 1})
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}

{algebra_section}

MODULE INTERFACE:
module mvl_alu_{k}_{bits}bit (
    input wire clk,
    input wire rst,
    input wire [{data_width - 1}:0] a,
    input wire [{data_width - 1}:0] b,
    input wire [3:0] opcode,
    output reg [{data_width - 1}:0] result,
    output reg zero,
    output reg negative,
    output reg carry
);

{arch_section}

⚠️ MANDATORY TESTBENCH REQUIREMENT:
You MUST include a testbench module mvl_alu_{k}_{bits}bit_tb with AT LEAST 20 $display test vectors. Code with fewer than 20 tests will be REJECTED.
Include:
- 6 edge-case tests (one per opcode with a=0, b=0)
- 6 max-value tests (one per opcode with a={mod-1}, b={mod-1})
- 8+ additional tests with various values
Each test: assign a, b, opcode → #10 → $display result
Use this EXACT $display format for every test:
  $display("Test %0d: OP=%0d a=%0d b=%0d result=%0d", test_num, opcode, a, b, result);
End the testbench with $finish.

Generate the complete Verilog module with testbench now:
"""
        return prompt

    @staticmethod
    def _build_vhdl_arch_section(k, bits, mod, data_width, is_extension, logic_info, mod_val_lit, neg_half_lit, max_val):
        """Build the VHDL architecture requirements section, conditional on algebraic structure."""
        import math
        bits_per_digit = math.ceil(math.log2(k)) if k > 1 else 1

        common_rules = f"""⚠️ VHDL COMMENT SYNTAX: Comments MUST use "--", NOT "!" or "//" or "#".
   WRONG: !this is a comment    WRONG: //this is a comment
   CORRECT: -- this is a comment

⚠️ VHDL FUNCTION PARAMETER NAMING (CRITICAL — causes GHDL compilation failure):
   Function parameters MUST NOT use the same names as entity ports!
   Entity has ports: a, b, result, zero, negative, carry.
   WRONG:  function gf_add(a : unsigned; b : unsigned) ...     ← "a" hides port "a"!
   CORRECT: function gf_add(x_val : unsigned; y_val : unsigned) ...  ← unique names

⚠️ VHDL VARIABLE DECLARATION RULES (CRITICAL — violations cause compilation failure):
1. ALL variable declarations MUST be in the process DECLARATIVE REGION (between "process" and "begin").
   Do NOT declare variables inside case/when branches or if/else blocks — this is ILLEGAL in VHDL!
2. Do NOT use "when...else" syntax for variable assignment inside a process — ANYWHERE!
   It is only valid in VHDL-2008 and GHDL defaults to VHDL-93.
   WRONG:  v_carry := '1' when condition else '0';
   CORRECT: if condition then v_carry := '1'; else v_carry := '0'; end if;
- Use variables (NOT signals) inside process for intermediate results
⚠️ CASE STATEMENT RULE (CRITICAL — missing this causes GHDL compilation failure):
   EVERY case statement MUST have "when others =>" as the LAST choice!
   to_integer() returns 'natural' (0 to 2147483647), so case statements must
   cover all values. Without 'when others', GHDL reports:
     "error: no choices for N to 2147483647"
   WRONG:  case to_integer(x) is when 0 => ... when 1 => ... end case;
   CORRECT: case to_integer(x) is when 0 => ... when 1 => ... when others => return to_unsigned(0, 2); end case;"""

        if is_extension:
            p = logic_info['p']
            # Build example literals outside f-string to avoid syntax issues
            example_lits = ", ".join('"' + format(i, '0' + str(bits_per_digit) + 'b') + '"' for i in range(min(k, 4)))
            example_row = ", ".join('"' + format(i % k, '0' + str(bits_per_digit) + 'b') + '"' for i in range(min(k, 4)))
            hex_warning = ('WRONG: x"0" (hex literal = 4 bits, does NOT match ' + str(bits_per_digit) + '-bit element!)') if bits_per_digit != 4 else ''
            return f"""VHDL ARCHITECTURE REQUIREMENTS:
⚠️ This is GF({k}) = GF({p}^{logic_info['n']}) digit-wise arithmetic — NOT standard modular arithmetic!
  Do NOT use (a + b) mod MOD_VAL or unsigned(a) * unsigned(b). These give WRONG results!
  Each {data_width}-bit operand encodes {bits} digits of {bits_per_digit} bits each.

{common_rules}

⚠️ CRITICAL VHDL SYNTAX RULES FOR GF TABLES:
- GF table elements are unsigned({bits_per_digit-1} downto 0) = {bits_per_digit} bits wide.
  {hex_warning}
  CORRECT: use {bits_per_digit}-bit BINARY string literals, e.g. {example_lits}
  Example table row: ({example_row})
- Function return types MUST be unconstrained: "return unsigned" NOT "return unsigned(N downto 0)"
  WRONG:  function foo(x : unsigned) return unsigned({bits_per_digit-1} downto 0) is
  CORRECT: function foo(x : unsigned({bits_per_digit-1} downto 0); y : unsigned({bits_per_digit-1} downto 0)) return unsigned is
- Do NOT declare custom array types with constrained element types. Instead use individual variables.
  WRONG:  type digit_array is array(0 to {bits-1}) of unsigned({bits_per_digit-1} downto 0);
  CORRECT: use individual variables: variable a_d0 : unsigned({bits_per_digit-1} downto 0); etc.
  OR use unconstrained arrays with proper type definitions.
- Use to_unsigned(N, {bits_per_digit}) for digit constants, NOT hex literals.

IMPLEMENTATION APPROACH:
- Use individual digit variables: a_d0, a_d1, ..., a_d{bits-1} of unsigned({bits_per_digit-1} downto 0)
- Extract digits: a_d_i := unsigned(a(i*{bits_per_digit}+{bits_per_digit-1} downto i*{bits_per_digit}))
- Implement GF_ADD and GF_MUL as VHDL functions using case statements on to_integer(a) and to_integer(b)
- ALL operations MUST use digit-wise GF({k}) table lookups as described in ALGEBRAIC STRUCTURE.
- Carry and negative flags are always '0' for GF field arithmetic.
- Zero flag: check if result = all zeros."""
        else:
            return f"""VHDL ARCHITECTURE REQUIREMENTS:
⚠️ MOD constant MUST equal exactly {mod}. Do NOT use (others => '1') or any other value!
   constant MOD_VAL : unsigned({data_width} downto 0) := {mod_val_lit};
{"⚠️ CRITICAL: " + str(mod) + " EXCEEDS the VHDL integer range (max 2^31-1 = 2147483647)!" + chr(10) + "   Do NOT use to_unsigned(" + str(mod) + ", ...) — the integer argument overflows!" + chr(10) + "   The hex literal above is the ONLY correct way to declare this constant." if mod > 2147483647 else ""}

{common_rules}

3. NEVER compare unsigned with < 0. Unsigned types are ALWAYS >= 0 in VHDL.
   For SUB underflow detection, compare the ORIGINAL operands BEFORE subtraction:
     if unsigned(a) < unsigned(b) then
       sub_tmp := resize(unsigned(a), {data_width + 1}) + MOD_VAL - resize(unsigned(b), {data_width + 1});
     else
       sub_tmp := resize(unsigned(a), {data_width + 1}) - resize(unsigned(b), {data_width + 1});
     end if;

- Use unsigned type for ALL arithmetic, convert with unsigned() and std_logic_vector()

ADDITION/SUBTRACTION WIDTH:
- Two {data_width}-bit operands added can produce a ({data_width}+1)-bit result
- You MUST resize operands to {data_width + 1} bits BEFORE adding:
    sum_tmp := resize(unsigned(a), {data_width + 1}) + resize(unsigned(b), {data_width + 1});
    v_result := resize(sum_tmp mod MOD_VAL, {data_width});
- Carry flag: use if/else to check sum_tmp >= MOD_VAL (check BEFORE mod)

SUBTRACTION:
- Compare operands FIRST, then compute. v_carry := '1' if a < b, else '0'.

MULTIPLICATION:
- unsigned(a) * unsigned(b) returns {data_width * 2}-bit result automatically.
- Do NOT resize before multiplying! Use:
    variable mul_tmp : unsigned({data_width * 2 - 1} downto 0);
    mul_tmp := unsigned(a) * unsigned(b);
    v_result := resize(mul_tmp mod MOD_VAL, {data_width});
- MUL carry is always '0'

NEG SPECIAL CASE:
- NEG(0) = 0. Use: v_result := resize((MOD_VAL - resize(unsigned(a), {data_width + 1})) mod MOD_VAL, {data_width});
- NEG carry is always '0'

INC SPECIAL CASE:
- INC(max_val) = 0 with carry.
{"- Do NOT use to_unsigned(" + str(max_val) + ", " + str(data_width) + ") — integer overflow! Use resize(MOD_VAL - 1, " + str(data_width) + ")." if max_val > 2147483647 else ""}
    if unsigned(a) = resize(MOD_VAL - 1, {data_width}) then
      v_result := to_unsigned(0, {data_width}); v_carry := '1';
    else v_result := unsigned(a) + 1; v_carry := '0'; end if;

DEC SPECIAL CASE:
- DEC(0) = {mod - 1}. Do NOT assign MOD_VAL directly to v_result (width mismatch). Use resize().
    if unsigned(a) = 0 then
      v_result := resize(MOD_VAL - 1, {data_width}); v_carry := '1';
    else v_result := resize(resize(unsigned(a), {data_width + 1}) - 1, {data_width}); v_carry := '0'; end if;"""

    def _create_vhdl_prompt(self, k: int, bits: int, mod: int, operations: List[str], logic_info: Dict = None) -> str:
        """Create prompt for VHDL code generation"""
        ops_str = ', '.join(operations)
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)

        import math
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
        is_extension = (logic_info['category'] == 'extension_field' and logic_info.get('tables'))

        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, 'vhdl')

        # Pre-compute expected test results — use GF operations for extension fields
        max_val = mod - 1
        neg_half = mod // 2

        if is_extension:
            gf_vals = self._compute_gf_test_values(k, bits, logic_info)
            add_max = gf_vals['add_max']
            mul_max = gf_vals['mul_max']
            neg_max = gf_vals['neg_max']
            inc_max = gf_vals['inc_max']
            dec_max = gf_vals['dec_max']
            dec_0 = gf_vals['dec_0']
            neg_10 = gf_vals['neg_10']
            neg_1 = gf_vals['neg_1']
            sub_10_20 = gf_vals['sub_10_20']
            mul_10_20 = gf_vals['mul_10_20']
            add_10_20 = gf_vals['add_10_20']
            sub_20_10 = gf_vals['sub_20_10']
            inc_10 = gf_vals['inc_10']
            dec_10 = gf_vals['dec_10']
        else:
            add_max = (2 * max_val) % mod
            mul_max = (max_val * max_val) % mod
            neg_max = (mod - max_val) % mod
            inc_max = (max_val + 1) % mod
            dec_max = (max_val - 1 + mod) % mod
            dec_0 = (0 - 1 + mod) % mod
            neg_10 = (mod - 10) % mod
            neg_1 = (mod - 1) % mod
            sub_10_20 = (10 - 20 + mod) % mod
            mul_10_20 = (10 * 20) % mod
            add_10_20 = 30
            sub_20_10 = 10
            inc_10 = 11
            dec_10 = 9

        # Pre-compute VHDL-safe literals (hex for values > 2^31-1)
        mod_val_lit = self._vhdl_unsigned_literal(mod, data_width + 1)
        max_val_slv = self._vhdl_slv_literal(max_val, data_width)
        neg_half_lit = self._vhdl_unsigned_literal(neg_half, data_width)
        # Testbench assertion literals
        dec0_slv = self._vhdl_slv_literal(dec_0, data_width)
        add_max_slv = self._vhdl_slv_literal(add_max, data_width)
        mul_max_slv = self._vhdl_slv_literal(mul_max, data_width)
        neg_max_slv = self._vhdl_slv_literal(neg_max, data_width)
        dec_max_slv = self._vhdl_slv_literal(dec_max, data_width)
        neg_10_slv = self._vhdl_slv_literal(neg_10, data_width)
        neg_1_slv = self._vhdl_slv_literal(neg_1, data_width)
        sub_10_20_slv = self._vhdl_slv_literal(sub_10_20, data_width)

        prompt = f"""Generate a complete VHDL design for a {bits}-trit ALU operating in base-{k}.

CRITICAL RULES:
1. Output ONLY VHDL code, no markdown, no explanations
2. Must be synthesizable and simulatable with GHDL
3. Include BOTH the entity/architecture AND a testbench

SPECIFICATIONS:
- K-value: {k}
- Bitwidth: {bits} trits (need {data_width} bits to represent 0 to {mod - 1})
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}

{algebra_section}

ENTITY INTERFACE:
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity mvl_alu_{k}_{bits}bit is
    port (
        clk     : in  std_logic;
        rst     : in  std_logic;
        a       : in  std_logic_vector({data_width - 1} downto 0);
        b       : in  std_logic_vector({data_width - 1} downto 0);
        opcode  : in  std_logic_vector(3 downto 0);
        result  : out std_logic_vector({data_width - 1} downto 0);
        zero    : out std_logic;
        negative: out std_logic;
        carry   : out std_logic
    );
end entity mvl_alu_{k}_{bits}bit;

{self._build_vhdl_arch_section(k, bits, mod, data_width, is_extension, logic_info, mod_val_lit, neg_half_lit, max_val)}

TESTBENCH EXPECTED VALUES (pre-computed, mathematically verified — use these EXACT values):
  ADD(0, 0) = 0
  SUB(0, 0) = 0
  MUL(0, 0) = 0
  NEG(0) = 0
  INC(0) = 1
  DEC(0) = {dec_0}
  ADD({max_val}, {max_val}) = {add_max}
  SUB({max_val}, {max_val}) = 0
  MUL({max_val}, {max_val}) = {mul_max}
  NEG({max_val}) = {neg_max}
  INC({max_val}) = {inc_max}
  DEC({max_val}) = {dec_max}
  ADD(10, 20) = {add_10_20}
  SUB(20, 10) = {sub_20_10}
  SUB(10, 20) = {sub_10_20}
  MUL(10, 20) = {mul_10_20}
  NEG(10) = {neg_10}
  NEG(1) = {neg_1}
  INC(10) = {inc_10}
  DEC(10) = {dec_10}

⚠️ MANDATORY TESTBENCH REQUIREMENT:
The testbench MUST begin with library/use declarations:
  library IEEE;
  use IEEE.STD_LOGIC_1164.ALL;
  use IEEE.NUMERIC_STD.ALL;
You MUST include a testbench entity mvl_alu_{k}_{bits}bit_tb with AT LEAST 20 assert/report test vectors. Code with fewer than 20 tests will be REJECTED.
Include:
- 6 edge-case tests (one per opcode with a=0, b=0)
- 6 max-value tests using the pre-computed expected values above
- 8+ additional tests with various values
⚠️ EACH test MUST explicitly assign BOTH a_sig AND b_sig before setting the opcode — even if the values
are the same as the previous test! Do NOT rely on previous test values.
⚠️ CRITICAL: Use the pre-computed expected values above — do NOT compute NEG/SUB/DEC results yourself!
{"CORRECT testbench pattern — for SMALL values (≤ 2147483647), use to_unsigned:" + chr(10) + "  a_sig <= std_logic_vector(to_unsigned(VALUE_A, " + str(data_width) + "));" + chr(10) + "  assert result_sig = std_logic_vector(to_unsigned(EXPECTED, " + str(data_width) + ")) report ...;" + chr(10) + "CORRECT testbench pattern — for LARGE values (> 2147483647), use hex literal DIRECTLY — NO std_logic_vector() wrapper:" + chr(10) + "  a_sig <= " + max_val_slv + ";  -- max_val = " + str(max_val) + chr(10) + "  b_sig <= " + max_val_slv + ";" + chr(10) + "  opcode_sig <= " + '"' + "0000" + '"' + ";" + chr(10) + "  wait for 20 ns;" + chr(10) + "  assert result_sig = (" + add_max_slv + ") report " + '"' + "Test: ADD max max" + '"' + " severity error;" + chr(10) + "⚠️ CRITICAL: to_unsigned(N, " + str(data_width) + ") WILL FAIL for ANY N > 2147483647!" + chr(10) + "   Pre-computed hex literals (copy-paste these EXACTLY):" + chr(10) + "   max_val " + str(max_val) + ": " + max_val_slv + chr(10) + "   DEC(0) = " + str(dec_0) + ": " + dec0_slv + chr(10) + "   ADD(max,max) = " + str(add_max) + ": " + add_max_slv + chr(10) + "   MUL(max,max) = " + str(mul_max) + ": " + mul_max_slv + chr(10) + "   DEC(max) = " + str(dec_max) + ": " + dec_max_slv + chr(10) + "   NEG(max) = " + str(neg_max) + ": " + neg_max_slv + chr(10) + "   NEG(10) = " + str(neg_10) + ": " + neg_10_slv + chr(10) + "   NEG(1) = " + str(neg_1) + ": " + neg_1_slv + chr(10) + "   SUB(10,20) = " + str(sub_10_20) + ": " + sub_10_20_slv if mod > 2147483647 else "CORRECT pattern for EVERY test:" + chr(10) + "  a_sig <= std_logic_vector(to_unsigned(VALUE_A, " + str(data_width) + "));" + chr(10) + "  b_sig <= std_logic_vector(to_unsigned(VALUE_B, " + str(data_width) + "));" + chr(10) + "  opcode_sig <= " + '"' + "XXXX" + '"' + ";" + chr(10) + "  wait for 10 ns;" + chr(10) + "  assert result_sig = std_logic_vector(to_unsigned(EXPECTED, " + str(data_width) + ")) report " + '"' + "Test N: ..." + '"' + " severity error;"}
⚠️ IMPORTANT — ALWAYS print test results using standalone `report` statements (severity note):
  report "Test 1: ADD A=0 B=0 -> R=0 Z=1 N=0 C=0" severity note;
  assert result_sig = ... report "FAIL Test 1" severity error;
Each test MUST have BOTH: a `report ... severity note` line (always prints) AND an `assert` line (prints only on failure).
This ensures GHDL outputs visible test results regardless of pass/fail.
⚠️ CRITICAL: The testbench process MUST end with a bare "wait;" statement (no "for" clause) AFTER the last test.
This stops the process and prevents infinite re-execution. Without it, the simulation will loop forever and timeout!
  report "All tests complete" severity note;
  wait;  -- STOP simulation here (MANDATORY!)
  end process;

Generate the complete VHDL code (entity + architecture + testbench) now:
"""
        return prompt

    @staticmethod
    def _detect_language(code: str) -> Optional[str]:
        """Detect the actual language of generated code based on syntax markers."""
        code_lower = code.lower()

        # VHDL markers (must check before Verilog since both may share some keywords)
        vhdl_markers = ['library ieee', 'use ieee.', 'entity ', 'architecture ', 'std_logic', 'process(', 'process (']
        vhdl_score = sum(1 for m in vhdl_markers if m in code_lower)

        # Verilog markers
        verilog_markers = ['module ', 'endmodule', 'input wire', 'output reg', 'output wire',
                           'always @', 'always@', 'reg [', 'wire [', '$display', '$finish']
        verilog_score = sum(1 for m in verilog_markers if m in code_lower)

        # C markers
        c_markers = ['#include', 'int main', 'printf(', 'uint32_t', 'uint64_t', 'typedef ', '#define ']
        c_score = sum(1 for m in c_markers if m in code_lower)

        # Python markers
        python_markers = ['def ', 'import ', 'class ', 'if __name__', 'print(', 'self.']
        python_score = sum(1 for m in python_markers if m in code_lower)

        scores = {'vhdl': vhdl_score, 'verilog': verilog_score, 'c': c_score, 'python': python_score}
        best = max(scores, key=scores.get)

        # Only return if there's a clear signal (at least 2 markers)
        if scores[best] >= 2:
            return best
        return None

    def _extract_code(self, response: str, language: str) -> Optional[str]:
        """Extract code from LLM response, with language mismatch detection."""
        code = response

        # Try to extract from code blocks — prefer language-specific match first
        patterns = [
            rf'```{language}\s*(.*?)```',
            rf'```{language.lower()}\s*(.*?)```',
            r'```\s*(.*?)```',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1)
                break

        # Clean up
        code = code.strip()
        code = re.sub(r'^```\w*\s*', '', code)
        code = re.sub(r'\s*```$', '', code)

        if not code:
            return None

        # Language validation: detect if the extracted code actually matches the requested language
        detected = self._detect_language(code)
        if detected and detected != language.lower():
            # Map of related HDL languages for clearer warning
            print(f"\n⚠️  LANGUAGE MISMATCH DETECTED!")
            print(f"   Requested: {language.upper()}, but LLM generated: {detected.upper()}")
            print(f"   The generated code will be returned but may not compile as {language.upper()}")

        return code

    def _fix_code(self, code: str, language: str) -> str:
        """Fix common code issues across all languages"""
        lang = language.lower()

        if lang == 'c':
            # Ensure essential includes are present
            essential_includes = {
                'stdio.h': '#include <stdio.h>',
                'stdlib.h': '#include <stdlib.h>',
                'stdint.h': '#include <stdint.h>',
            }
            missing = []
            for header, include_line in essential_includes.items():
                if header not in code:
                    missing.append(include_line)
            if missing:
                code = '\n'.join(missing) + '\n\n' + code

            # Fix pow() usage without math.h
            if 'pow(' in code and 'math.h' not in code:
                code = '#include <math.h>\n' + code

            # Fix time() usage without time.h
            if 'time(' in code and 'time.h' not in code:
                code = '#include <time.h>\n' + code

        elif lang == 'python':
            if 'import random' not in code and 'random.' in code:
                code = 'import random\n' + code

        elif lang == 'verilog':
            # Fix duplicate reg/wire declarations (keep the widest one)
            code = self._fix_verilog_duplicate_decls(code)

            # Fix 'output' without 'reg' when assigned in always block
            # Find signals assigned in always blocks
            always_assigned = set()
            in_always = False
            for line in code.split('\n'):
                stripped = line.strip()
                if 'always' in stripped and '@' in stripped:
                    in_always = True
                elif in_always and ('endmodule' in stripped or (stripped.startswith('always') and '@' in stripped)):
                    in_always = False
                elif in_always:
                    # Look for assignments: signal_name <= or signal_name =
                    m = re.match(r'\s*(\w+)\s*<?=', stripped)
                    if m:
                        always_assigned.add(m.group(1))

            # Fix output declarations that should be output reg
            for sig in always_assigned:
                # Replace "output [N:0] sig" with "output reg [N:0] sig" if not already reg
                code = re.sub(
                    rf'output\s+(\[.*?\])\s+{sig}\b(?!\s*;)',
                    rf'output reg \1 {sig}',
                    code
                )
                code = re.sub(
                    rf'output\s+{sig}\b',
                    rf'output reg {sig}',
                    code
                )

            # Ensure testbench has $finish to prevent infinite simulation
            code = self._fix_verilog_finish(code)

            # Fix function return variable: LLMs often use 'result' instead of the function name
            code = self._fix_verilog_function_returns(code)

            # Fix invalid binary literals (e.g., 2'b02 -> 2'd2, 2'b03 -> 2'd3)
            code = re.sub(r"(\d+)'b([0-9]+)", lambda m: self._fix_verilog_literal(m), code)

        elif lang == 'vhdl':
            # Ensure IEEE library is present
            if 'library ieee' not in code.lower() and 'library IEEE' not in code:
                code = 'library IEEE;\nuse IEEE.STD_LOGIC_1164.ALL;\nuse IEEE.NUMERIC_STD.ALL;\n\n' + code

            # Remove conflicting std_logic_arith/std_logic_unsigned (incompatible with numeric_std)
            code = re.sub(r'(?i)\buse\s+IEEE\.STD_LOGIC_ARITH\.ALL\s*;', '-- removed: std_logic_arith conflicts with numeric_std', code)
            code = re.sub(r'(?i)\buse\s+IEEE\.STD_LOGIC_UNSIGNED\.ALL\s*;', '-- removed: std_logic_unsigned conflicts with numeric_std', code)
            code = re.sub(r'(?i)\buse\s+IEEE\.STD_LOGIC_SIGNED\.ALL\s*;', '-- removed: std_logic_signed conflicts with numeric_std', code)

            # Replace conv_integer (from std_logic_arith) with to_integer (from numeric_std)
            code = re.sub(r'\bconv_integer\b', 'to_integer', code)

            # Move variable declarations from architecture body into process
            code = self._fix_vhdl_arch_variables(code)

            # Fix function parameter names that shadow entity ports (GHDL -Whide error)
            code = self._fix_vhdl_port_shadows(code)

            # Fix case statements missing "when others" (GHDL requires full coverage)
            code = self._fix_vhdl_case_others(code)

            # Detect and fix truncated output (unterminated strings, missing end statements)
            code = self._fix_vhdl_truncation(code)

            # Ensure testbench process ends with "wait;" to prevent infinite loop
            code = self._fix_vhdl_missing_wait(code)

        return code

    @staticmethod
    def _fix_vhdl_arch_variables(code: str) -> str:
        """Move variable declarations from architecture body into the process.

        VHDL only allows variables inside processes or subprograms, not in
        the architecture declarative region. LLMs sometimes place variable
        declarations between 'begin' (of architecture) and 'process'.

        Strategy: move stray variable (and associated type) declarations
        into the first process's declarative region (between process(...) and begin).
        """
        lines = code.split('\n')

        # Phase 1: find architecture 'begin' — skip begin inside functions/procedures
        arch_begin = None
        first_process = None
        process_begin = None  # the 'begin' that belongs to the first process
        arch_found = False
        subprog_depth = 0

        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if re.match(r'architecture\s+\w+\s+of\s+', stripped):
                arch_found = True
                subprog_depth = 0
            elif arch_found and arch_begin is None:
                if re.match(r'(impure\s+)?function\s+', stripped) or re.match(r'procedure\s+', stripped):
                    subprog_depth += 1
                elif re.match(r'end\s+(function|procedure)\b', stripped):
                    subprog_depth = max(0, subprog_depth - 1)
                elif stripped == 'begin' and subprog_depth == 0:
                    arch_begin = i
            elif arch_begin is not None and first_process is None:
                if re.match(r'process\b', stripped) or re.match(r'\w+\s*:\s*process\b', stripped):
                    first_process = i
            elif first_process is not None and process_begin is None:
                if stripped == 'begin':
                    process_begin = i

        if arch_begin is None:
            return code

        # Phase 2: find stray variable/type declarations between arch begin and first process
        scan_end = first_process if first_process is not None else len(lines)
        stray_indices = []  # indices of lines to move

        for i in range(arch_begin + 1, scan_end):
            stripped_lower = lines[i].strip().lower()
            if re.match(r'variable\s+', stripped_lower):
                stray_indices.append(i)
            elif re.match(r'type\s+\w+\s+is\s+', stripped_lower):
                stray_indices.append(i)

        if not stray_indices:
            return code

        # Phase 3: move declarations into the process declarative region
        # If there's a process with a 'begin', insert just before it
        # Otherwise, move to architecture declarative region as signals
        new_lines = list(lines)
        moved = []
        for idx in sorted(stray_indices):
            orig = lines[idx].strip()
            moved.append('        ' + orig)  # process-level indent
            new_lines[idx] = None  # mark for removal
            print(f"   ⚠️ Moved to process: {orig[:80]}")

        if process_begin is not None:
            # Insert just before the process 'begin'
            result = []
            for i, line in enumerate(new_lines):
                if i == process_begin:
                    result.extend(moved)
                if line is not None:
                    result.append(line)
        else:
            # No process found — convert to signals in arch declarative region
            for j, m in enumerate(moved):
                moved[j] = re.sub(r'(?i)\bvariable\b', 'signal', m)
            result = []
            for i, line in enumerate(new_lines):
                if i == arch_begin:
                    result.extend(moved)
                if line is not None:
                    result.append(line)

        return '\n'.join(result)

    @staticmethod
    def _fix_vhdl_case_others(code: str) -> str:
        """Add 'when others' to VHDL case statements that are missing it.

        GHDL --std=08 requires all case statements to fully cover the
        selector type's range. to_integer() returns 'natural' (0..2^31-1),
        so case statements with only specific values (0,1,2,3) must have
        'when others' to cover the rest. Without it:
          error: no choices for 4 to 2147483647

        Uses a two-pass approach:
          1. Parse the source to find case/end-case pairs and track nesting.
          2. Insert 'when others' lines (from bottom to top) where missing.
        """
        lines = code.split('\n')

        # Stack entries: (case_line_index, has_when_others, has_return)
        # has_return tracks whether any 'return' appears within this case
        stack: list[dict] = []
        # Collect insertions: list of (line_index, indent, in_function)
        insertions: list[tuple[int, str, bool]] = []

        for i, line in enumerate(lines):
            stripped = line.strip().lower()

            # Detect "case ... is"
            if re.match(r'case\s+.+\s+is\b', stripped):
                stack.append({
                    'case_line': i,
                    'has_when_others': False,
                    'has_return': False,
                })

            # Track "when others" within current (innermost) case
            elif 'when others' in stripped and stack:
                stack[-1]['has_when_others'] = True

            # Track "return" within current case (indicates we're in a function)
            elif 'return ' in stripped and stack:
                stack[-1]['has_return'] = True

            # Detect "end case;"
            elif re.match(r'end\s+case\s*;', stripped):
                if stack:
                    info = stack.pop()
                    if not info['has_when_others']:
                        indent = line[:len(line) - len(line.lstrip())]
                        insertions.append((i, indent, info['has_return']))

        # Apply insertions from bottom to top so line indices stay valid
        for line_idx, indent, in_function in reversed(insertions):
            if in_function:
                new_line = f'{indent}    when others => return to_unsigned(0, 2);'
            else:
                new_line = f'{indent}    when others => null;'
            lines.insert(line_idx, new_line)

        return '\n'.join(lines)

    @staticmethod
    def _fix_vhdl_port_shadows(code: str) -> str:
        """Fix VHDL function parameters that shadow entity port names.

        GHDL treats 'declaration of "a" hides port "a"' as an error with --std=08.
        Renames function parameters a->x_val, b->y_val to avoid conflicts.
        """
        # Find entity port names
        port_names = set()
        in_port = False
        paren_depth = 0
        for line in code.split('\n'):
            stripped = line.strip().lower()
            if re.match(r'\s*port\s*\(', stripped):
                in_port = True
                paren_depth = 0
            if in_port:
                paren_depth += line.count('(') - line.count(')')
                # Match port declarations like "a : in std_logic_vector(...)"
                m = re.match(r'\s*(\w+)\s*:\s*(in|out|inout|buffer)\s+', line, re.IGNORECASE)
                if m:
                    port_names.add(m.group(1).lower())
                # End of port section: when parens balance back to 0
                if paren_depth <= 0 and ';' in stripped:
                    in_port = False

        if not port_names:
            return code

        # Find and rename function parameters that conflict
        # Pattern: function name(param : type; param : type) return ...
        renames = {
            'a': 'x_val', 'b': 'y_val', 'result': 'r_val',
            'carry': 'c_val', 'zero': 'z_val', 'negative': 'n_val',
        }

        lines = code.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for function declaration
            func_match = re.match(
                r'(\s*function\s+\w+\s*\()(.*?)(\)\s*return\b.*)',
                line, re.IGNORECASE
            )
            if func_match:
                prefix, params_str, suffix = func_match.groups()
                # Find parameter names that shadow ports
                active_renames = {}
                for pname in port_names:
                    if re.search(r'\b' + pname + r'\s*:', params_str, re.IGNORECASE):
                        if pname in renames:
                            active_renames[pname] = renames[pname]

                if active_renames:
                    # Rename in function parameter list
                    new_params = params_str
                    for old, new in active_renames.items():
                        new_params = re.sub(
                            r'\b' + old + r'\b',
                            new,
                            new_params,
                            flags=re.IGNORECASE
                        )
                    line = prefix + new_params + suffix

                    # Find the end of this function and rename within its body
                    new_lines.append(line)
                    i += 1
                    func_depth = 1
                    while i < len(lines) and func_depth > 0:
                        fline = lines[i]
                        if re.search(r'\bend\s+function\b', fline, re.IGNORECASE):
                            func_depth -= 1
                        elif re.search(r'\bfunction\b', fline, re.IGNORECASE):
                            func_depth += 1
                        # Rename parameter references in function body
                        for old, new in active_renames.items():
                            # Only rename standalone identifiers, not parts of other names
                            fline = re.sub(
                                r'\b' + old + r'\b',
                                new,
                                fline,
                                flags=re.IGNORECASE
                            )
                        new_lines.append(fline)
                        i += 1
                    continue

            new_lines.append(line)
            i += 1

        return '\n'.join(new_lines)

    @staticmethod
    def _fix_vhdl_missing_wait(code: str) -> str:
        """Ensure every VHDL testbench process ends with a bare 'wait;'.

        Without a terminal wait, the process restarts from the top after reaching
        'end process', causing an infinite simulation loop that hits the timeout.
        """
        lines = code.split('\n')
        result = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip().lower()
            # Detect "end process" lines
            if re.match(r'end\s+process\b', stripped):
                # Check if the previous non-blank line is "wait;"
                has_wait = False
                for j in range(len(result) - 1, max(len(result) - 5, -1), -1):
                    prev = result[j].strip().lower()
                    if prev == '':
                        continue
                    if prev == 'wait;' or prev == 'wait ;':
                        has_wait = True
                    break
                if not has_wait:
                    # Determine indent from end process line
                    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                    result.append(indent + '    wait;  -- auto-added: prevent infinite loop')
                    print("   ⚠️ Added terminal 'wait;' before 'end process' to prevent simulation timeout")
            result.append(lines[i])
            i += 1
        return '\n'.join(result)

    @staticmethod
    def _fix_vhdl_truncation(code: str) -> str:
        """Detect and attempt to fix truncated VHDL output from LLM.

        Common truncation symptoms:
        - Unterminated string literals
        - Missing 'end architecture', 'end process', etc.
        - Code ends abruptly mid-statement
        """
        lines = code.split('\n')

        # Check for unterminated string on the last few lines
        # Remove trailing lines that have unterminated strings
        while lines:
            last_line = lines[-1].strip()
            if not last_line:
                lines.pop()
                continue
            # Count quotes: odd number means unterminated string
            quote_count = last_line.count('"')
            if quote_count % 2 != 0:
                # Unterminated string — remove this line
                print(f"   ⚠️ Removing truncated line: {last_line[:80]}...")
                lines.pop()
                continue
            break

        # Check if code has proper VHDL endings
        code_lower = '\n'.join(lines).lower()

        # Count architecture/end architecture pairs
        arch_count = len(re.findall(r'\barchitecture\b\s+\w+\s+of\b', code_lower))
        end_arch_count = len(re.findall(r'\bend\s+(architecture\b|behavioral\b|rtl\b|structural\b)', code_lower))

        # Count process/end process pairs
        # Match lines starting with "process" (with optional whitespace)
        # Handles: process(clk), process (clk, rst), process, process is
        process_starts = len(re.findall(r'^\s*process\b', code_lower, re.MULTILINE))
        process_ends = len(re.findall(r'\bend\s+process\b', code_lower))

        # Add missing end statements
        additions = []
        for _ in range(process_starts - process_ends):
            additions.append('    end process;')
            print("   ⚠️ Added missing 'end process;'")

        for _ in range(arch_count - end_arch_count):
            additions.append('end architecture Behavioral;')
            print("   ⚠️ Added missing 'end architecture;'")

        if additions:
            lines.extend(additions)

        # Final check: ensure there's a semicolon-terminated statement at the end
        # (skip empty/comment lines)
        result = '\n'.join(lines)
        return result

    def _fix_verilog_duplicate_decls(self, code: str) -> str:
        """Remove duplicate reg/wire declarations in Verilog.

        LLMs commonly declare the same variable twice, e.g.:
            reg [25:0] temp;   // wide version for MUL
            reg temp;           // duplicate 1-bit version

        Strategy: for each variable name, keep the declaration with the
        widest bit range and remove the rest.
        """
        lines = code.split('\n')
        # decl_pat matches: reg [N:M] name; / reg name; / wire [N:M] name;
        decl_pat = re.compile(
            r'^(\s*)(reg|wire)\s+'           # indent + keyword
            r'(?:\[(\d+):(\d+)\]\s+)?'       # optional [hi:lo]
            r'(\w+)\s*;'                      # variable name + semicolon
        )

        # First pass: collect all declarations per variable name
        # { var_name: [ (line_idx, width, full_match) ] }
        decl_map = {}
        for i, line in enumerate(lines):
            m = decl_pat.match(line)
            if not m:
                continue
            _indent, _kw, hi, lo, var_name = m.groups()
            if hi is not None and lo is not None:
                width = abs(int(hi) - int(lo)) + 1
            else:
                width = 1
            decl_map.setdefault(var_name, []).append((i, width))

        # Second pass: for vars with multiple declarations, remove all but widest
        lines_to_remove = set()
        for var_name, entries in decl_map.items():
            if len(entries) <= 1:
                continue
            # Keep the entry with the largest width (first occurrence on tie)
            entries.sort(key=lambda e: (-e[1], e[0]))
            keep_idx = entries[0][0]
            for idx, _w in entries[1:]:
                lines_to_remove.add(idx)

        if lines_to_remove:
            lines = [l for i, l in enumerate(lines) if i not in lines_to_remove]

        return '\n'.join(lines)

    def _fix_verilog_finish(self, code: str) -> str:
        """Ensure Verilog testbench has $finish to prevent infinite simulation.

        Common LLM issue: testbench has `forever #5 clk = ~clk;` in one
        initial block, but the other initial block's `$finish` is missing
        or unreachable. This adds a safety-net `$finish` if needed.
        """
        has_finish = '$finish' in code
        has_tb = '_tb' in code or 'testbench' in code.lower()

        if has_finish or not has_tb:
            return code

        # No $finish found in a testbench — insert before final `endmodule`
        lines = code.split('\n')
        # Find the last `endmodule`
        last_endmod = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == 'endmodule':
                last_endmod = i
                break

        if last_endmod is not None:
            # Insert a safety initial block before the last endmodule
            safety = [
                '',
                '// Safety net: auto-added $finish',
                'initial begin',
                '    #100000;',
                '    $display("TIMEOUT: simulation auto-terminated");',
                '    $finish;',
                'end',
            ]
            lines = lines[:last_endmod] + safety + lines[last_endmod:]

        return '\n'.join(lines)

    @staticmethod
    def _fix_verilog_literal(m) -> str:
        """Fix invalid binary literals like 2'b02 -> 2'd2.

        Binary literals can only contain 0/1. If digits > 1 are found,
        convert to decimal format.
        """
        width = m.group(1)
        digits = m.group(2)
        # Check if all digits are valid binary (0 or 1)
        if all(c in '01' for c in digits):
            return m.group(0)  # Valid binary, keep as-is
        # Invalid binary — convert to decimal
        try:
            # Treat the digit string as a decimal number
            val = int(digits)
            return f"{width}'d{val}"
        except ValueError:
            return m.group(0)

    def _fix_verilog_function_returns(self, code: str) -> str:
        """Fix Verilog functions that use wrong return variable names.

        LLMs often write:
            function [1:0] gf_sub;
                ...
                result = a;  // WRONG: should be gf_sub = a;
            endfunction

        This detects function blocks and replaces assignments to undeclared
        variables with the function name.
        """
        lines = code.split('\n')
        result_lines = []
        current_func = None  # Name of the function we're inside
        func_locals = set()  # Declared local variables in current function

        for line in lines:
            stripped = line.strip()

            # Detect function start: "function [1:0] func_name;"
            func_match = re.match(
                r'\s*function\s+(?:\[\s*\d+\s*:\s*\d+\s*\]\s+)?(\w+)\s*;', stripped
            )
            if func_match:
                current_func = func_match.group(1)
                func_locals = set()
                result_lines.append(line)
                continue

            if current_func:
                # Track local variable declarations: "input [1:0] a, b;" or "reg [1:0] temp;"
                input_match = re.match(r'\s*(?:input|reg)\s+(?:\[.*?\]\s+)?(.+);', stripped)
                if input_match:
                    vars_str = input_match.group(1)
                    for v in vars_str.split(','):
                        func_locals.add(v.strip())

                # Detect endfunction
                if stripped == 'endfunction':
                    current_func = None
                    func_locals = set()
                    result_lines.append(line)
                    continue

                # Fix assignments to undeclared variables (likely meant to be function name)
                # Handle both plain "result = X;" and case-label "2'b00: result = X;"
                assign_match = re.search(r'(?:^|\:\s*)(\w+)\s*=\s*(.+)', stripped)
                if assign_match:
                    var_name = assign_match.group(1)
                    rhs_value = assign_match.group(2).rstrip(';').strip()
                    # If var_name is not the function name and not a declared local
                    if (var_name != current_func
                            and var_name not in func_locals
                            and var_name not in ('i', 'j', 'k')):
                        # Replace the variable name with function name in the line
                        line = line.replace(
                            f'{var_name} = {assign_match.group(2)}',
                            f'{current_func} = {assign_match.group(2)}'
                        )
                    # Remove self-referencing line: "gf_sub = result;" where result
                    # was the old temp variable name we've been renaming
                    elif (var_name == current_func
                          and rhs_value not in func_locals
                          and re.match(r'^\w+$', rhs_value)
                          and rhs_value not in ('0', '1')
                          and rhs_value != current_func):
                        # This is "func = old_temp_var;" — skip it
                        continue

            result_lines.append(line)

        return '\n'.join(result_lines)

    def _enhance_test_coverage(self, code: str, language: str, k: int, bits: int, mod: int) -> str:
        """If test coverage is low, do a focused LLM call to add more test vectors."""
        import math
        lang = language.lower()
        test_indicators = {
            'c': ['printf', 'test', 'assert'],
            'python': ['print', 'test', 'assert'],
            'verilog': ['$display', 'initial begin', '#'],
            'vhdl': ['assert', 'report', 'wait for'],
        }
        indicators = test_indicators.get(lang, [])
        test_count = sum(1 for line in code.split('\n')
                         if any(ind in line.lower() for ind in indicators))

        if test_count >= 10:
            return code  # Already has enough tests

        print(f"\n🔧 Low test coverage detected ({test_count} lines). Auto-enhancing with more test vectors...")

        lang_hint = {
            'c': f'Add more printf test lines in main(). Use alu_exec(a, b, op) and printf to print results. Include edge cases (0,0), max values ({mod-1},{mod-1}), and random values for all 6 operations (OP_ADD through OP_DEC).',
            'python': f'Add more print test lines in the if __name__ == "__main__" block. Use alu_exec(a, b, op) and print results. Include edge cases (0,0), max values ({mod-1},{mod-1}), and random values for all 6 operations.',
            'verilog': f'Add more $display test vectors in the testbench initial block. Test all 6 opcodes (0000-0101) with edge cases (a=0,b=0), max values (a={mod-1},b={mod-1}), and various other values. Use #10 between tests.',
            'vhdl': f'Add more assert/report test vectors in the testbench process. Test all 6 opcodes ("0000"-"0101") with edge cases (a=0,b=0), max values (a={mod-1},b={mod-1}), and various other values. Use wait for 10 ns between tests.',
        }

        enhance_prompt = f"""Here is existing {language.upper()} code that has too few test vectors. Add AT LEAST 15 more test vectors to the test/main section to bring the total to 20+.

RULES:
1. Output the COMPLETE code (original + new tests) — do NOT remove any existing code
2. Only add test vectors, do NOT modify the ALU logic
3. {lang_hint.get(lang, lang_hint['c'])}
4. No markdown, no explanations — output ONLY {language.upper()} code

EXISTING CODE:
{code}

Output the complete enhanced code now:"""

        try:
            response = self.llm._call_api(
                enhance_prompt,
                max_tokens=16384,
                system_prompt="You are an expert programmer. Add test vectors to the provided code. Output ONLY code, no explanations."
            )
            enhanced_code = self._extract_code(response, language)
            if enhanced_code:
                enhanced_code = self._fix_code(enhanced_code, language)
                # Verify enhancement actually improved test coverage
                new_test_count = sum(1 for line in enhanced_code.split('\n')
                                     if any(ind in line.lower() for ind in indicators))
                if new_test_count > test_count:
                    print(f"✅ Test coverage enhanced: {test_count} → {new_test_count} test-related lines")
                    return enhanced_code
                else:
                    print(f"⚠️ Enhancement did not improve coverage, keeping original code")
            else:
                print(f"⚠️ Failed to extract enhanced code, keeping original")
        except Exception as e:
            print(f"⚠️ Test enhancement failed ({e}), keeping original code")

        return code

    def _validate_code(self, code: str, language: str, k: int, bits: int, mod: int,
                       logic_info: Dict = None) -> List[str]:
        """Validate generated code quality. Returns a list of warning strings."""
        import math
        warnings = []
        lang = language.lower()
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
        code_lower = code.lower()
        is_extension = (logic_info and logic_info.get('category') == 'extension_field'
                        and logic_info.get('tables'))

        # 1. Language mismatch check
        detected = self._detect_language(code)
        if detected and detected != lang:
            warnings.append(f"LANGUAGE MISMATCH: requested {lang.upper()}, detected {detected.upper()}")

        # 2. Check all 6 operations are present
        op_keywords = {
            'c': {'add': ['add', 'op_add'], 'sub': ['sub', 'op_sub'], 'mul': ['mul', 'op_mul'],
                   'neg': ['neg', 'op_neg'], 'inc': ['inc', 'op_inc'], 'dec': ['dec', 'op_dec']},
            'python': {'add': ['add', 'op_add'], 'sub': ['sub', 'op_sub'], 'mul': ['mul', 'op_mul'],
                       'neg': ['neg', 'op_neg'], 'inc': ['inc', 'op_inc'], 'dec': ['dec', 'op_dec']},
            'verilog': {'add': ['0000', 'add'], 'sub': ['0001', 'sub'], 'mul': ['0010', 'mul'],
                        'neg': ['0011', 'neg'], 'inc': ['0100', 'inc'], 'dec': ['0101', 'dec']},
            'vhdl': {'add': ['0000', 'add'], 'sub': ['0001', 'sub'], 'mul': ['0010', 'mul'],
                     'neg': ['0011', 'neg'], 'inc': ['0100', 'inc'], 'dec': ['0101', 'dec']},
        }
        if lang in op_keywords:
            for op_name, keywords in op_keywords[lang].items():
                if not any(kw in code_lower for kw in keywords):
                    warnings.append(f"MISSING OPERATION: {op_name.upper()} not found in code")

        # 3. MOD value check (skip for GF extension fields — they use digit-wise
        #    table lookups, not modular arithmetic with the full mod value)
        if not is_extension:
            mod_str = str(mod)
            if mod_str not in code:
                warnings.append(f"MOD VALUE: {mod} not found in code — arithmetic may be wrong")

        # 4. Bit width check for HDL
        if lang == 'verilog':
            # Check for wrong bit widths like [9:0] when should be [15:0]
            expected_range = f'[{data_width - 1}:0]'
            if expected_range not in code and f'[{bits - 1}:0]' in code:
                warnings.append(
                    f"BIT WIDTH: found [{bits - 1}:0] (trit count) instead of {expected_range} ({data_width} bits needed for mod {mod})")
        elif lang == 'vhdl':
            expected_range = f'{data_width - 1} downto 0'
            if expected_range not in code and f'{bits - 1} downto 0' in code:
                warnings.append(
                    f"BIT WIDTH: found '{bits - 1} downto 0' (trit count) instead of '{expected_range}' ({data_width} bits needed for mod {mod})")

        # 5. Test coverage check
        test_indicators = {
            'c': ['printf', 'test', 'assert'],
            'python': ['print', 'test', 'assert'],
            'verilog': ['$display', 'initial begin', '#'],
            'vhdl': ['assert', 'report', 'wait for'],
        }
        indicators = test_indicators.get(lang, [])
        test_count = sum(1 for line in code.split('\n')
                         if any(ind in line.lower() for ind in indicators))
        if test_count < 10:
            warnings.append(f"LOW TEST COVERAGE: only ~{test_count} test-related lines found (recommend 20+)")

        # 6. C-specific: check for overflow risk in multiplication
        if lang == 'c':
            if 'uint32_t' in code and 'uint64_t' not in code and mod > 256:
                warnings.append(
                    f"OVERFLOW RISK: uint32_t may overflow for MUL ({mod - 1}*{mod - 1}={((mod - 1) ** 2)}), use uint64_t")

        # Print warnings
        if warnings:
            print(f"\n🔍 Code validation found {len(warnings)} issue(s):")
            for i, w in enumerate(warnings, 1):
                print(f"   {i}. {w}")
        else:
            print(f"\n✅ Code validation passed — no issues detected")

        return warnings

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        """Convert natural language text to a safe filename component.

        e.g. '10-trit ternary calculator' -> '10_trit_ternary_calculator'
        """
        # Replace non-alphanumeric characters (except hyphens) with underscores
        name = re.sub(r'[^a-zA-Z0-9\-]', '_', text.strip())
        # Collapse multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Limit length
        if len(name) > 60:
            name = name[:60].rstrip('_')
        return name.lower()

    def _save_code(self, code: str, k: int, bits: int, language: str,
                   natural_input: str = '', module_type: str = 'alu') -> Path:
        """Save generated code to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # File extension
        extensions = {
            'c': 'c',
            'python': 'py',
            'verilog': 'v',
            'vhdl': 'vhd'
        }
        ext = extensions.get(language.lower(), 'txt')

        # Module type label for filename
        mod_label = module_type.replace('-', '_')

        # When natural_input is provided, use it for the filename
        if natural_input:
            safe_name = self._sanitize_filename(natural_input)
            filename = f"{safe_name}_{timestamp}.{ext}"
            description = natural_input
        else:
            filename = f"mvl_{mod_label}_k{k}_{bits}trit_{timestamp}.{ext}"
            description = f"k={k} {bits}-trit {module_type.upper()}"

        file_path = self.output_dir / filename

        # Add header comment
        if language.lower() == 'c':
            header = f"""/*
 * MVL ALU Benchmark - {description}
 * Generated by: MVL Benchmark Generator
 * LLM Provider: {self.llm_provider_name}
 * Generated at: {timestamp}
 * MOD value: {k ** bits}
 */

"""
        elif language.lower() == 'python':
            header = f'''"""
MVL ALU Benchmark - {description}
Generated by: MVL Benchmark Generator
LLM Provider: {self.llm_provider_name}
Generated at: {timestamp}
MOD value: {k ** bits}
"""

'''
        elif language.lower() == 'vhdl':
            header = f"""-- MVL ALU Benchmark - {description}
-- Generated by: MVL Benchmark Generator
-- LLM Provider: {self.llm_provider_name}
-- Generated at: {timestamp}
-- MOD value: {k ** bits}

"""
        else:
            header = f"""// MVL ALU Benchmark - {description}
// Generated by: MVL Benchmark Generator
// LLM Provider: {self.llm_provider_name}
// Generated at: {timestamp}
// MOD value: {k ** bits}

"""

        full_code = header + code

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_code)

        return file_path


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate MVL ALU Benchmarks')
    parser.add_argument('--llm', default='groq', help='LLM provider')
    parser.add_argument('--k', type=int, default=3, help='K-value (2-8)')
    parser.add_argument('--bits', type=int, default=8, help='Bitwidth (8,10,12,14)')
    parser.add_argument('--lang', default='c', help='Language (c, python, verilog)')
    parser.add_argument('--output', help='Output directory')

    args = parser.parse_args()

    generator = MVLGenerator(
        llm_provider=args.llm,
        output_dir=args.output
    )

    result = generator.generate(
        k_value=args.k,
        bitwidth=args.bits,
        language=args.lang
    )

    if result['success']:
        print(f"\n✅ Generation successful!")
        print(f"   File: {result['file_path']}")
    else:
        print(f"\n❌ Generation failed: {result.get('error')}")


if __name__ == '__main__':
    main()
