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
  - NEG: for each digit position i: result_digit[i] = ({p} - a_digit[i]) % {p}
         (additive inverse in GF({p}), since digit addition is mod {p})
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
            module_type: str = 'alu'
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
        print(f"   LLM Provider: {self.llm_provider_name} ({actual_class})")
        print(f"   LLM Model: {actual_model}")
        print(f"{'=' * 60}")

        # Calculate MOD value
        mod_value = k_value ** bitwidth

        # Create prompt: use natural_input when provided, otherwise use structured prompt
        if natural_input:
            prompt = self._create_natural_prompt(natural_input, k_value, bitwidth, mod_value, operations, language, logic_info)
        elif language.lower() == 'c':
            prompt = self._create_c_prompt(k_value, bitwidth, mod_value, operations, logic_info)
        elif language.lower() == 'python':
            prompt = self._create_python_prompt(k_value, bitwidth, mod_value, operations, logic_info)
        elif language.lower() == 'verilog':
            prompt = self._create_verilog_prompt(k_value, bitwidth, mod_value, operations, logic_info)
        elif language.lower() == 'vhdl':
            prompt = self._create_vhdl_prompt(k_value, bitwidth, mod_value, operations, logic_info)
        else:
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
                max_tokens=8192,
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
            validation_warnings = self._validate_code(code, language, k_value, bitwidth, mod_value)

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
            module_type: str = 'alu'
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

        # Create prompt: use natural_input when provided
        if natural_input:
            prompt = self._create_natural_prompt(natural_input, k_value, bitwidth, mod_value, operations, language, logic_info)
        elif language.lower() == 'c':
            prompt = self._create_c_prompt(k_value, bitwidth, mod_value, operations, logic_info)
        elif language.lower() == 'python':
            prompt = self._create_python_prompt(k_value, bitwidth, mod_value, operations, logic_info)
        elif language.lower() == 'verilog':
            prompt = self._create_verilog_prompt(k_value, bitwidth, mod_value, operations, logic_info)
        elif language.lower() == 'vhdl':
            prompt = self._create_vhdl_prompt(k_value, bitwidth, mod_value, operations, logic_info)
        else:
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
                                   module_type=module_type)
            if result.get('success') and result.get('code'):
                yield ("chunk", result['code'])
            yield ("done", result)
            return

        try:
            full_response = ""
            system_prompt = "You are an expert programmer. Generate clean, compilable code without any explanations."

            for chunk in self.llm._call_api_stream(prompt, max_tokens=8192, system_prompt=system_prompt):
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
3. #define MOD {mod}ULL
4. Result struct containing BOTH uint64_t result AND Flags (z, n, c)
   ⚠️ alu_exec() MUST return a struct with both the result value AND the flags!
   Do NOT return only flags while discarding the result as a local variable.
5. For MUL: use {c_mul_type} for intermediate product ({mod-1} * {mod-1} = {mul_product_max})
6. ⚠️ NEG: MUST use (MOD - a) % MOD, NOT just (MOD - a). When a=0, (MOD-0)=MOD which is NOT in range!
   NEG(0) must equal 0. The % MOD is mandatory.
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

REQUIRED STRUCTURE:
1. imports (random, dataclasses or namedtuple)
2. MOD = {mod}
3. Operation constants (OP_ADD=0, OP_SUB=1, etc.)
4. Flags dataclass or namedtuple (z, n, c)
5. alu_exec(a, b, op) function returning (result, flags)
6. op_name(op) function
7. if __name__ == "__main__": block with AT LEAST 20 print test lines

⚠️ MANDATORY TEST REQUIREMENT:
The main block MUST include AT LEAST 20 test vectors using print(). Code with fewer than 20 tests will be REJECTED.
Include:
- 6 edge-case tests (one per operation with a=0, b=0)
- 6 max-value tests (one per operation with a={mod-1}, b={mod-1})
- 8+ random tests with random.randint(0, {mod-1})

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

        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, 'verilog')

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

ARCHITECTURE REQUIREMENTS:
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
- Use 'output reg' for result/flags since they are assigned in always block

⚠️ MANDATORY TESTBENCH REQUIREMENT:
You MUST include a testbench module mvl_alu_{k}_{bits}bit_tb with AT LEAST 20 $display test vectors. Code with fewer than 20 tests will be REJECTED.
Include:
- 6 edge-case tests (one per opcode with a=0, b=0)
- 6 max-value tests (one per opcode with a={mod-1}, b={mod-1})
- 8+ additional tests with various values
Each test: assign a, b, opcode → #10 → $display result

Generate the complete Verilog module with testbench now:
"""
        return prompt

    def _create_vhdl_prompt(self, k: int, bits: int, mod: int, operations: List[str], logic_info: Dict = None) -> str:
        """Create prompt for VHDL code generation"""
        ops_str = ', '.join(operations)
        if logic_info is None:
            logic_info = self._resolve_logic_type(k)

        import math
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1

        algebra_section = self._build_algebra_section(k, bits, mod, logic_info, 'vhdl')

        # Pre-compute some expected test results for testbench hints
        max_val = mod - 1
        add_max = (2 * max_val) % mod
        mul_max = (max_val * max_val) % mod
        neg_max = (mod - max_val) % mod
        inc_max = (max_val + 1) % mod
        dec_max = (max_val - 1 + mod) % mod
        neg_half = mod // 2
        # Pre-compute small-value NEG results (LLMs often miscalculate these)
        neg_10 = (mod - 10) % mod
        neg_1 = (mod - 1) % mod
        sub_10_20 = (10 - 20 + mod) % mod
        mul_10_20 = (10 * 20) % mod

        # Pre-compute VHDL-safe literals (hex for values > 2^31-1)
        mod_val_lit = self._vhdl_unsigned_literal(mod, data_width + 1)
        max_val_slv = self._vhdl_slv_literal(max_val, data_width)
        neg_half_lit = self._vhdl_unsigned_literal(neg_half, data_width)
        # Testbench assertion literals
        dec0_slv = self._vhdl_slv_literal((0 - 1 + mod) % mod, data_width)
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

VHDL ARCHITECTURE REQUIREMENTS:
⚠️ MOD constant MUST equal exactly {mod}. Do NOT use (others => '1') or any other value!
   constant MOD_VAL : unsigned({data_width} downto 0) := {mod_val_lit};
{"⚠️ CRITICAL: " + str(mod) + " EXCEEDS the VHDL integer range (max 2^31-1 = 2147483647)!" + chr(10) + "   Do NOT use to_unsigned(" + str(mod) + ", ...) — the integer argument overflows!" + chr(10) + "   Do NOT use integer type for MOD_VAL!" + chr(10) + "   The hex literal above is the ONLY correct way to declare this constant." if mod > 2147483647 else ""}

⚠️ VHDL VARIABLE DECLARATION RULES (CRITICAL — violations cause compilation failure):
1. ALL variable declarations MUST be in the process DECLARATIVE REGION (between "process" and "begin").
   Do NOT declare variables inside case/when branches or if/else blocks — this is ILLEGAL in VHDL!
   CORRECT:
     process(clk, rst)
       variable v_result : unsigned({data_width - 1} downto 0);
       variable sum_tmp  : unsigned({data_width} downto 0);
       variable mul_tmp  : unsigned({data_width * 2 - 1} downto 0);
       variable v_carry  : std_logic;
     begin
       -- use variables here in case/when branches
   WRONG (will NOT compile):
     when "0000" =>
       variable sum_tmp : unsigned({data_width} downto 0);  -- ❌ ILLEGAL

2. Do NOT use "when...else" syntax for variable assignment inside a process — ANYWHERE!
   It is only valid in VHDL-2008 and GHDL defaults to VHDL-93.
   This applies to ALL variable assignments including carry, zero, and negative flags.
   WRONG:  v_carry := '1' when condition else '0';          -- ❌ VHDL-93 error
   WRONG:  v_zero := '1' when v_result = 0 else '0';       -- ❌ VHDL-93 error
   WRONG:  v_negative := '1' when v_result >= X else '0';   -- ❌ VHDL-93 error
   CORRECT (use if/else for EVERY conditional variable assignment):
     if sum_tmp >= MOD_VAL then v_carry := '1'; else v_carry := '0'; end if;
     if v_result = to_unsigned(0, {data_width}) then v_zero := '1'; else v_zero := '0'; end if;
     if v_result >= {neg_half_lit} then v_negative := '1'; else v_negative := '0'; end if;

3. NEVER compare unsigned with < 0. Unsigned types are ALWAYS >= 0 in VHDL.
   The comparison "unsigned_var < 0" is ALWAYS false and is a logic bug!
   For SUB underflow detection, compare the ORIGINAL operands BEFORE subtraction:
   WRONG:  sub_tmp := unsigned(a) - unsigned(b); if sub_tmp < 0 then ...  -- ❌ always false
   CORRECT:
     if unsigned(a) < unsigned(b) then
       sub_tmp := resize(unsigned(a), {data_width + 1}) + MOD_VAL - resize(unsigned(b), {data_width + 1});
     else
       sub_tmp := resize(unsigned(a), {data_width + 1}) - resize(unsigned(b), {data_width + 1});
     end if;

- Use variables (NOT signals) inside process for intermediate results — signals update one delta later
- Use unsigned type for ALL arithmetic, convert with unsigned() and std_logic_vector()

ADDITION/SUBTRACTION WIDTH:
- Two {data_width}-bit operands added can produce a ({data_width}+1)-bit result
- You MUST resize operands to {data_width + 1} bits BEFORE adding:
    sum_tmp := resize(unsigned(a), {data_width + 1}) + resize(unsigned(b), {data_width + 1});
    v_result := resize(sum_tmp mod MOD_VAL, {data_width});  -- truncate back to {data_width} bits
- Carry flag: use if/else to check sum_tmp >= MOD_VAL (check BEFORE mod)

SUBTRACTION:
- Do NOT subtract then check < 0 — unsigned can never be < 0!
- Compare operands FIRST, then compute:
    if unsigned(a) < unsigned(b) then
      sub_tmp := resize(unsigned(a), {data_width + 1}) + MOD_VAL - resize(unsigned(b), {data_width + 1});
      v_carry := '1';
    else
      sub_tmp := resize(unsigned(a), {data_width + 1}) - resize(unsigned(b), {data_width + 1});
      v_carry := '0';
    end if;
    v_result := resize(sub_tmp mod MOD_VAL, {data_width});

MULTIPLICATION:
- In VHDL numeric_std, the "*" operator returns length = left'length + right'length.
  So unsigned(a) * unsigned(b) with {data_width}-bit operands returns {data_width * 2}-bit result automatically.
- ⚠️ Do NOT resize operands before multiplying! resize(a,{data_width*2}) * resize(b,{data_width*2}) returns {data_width*4} bits,
  which causes a length mismatch when assigned to a {data_width*2}-bit variable!
- CORRECT multiplication:
    variable mul_tmp : unsigned({data_width * 2 - 1} downto 0);  -- {data_width * 2} bits
    mul_tmp := unsigned(a) * unsigned(b);  -- {data_width}*2 = {data_width * 2} bits, matches mul_tmp
    v_result := resize(mul_tmp mod MOD_VAL, {data_width});
- WRONG (will cause bound check failure):
    mul_tmp := resize(unsigned(a), {data_width * 2}) * resize(unsigned(b), {data_width * 2});  -- ❌ returns {data_width * 4} bits!
- MUL carry is always '0'

NEG SPECIAL CASE:
- NEG(0) = 0, NOT {mod}. Because ({mod} - 0) mod {mod} = 0.
- Use: v_result := resize((MOD_VAL - resize(unsigned(a), {data_width + 1})) mod MOD_VAL, {data_width});
- NEG carry is always '0'

DEC SPECIAL CASE:
- DEC(0) = {mod - 1}. Do NOT use unsigned subtraction then check < 0!
- ⚠️ Do NOT use (others => '0') in comparisons — VHDL cannot infer its length! Use "= 0" instead.
- ⚠️ Do NOT assign MOD_VAL directly to v_result — MOD_VAL is {data_width + 1} bits but v_result is {data_width} bits! Always wrap with resize().
- Copy this EXACT code (do NOT modify it):
    if unsigned(a) = 0 then
      v_result := resize(MOD_VAL - 1, {data_width});
      v_carry := '1';
    else
      v_result := resize(resize(unsigned(a), {data_width + 1}) - 1, {data_width});
      v_carry := '0';
    end if;

TESTBENCH EXPECTED VALUES (pre-computed, mathematically verified — use these EXACT values):
  ADD(0, 0) = 0
  SUB(0, 0) = 0
  MUL(0, 0) = 0
  NEG(0) = 0
  INC(0) = 1
  DEC(0) = {(0 - 1 + mod) % mod}
  ADD({max_val}, {max_val}) = {add_max}
  SUB({max_val}, {max_val}) = 0
  MUL({max_val}, {max_val}) = {mul_max}
  NEG({max_val}) = {neg_max}
  INC({max_val}) = {inc_max}
  DEC({max_val}) = {dec_max}
  ⚠️ Small-value tests (DO NOT compute these yourself — use these pre-verified values):
  ADD(10, 20) = 30
  SUB(20, 10) = 10
  SUB(10, 20) = {sub_10_20}
  MUL(10, 20) = {mul_10_20}
  NEG(10) = {neg_10}  ← this is {mod} - 10, NOT {mod} - 5 or any other value!
  NEG(1) = {neg_1}
  INC(10) = 11
  DEC(10) = 9

⚠️ MANDATORY TESTBENCH REQUIREMENT:
You MUST include a testbench entity mvl_alu_{k}_{bits}bit_tb with AT LEAST 20 assert/report test vectors. Code with fewer than 20 tests will be REJECTED.
Include:
- 6 edge-case tests (one per opcode with a=0, b=0)
- 6 max-value tests using the pre-computed expected values above
- 8+ additional tests with various values
⚠️ EACH test MUST explicitly assign BOTH a_sig AND b_sig before setting the opcode — even if the values
are the same as the previous test! Do NOT rely on previous test values. Changing only the opcode will
use stale a/b values and cause wrong results (e.g., SUB(10,20) gives a different result than SUB(20,10)).
⚠️ CRITICAL: Use the pre-computed expected values above — do NOT compute NEG/SUB/DEC results yourself!
  LLMs frequently miscalculate large subtractions. For example, NEG(10) = {neg_10}, NOT {neg_10 + 5} or {neg_10 - 5}.
CORRECT pattern for EVERY test:
  a_sig <= std_logic_vector(to_unsigned(VALUE_A, {data_width}));
  b_sig <= std_logic_vector(to_unsigned(VALUE_B, {data_width}));
  opcode_sig <= "XXXX";
  wait for 10 ns;
  assert result_sig = std_logic_vector(to_unsigned(EXPECTED, {data_width})) report "Test N: ..." severity error;
{"⚠️ INTEGER OVERFLOW WARNING: VHDL integer max = 2^31-1 = 2147483647." + chr(10) + "   ANY value > 2147483647 CANNOT use to_unsigned() — the integer argument overflows!" + chr(10) + "   This includes NEG results, SUB results where a < b, DEC(0), and max_val operands!" + chr(10) + "   Pre-computed hex literals (copy these EXACTLY):" + chr(10) + "   max_val " + str(max_val) + ": " + max_val_slv + chr(10) + "   DEC(0) = " + str((0-1+mod)%mod) + ": " + dec0_slv + chr(10) + "   ADD(max,max) = " + str(add_max) + ": " + add_max_slv + chr(10) + "   DEC(max) = " + str(dec_max) + ": " + dec_max_slv + chr(10) + "   NEG(10) = " + str(neg_10) + ": " + neg_10_slv + chr(10) + "   NEG(1) = " + str(neg_1) + ": " + neg_1_slv + chr(10) + "   SUB(10,20) = " + str(sub_10_20) + ": " + sub_10_20_slv + chr(10) + "   negative threshold " + str(neg_half) + ": " + neg_half_lit if mod > 2147483647 else ""}
Format: report "Test N: OP A=X B=Y -> R=Z"

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

        elif lang == 'vhdl':
            # Ensure IEEE library is present
            if 'library ieee' not in code.lower() and 'library IEEE' not in code:
                code = 'library IEEE;\nuse IEEE.STD_LOGIC_1164.ALL;\nuse IEEE.NUMERIC_STD.ALL;\n\n' + code

        return code

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
                max_tokens=8192,
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

    def _validate_code(self, code: str, language: str, k: int, bits: int, mod: int) -> List[str]:
        """Validate generated code quality. Returns a list of warning strings."""
        import math
        warnings = []
        lang = language.lower()
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1
        code_lower = code.lower()

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

        # 3. MOD value check
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
