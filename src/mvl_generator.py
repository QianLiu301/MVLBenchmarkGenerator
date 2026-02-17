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

    def generate(
            self,
            k_value: int = 3,
            bitwidth: int = 8,
            language: str = 'c',
            operations: List[str] = None,
            natural_input: str = ''
    ) -> Dict:
        """
        Generate MVL ALU code.

        Args:
            k_value: Number of logic values (2-8)
            bitwidth: Number of digits (8, 10, 12, 14)
            language: Output language ('c', 'python', 'verilog', 'vhdl')
            operations: List of operations to include
            natural_input: Natural language description (takes priority for prompt/naming when provided)

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

        actual_model = getattr(self.llm, 'model', 'N/A') if self.llm else 'N/A'
        actual_class = type(self.llm).__name__ if self.llm else 'None'
        print(f"\n{'=' * 60}")
        print(f"🔢 Generating MVL ALU")
        if natural_input:
            print(f"   Natural input: {natural_input}")
        print(f"   K-value: {k_value} (GF({k_value}))")
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
            prompt = self._create_natural_prompt(natural_input, k_value, bitwidth, mod_value, operations, language)
        elif language.lower() == 'c':
            prompt = self._create_c_prompt(k_value, bitwidth, mod_value, operations)
        elif language.lower() == 'python':
            prompt = self._create_python_prompt(k_value, bitwidth, mod_value, operations)
        elif language.lower() == 'verilog':
            prompt = self._create_verilog_prompt(k_value, bitwidth, mod_value, operations)
        elif language.lower() == 'vhdl':
            prompt = self._create_vhdl_prompt(k_value, bitwidth, mod_value, operations)
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
                max_tokens=4096,
                system_prompt="You are an expert programmer. Generate clean, compilable code without any explanations."
            )

            print(f"✅ LLM response received ({len(response)} chars)")

            # Extract code
            code = self._extract_code(response, language)

            if not code:
                return {'success': False, 'error': 'Failed to extract code from response'}

            # Fix common issues
            code = self._fix_code(code, language)

            # Save file
            file_path = self._save_code(code, k_value, bitwidth, language, natural_input=natural_input)

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
                'llm': self.llm_provider_name
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
            natural_input: str = ''
    ):
        """
        Generate MVL ALU code with streaming output.
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

        mod_value = k_value ** bitwidth

        # Create prompt: use natural_input when provided
        if natural_input:
            prompt = self._create_natural_prompt(natural_input, k_value, bitwidth, mod_value, operations, language)
        elif language.lower() == 'c':
            prompt = self._create_c_prompt(k_value, bitwidth, mod_value, operations)
        elif language.lower() == 'python':
            prompt = self._create_python_prompt(k_value, bitwidth, mod_value, operations)
        elif language.lower() == 'verilog':
            prompt = self._create_verilog_prompt(k_value, bitwidth, mod_value, operations)
        elif language.lower() == 'vhdl':
            prompt = self._create_vhdl_prompt(k_value, bitwidth, mod_value, operations)
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
                                   operations=operations, natural_input=natural_input)
            if result.get('success') and result.get('code'):
                yield ("chunk", result['code'])
            yield ("done", result)
            return

        try:
            full_response = ""
            system_prompt = "You are an expert programmer. Generate clean, compilable code without any explanations."

            for chunk in self.llm._call_api_stream(prompt, max_tokens=4096, system_prompt=system_prompt):
                full_response += chunk
                yield ("chunk", chunk)

            print(f"✅ LLM streaming response received ({len(full_response)} chars)")

            # Process the full response
            code = self._extract_code(full_response, language)
            if not code:
                yield ("error", 'Failed to extract code from response')
                return

            code = self._fix_code(code, language)
            file_path = self._save_code(code, k_value, bitwidth, language, natural_input=natural_input)

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
                'llm': self.llm_provider_name
            })

        except Exception as e:
            print(f"❌ Streaming generation failed: {e}")
            import traceback
            traceback.print_exc()
            yield ("error", str(e))

    def _create_natural_prompt(self, natural_input: str, k: int, bits: int, mod: int,
                                 operations: List[str], language: str) -> str:
        """Create prompt based on natural language input, supplemented with structured parameters."""
        ops_str = ', '.join(operations)

        lang_instructions = {
            'c': 'Output ONLY C code. Must be complete and compilable with gcc. Include main() with test cases.',
            'python': 'Output ONLY Python code. Must be complete and runnable with python3. Include test cases at the bottom.',
            'verilog': 'Output ONLY Verilog code. Must be synthesizable. Include a testbench module.',
            'vhdl': 'Output ONLY VHDL code. Must be synthesizable. Include a testbench.'
        }

        prompt = f"""Based on the following description, generate a complete implementation:

DESCRIPTION: {natural_input}

CONTEXT PARAMETERS (use these to fill in any details not specified in the description):
- K-value: {k} (operations are mod {k})
- Bitwidth: {bits} trits
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}
- Output language: {language.upper()}

CRITICAL RULES:
1. {lang_instructions.get(language.lower(), lang_instructions['c'])}
2. No markdown, no explanations, only code
3. The description above takes priority over the context parameters

Generate the complete code now:
"""
        return prompt

    def _create_c_prompt(self, k: int, bits: int, mod: int, operations: List[str]) -> str:
        """Create prompt for C code generation"""
        ops_str = ', '.join(operations)

        prompt = f"""Generate a complete, compilable C program for a {bits}-trit ALU over GF({k}).

CRITICAL RULES:
1. Output ONLY C code, no markdown, no explanations
2. Must be complete and compilable with gcc
3. Include main() with 20 random test vectors

SPECIFICATIONS:
- K-value: {k} (operations are mod {k})
- Bitwidth: {bits} trits
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}
- Each operand range: 0 to {mod - 1}

REQUIRED STRUCTURE:
1. #include statements (stdio.h, stdlib.h, stdint.h, stdbool.h, time.h)
2. Operation enum (OP_ADD=0, OP_SUB=1, OP_MUL=2, OP_NEG=3, OP_INC=4, OP_DEC=5)
3. #define MOD {mod}
4. Flags struct (Z: zero, N: negative, C: carry)
5. mod{mod}() helper function for negative number handling
6. alu_exec() function with switch-case for operations
7. op_name() function to get operation name string
8. main() with srand(), 20 random tests, printf results

EXAMPLE OUTPUT FORMAT IN main():
printf("Test %2d: %-3s A=%d B=%d -> R=%d Z=%d N=%d C=%d\\n", ...);

Generate the complete C code now:
"""
        return prompt

    def _create_python_prompt(self, k: int, bits: int, mod: int, operations: List[str]) -> str:
        """Create prompt for Python code generation"""
        ops_str = ', '.join(operations)

        prompt = f"""Generate a complete Python program for a {bits}-trit ALU over GF({k}).

CRITICAL RULES:
1. Output ONLY Python code, no markdown, no explanations
2. Must be complete and runnable with python3
3. Include main section with 20 random test vectors

SPECIFICATIONS:
- K-value: {k}
- Bitwidth: {bits} trits
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}
- Each operand range: 0 to {mod - 1}

REQUIRED STRUCTURE:
1. imports (random, dataclasses or namedtuple)
2. MOD = {mod}
3. Operation constants (OP_ADD=0, OP_SUB=1, etc.)
4. Flags dataclass or namedtuple (z, n, c)
5. alu_exec(a, b, op) function returning (result, flags)
6. op_name(op) function
7. if __name__ == "__main__": block with 20 random tests

Generate the complete Python code now:
"""
        return prompt

    def _create_verilog_prompt(self, k: int, bits: int, mod: int, operations: List[str]) -> str:
        """Create prompt for Verilog code generation"""
        ops_str = ', '.join(operations)

        # Calculate bit width needed to represent mod value
        import math
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1

        prompt = f"""Generate a complete Verilog module for a {bits}-trit ALU over GF({k}).

CRITICAL RULES:
1. Output ONLY Verilog code, no markdown, no explanations
2. Must be synthesizable and simulatable with iverilog
3. Use mod {mod} for all arithmetic operations

SPECIFICATIONS:
- K-value: {k}
- Bitwidth: {bits} trits (need {data_width} bits to represent 0 to {mod - 1})
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}

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

OPCODES:
- 4'b0000: ADD (a + b) mod {mod}
- 4'b0001: SUB (a - b) mod {mod}
- 4'b0010: MUL (a * b) mod {mod}
- 4'b0011: NEG (-a) mod {mod}
- 4'b0100: INC (a + 1) mod {mod}
- 4'b0101: DEC (a - 1) mod {mod}

Generate the complete Verilog module now:
"""
        return prompt

    def _create_vhdl_prompt(self, k: int, bits: int, mod: int, operations: List[str]) -> str:
        """Create prompt for VHDL code generation"""
        ops_str = ', '.join(operations)

        import math
        data_width = math.ceil(math.log2(mod)) if mod > 1 else 1

        prompt = f"""Generate a complete VHDL design for a {bits}-trit ALU over GF({k}).

CRITICAL RULES:
1. Output ONLY VHDL code, no markdown, no explanations
2. Must be synthesizable and simulatable with GHDL
3. Use mod {mod} for all arithmetic operations
4. Include BOTH the entity/architecture AND a testbench

SPECIFICATIONS:
- K-value: {k}
- Bitwidth: {bits} trits (need {data_width} bits to represent 0 to {mod - 1})
- MOD value: {mod} ({k}^{bits})
- Operations: {ops_str}

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

OPCODES:
- "0000": ADD (a + b) mod {mod}
- "0001": SUB (a - b) mod {mod}
- "0010": MUL (a * b) mod {mod}
- "0011": NEG (-a) mod {mod}
- "0100": INC (a + 1) mod {mod}
- "0101": DEC (a - 1) mod {mod}

TESTBENCH REQUIREMENTS:
- Entity: mvl_alu_{k}_{bits}bit_tb
- Generate at least 20 test vectors
- Use assert or report statements to display results
- Format: report "Test N: OP A=X B=Y -> R=Z"

Generate the complete VHDL code (entity + architecture + testbench) now:
"""
        return prompt

    def _extract_code(self, response: str, language: str) -> Optional[str]:
        """Extract code from LLM response"""
        # Remove markdown code blocks
        code = response

        # Try to extract from code blocks
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

        # Remove any remaining markdown
        code = re.sub(r'^```\w*\s*', '', code)
        code = re.sub(r'\s*```$', '', code)

        return code if code else None

    def _fix_code(self, code: str, language: str) -> str:
        """Fix common code issues"""
        if language.lower() == 'c':
            # Ensure includes are present
            if '#include' not in code:
                includes = """#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

"""
                code = includes + code

        elif language.lower() == 'python':
            # Ensure imports
            if 'import random' not in code:
                code = 'import random\n' + code

        return code

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

    def _save_code(self, code: str, k: int, bits: int, language: str, natural_input: str = '') -> Path:
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

        # When natural_input is provided, use it for the filename
        if natural_input:
            safe_name = self._sanitize_filename(natural_input)
            filename = f"{safe_name}_{timestamp}.{ext}"
            description = natural_input
        else:
            filename = f"mvl_alu_gf{k}_{bits}bit_{timestamp}.{ext}"
            description = f"GF({k}) {bits}-trit"

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
