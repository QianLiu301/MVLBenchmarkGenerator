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
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.llm = self._setup_llm()
        self.output_dir = self._setup_output_dir(output_dir, project_root)

        print(f"🔢 MVL Generator initialized")
        print(f"   LLM Provider: {self.llm_provider_name}")
        print(f"   Model: {self.model or 'default'}")
        print(f"   Output directory: {self.output_dir}")

    def _setup_llm(self):
        """Setup LLM provider, loading API keys from llm_config.json if available"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from llm_providers import LLMFactory, LLMConfig

            # Load API key from llm_config.json (search in project root)
            kwargs = {}
            provider_config = {}
            try:
                config_path = self.project_root / "llm_config.json"
                config = LLMConfig(config_file=str(config_path))
                provider_config = config.config.get(self.llm_provider_name, {})
                if isinstance(provider_config, dict) and provider_config.get("api_key"):
                    kwargs["api_key"] = provider_config["api_key"]
                    print(f"   🔑 API key loaded from {config_path}")
            except Exception as e:
                print(f"   ⚠️  Could not load llm_config.json: {e}")

            # Use the model from frontend selection, or fall back to config/default
            if self.model:
                kwargs["model"] = self.model
            elif isinstance(provider_config, dict) and provider_config.get("model"):
                kwargs["model"] = provider_config["model"]

            llm = LLMFactory.create_provider(self.llm_provider_name, **kwargs)

            # Log the actual model being used
            if hasattr(llm, 'model'):
                print(f"   🤖 Actual model in use: {llm.model}")

            return llm

        except ImportError as e:
            print(f"❌ Failed to import LLM providers: {e}")
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

    def generate(
            self,
            k_value: int = 3,
            bitwidth: int = 8,
            language: str = 'c',
            operations: List[str] = None
    ) -> Dict:
        """
        Generate MVL ALU code.

        Args:
            k_value: Number of logic values (2-8)
            bitwidth: Number of digits (8, 10, 12, 14)
            language: Output language ('c', 'python', 'verilog', 'vhdl')
            operations: List of operations to include

        Returns:
            Dict with generation results
        """
        if operations is None:
            operations = ['ADD', 'SUB', 'MUL', 'NEG', 'INC', 'DEC']

        print(f"\n{'=' * 60}")
        print(f"🔢 Generating MVL ALU")
        print(f"   K-value: {k_value} (GF({k_value}))")
        print(f"   Bitwidth: {bitwidth}-trit")
        print(f"   Language: {language.upper()}")
        print(f"   Operations: {', '.join(operations)}")
        print(f"   LLM: {self.llm_provider_name.upper()}")
        print(f"{'=' * 60}")

        # Calculate MOD value
        mod_value = k_value ** bitwidth

        # Create prompt based on language
        if language.lower() == 'c':
            prompt = self._create_c_prompt(k_value, bitwidth, mod_value, operations)
        elif language.lower() == 'python':
            prompt = self._create_python_prompt(k_value, bitwidth, mod_value, operations)
        elif language.lower() == 'verilog':
            prompt = self._create_verilog_prompt(k_value, bitwidth, mod_value, operations)
        else:
            return {'success': False, 'error': f'Unsupported language: {language}'}

        # Call LLM
        print(f"\n🤖 Calling {self.llm_provider_name.upper()} API...")

        try:
            if hasattr(self.llm, '_call_api'):
                response = self.llm._call_api(
                    prompt,
                    max_tokens=4096,
                    system_prompt="You are an expert programmer. Generate clean, compilable code without any explanations."
                )
            else:
                return {'success': False, 'error': 'LLM provider does not support _call_api'}

            print(f"✅ LLM response received ({len(response)} chars)")

            # Extract code
            code = self._extract_code(response, language)

            if not code:
                return {'success': False, 'error': 'Failed to extract code from response'}

            # Fix common issues
            code = self._fix_code(code, language)

            # Save file
            file_path = self._save_code(code, k_value, bitwidth, language)

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

    def _save_code(self, code: str, k: int, bits: int, language: str) -> Path:
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

        filename = f"mvl_alu_gf{k}_{bits}bit_{timestamp}.{ext}"
        file_path = self.output_dir / filename

        # Add header comment
        if language.lower() == 'c':
            header = f"""/*
 * MVL ALU Benchmark - GF({k}) {bits}-trit
 * Generated by: MVL Benchmark Generator
 * LLM Provider: {self.llm_provider_name}
 * Generated at: {timestamp}
 * MOD value: {k ** bits}
 */

"""
        elif language.lower() == 'python':
            header = f'''"""
MVL ALU Benchmark - GF({k}) {bits}-trit
Generated by: MVL Benchmark Generator
LLM Provider: {self.llm_provider_name}
Generated at: {timestamp}
MOD value: {k ** bits}
"""

'''
        else:
            header = f"""// MVL ALU Benchmark - GF({k}) {bits}-trit
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
