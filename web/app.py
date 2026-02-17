"""
MVL Benchmark Generator - Flask API
===================================
Web API for generating and running MVL benchmarks.
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# 从环境变量读取配置
PORT = int(os.environ.get('PORT', 5001))
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# ============================================================
# 代理配置
# 除 DeepSeek 和 Qwen 外，其他 LLM API 需要代理访问
# 设置方式（选择一种）：
#   1. 环境变量: export HTTPS_PROXY=http://127.0.0.1:7890
#   2. 直接修改下方 PROXY_URL
# ============================================================
PROXY_URL = os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:10809')  # 例如: http://127.0.0.1:10809


def _setup_proxy():
    """设置代理环境变量，供 LLM providers 使用"""
    proxy = PROXY_URL
    if proxy:
        os.environ['HTTPS_PROXY'] = proxy
        os.environ['HTTP_PROXY'] = proxy
        print(f"🌐 Proxy configured: {proxy}")
        print("   (DeepSeek and Qwen will bypass proxy automatically)")
    else:
        print("⚠️  No proxy configured. Set HTTPS_PROXY env var if needed.")
        print("   Example: export HTTPS_PROXY=http://127.0.0.1:10809")


_setup_proxy()

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from mvl_generator import MVLGenerator
from mvl_simulation_runner import MVLSimulationRunner

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global instances
simulation_runner = MVLSimulationRunner(project_root=str(PROJECT_ROOT))


# ============================================================
# API Endpoints
# ============================================================

@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory('templates', 'index.html')


@app.route('/api/status')
def api_status():
    """Get system status"""
    return jsonify({
        'status': 'ok',
        'tools': simulation_runner.get_tools_status(),
        'llm_providers': [
            'gemini', 'openai', 'gpt', 'claude', 'groq',
            'deepseek', 'qwen', 'mistral', 'together', 'grok', 'local'
        ]
    })


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate MVL ALU code"""
    try:
        data = request.json

        llm_provider = data.get('llm', 'groq')
        model = data.get('model', None)
        k_value = int(data.get('k_value', 3))
        bitwidth = int(data.get('bitwidth', 8))
        language = data.get('language', 'c')
        operations = data.get('operations', ['ADD', 'SUB', 'MUL', 'NEG', 'INC', 'DEC'])

        # Validate parameters
        if k_value < 2 or k_value > 8:
            return jsonify({'success': False, 'error': 'K-value must be between 2 and 8'}), 400

        if bitwidth not in [8, 10, 12, 14]:
            return jsonify({'success': False, 'error': 'Bitwidth must be 8, 10, 12, or 14'}), 400

        if language not in ['c', 'python', 'verilog', 'vhdl']:
            return jsonify({'success': False, 'error': 'Language must be c, python, verilog, or vhdl'}), 400

        # Create generator and generate
        print(f"\n📋 [API] Generate request:")
        print(f"   Requested provider: {llm_provider}")
        print(f"   Requested model: {model or '(default)'}")
        print(f"   Parameters: k={k_value}, bitwidth={bitwidth}, lang={language}")

        generator = MVLGenerator(
            llm_provider=llm_provider,
            model=model,
            project_root=str(PROJECT_ROOT)
        )

        # Log actual provider/model after initialization
        actual_model = getattr(generator.llm, 'model', 'N/A') if generator.llm else 'N/A'
        actual_class = type(generator.llm).__name__ if generator.llm else 'None'
        print(f"   Actual provider: {generator.llm_provider_name} ({actual_class})")
        print(f"   Actual model: {actual_model}")

        result = generator.generate(
            k_value=k_value,
            bitwidth=bitwidth,
            language=language,
            operations=operations
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run-simulation', methods=['POST'])
def api_run_simulation():
    """Run simulation for generated code"""
    try:
        data = request.json
        file_path = data.get('file_path')
        language = data.get('language')

        if not file_path:
            return jsonify({'success': False, 'error': 'file_path is required'}), 400

        # Make path absolute
        full_path = PROJECT_ROOT / file_path

        result = simulation_runner.run_simulation(str(full_path), language)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/check-tools')
def api_check_tools():
    """Check available simulation tools"""
    return jsonify(simulation_runner.get_tools_status())


@app.route('/api/download/<path:filepath>')
def api_download(filepath):
    """Download generated file"""
    filepath = filepath.replace('\\', '/')
    file_path = PROJECT_ROOT / filepath

    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(
        str(file_path.parent.absolute()),
        file_path.name,
        as_attachment=True
    )


@app.route('/api/view-code/<path:filepath>')
def api_view_code(filepath):
    """View code file content"""
    filepath = filepath.replace('\\', '/')
    file_path = PROJECT_ROOT / filepath

    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({
            'success': True,
            'filename': file_path.name,
            'content': content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# Static Files
# ============================================================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🔢 MVL Benchmark Generator")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Port: {PORT}")
    print(f"Debug: {DEBUG}")
    print(f"Tools: {simulation_runner.get_tools_status()}")
    print("=" * 60)

    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)