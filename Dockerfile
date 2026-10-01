# ============================================================
# MVL Benchmark Generator - Dockerfile
# ============================================================
# Supports: GCC (C), Python, Icarus Verilog, GHDL, SystemC simulation tools
# ============================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including simulation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gcc \
    g++ \
    make \
    # Verilog simulation
    iverilog \
    # VHDL simulation (Debian ghdl package; the library verifies .vhd submissions with it)
    ghdl \
    # SystemC simulation: headers and library, compiled against with g++ -lsystemc
    libsystemc-dev \
    # Utilities
    curl \
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify simulation tools installation
RUN echo "=== Checking installed tools ===" && \
    gcc --version && \
    python3 --version && \
    iverilog -V &&     ghdl --version && \
    echo "=== All tools installed successfully ==="

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# SystemC has no version command, so prove it works with the application's own
# probe: it builds a small model with the port types of an entry, runs it and
# checks the result, trying each C++ standard and link method. A two-line
# program is not enough -- it linked where real models did not. A failure is
# reported loudly in the build log but does not stop the build: the rest of the
# application deploys, and SystemC simulations report "unavailable" with the
# probe's reason until it is fixed.
RUN python -c "import sys; sys.path.insert(0, 'src'); \
from mvl_simulation_runner import MVLSimulationRunner as R; \
s = R('/app')._systemc_status(); print('SystemC probe:', s); \
print('SystemC OK' if s['ok'] else '*** WARNING: SystemC unavailable: ' + s['reason'])"

# SystemVerilog submissions are compiled by Icarus in IEEE 1800-2012 mode. Old
# Icarus releases (10.x) reject always_ff/always_comb, so prove the installed
# one accepts them; like SystemC, a failure is reported but does not stop the build.
RUN printf 'module t; logic [3:0] x = 0; logic clk = 0;\n always_ff @(posedge clk) x <= x + 1;\n always_comb begin end\n initial begin $display("SystemVerilog OK"); $finish; end\nendmodule\n' > /tmp/sv_probe.sv && \
    (iverilog -g2012 -o /tmp/sv_probe.vvp /tmp/sv_probe.sv && vvp /tmp/sv_probe.vvp) || \
    echo "*** WARNING: iverilog cannot compile SystemVerilog (-g2012)"; \
    rm -f /tmp/sv_probe.sv /tmp/sv_probe.vvp

# Create output directories
RUN mkdir -p /app/output/mvl_code/gemini \
             /app/output/mvl_code/mistral \
             /app/output/mvl_code/deepseek \
             /app/output/mvl_code/openai \
             /app/output/mvl_code/qwen \
             /app/output/mvl_code/gptoss \
             /app/output/mvl_code/glm \
             /app/output/mvl_code/together \
             /app/output/mvl_results \
    && chmod -R 777 /app/output

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/api/status || exit 1

# Run the application
CMD ["python", "web/app.py"]