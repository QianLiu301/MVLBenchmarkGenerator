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

# SystemC has no version command, so prove it works: compile, link and run a
# minimal model. A broken install then fails the build here instead of every
# SystemC simulation reporting "unavailable" once deployed.
RUN printf '#include <systemc.h>\nint sc_main(int, char*[]) { return 0; }\n' > /tmp/sc_probe.cpp && \
    g++ -std=c++17 /tmp/sc_probe.cpp -o /tmp/sc_probe -lsystemc -lpthread && \
    SC_COPYRIGHT_MESSAGE=DISABLE /tmp/sc_probe && \
    rm -f /tmp/sc_probe /tmp/sc_probe.cpp && \
    echo "=== SystemC compiles, links and runs ==="

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

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