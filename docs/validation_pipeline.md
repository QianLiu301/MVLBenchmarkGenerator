# MVL Benchmark Validation Pipeline

## Architecture Overview

The validation pipeline consists of **two completely independent components**:

1. **LLM-based Code Generation** — generates MVL ALU implementations
2. **Deterministic Golden Model Validation** — mathematically verifies correctness post-simulation

The Golden Model is a **standalone mathematical reference** with no dependency on any LLM. It computes expected results using Galois Field arithmetic (GF(p), GF(p^n)) or modular arithmetic (Z/kZ), serving as the ground truth for validation.

## Pipeline Flowchart

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: LLM Code Generation"]
        A["User Input\n(k-value, bitwidth, language, operations)"] --> B["Prompt Construction"]
        B --> C["LLM Provider\n(Gemini / OpenAI / DeepSeek / ...)"]
        C --> D["Generated MVL ALU Code\n(C / Python / Verilog / VHDL)"]
    end

    subgraph Phase2["Phase 2: Simulation"]
        D --> E["Compiler / Simulator\n(GCC / Python / Iverilog / GHDL)"]
        E --> F["Simulation Output\n(test results from stdout)"]
    end

    subgraph Phase3["Phase 3: Independent Validation"]
        direction TB
        G["Golden Model\n(Deterministic Math Reference)"]
        H["Galois Field Arithmetic\nGF(p), GF(p^n), Z/kZ"]
        H --> G
        G --> I["Expected Results\n(result, zero/neg/carry flags)"]
    end

    F --> J{"Comparator"}
    I --> J

    J -->|Match| K["PASS"]
    J -->|Mismatch| L["Error Classification\n(logic error, flag error, etc.)"]

    style Phase1 fill:#e3f2fd,stroke:#1565c0,color:#000
    style Phase2 fill:#fff3e0,stroke:#e65100,color:#000
    style Phase3 fill:#e8f5e9,stroke:#2e7d32,color:#000
    style G fill:#c8e6c9,stroke:#2e7d32,color:#000
    style C fill:#bbdefb,stroke:#1565c0,color:#000
    style J fill:#fff9c4,stroke:#f57f17,color:#000
```

## Key Point: Independence

```
┌─────────────────────────┐     ┌─────────────────────────────┐
│   LLM (Black Box)       │     │   Golden Model (White Box)  │
│                         │     │                             │
│  - Non-deterministic    │     │  - Fully deterministic      │
│  - Prompt-dependent     │     │  - Math-based (GF algebra)  │
│  - Provider-variable    │     │  - No LLM dependency        │
│  - Generates code       │     │  - Computes ground truth    │
│                         │     │                             │
│  Output: ALU code       │     │  Output: Expected values    │
└──────────┬──────────────┘     └──────────────┬──────────────┘
           │                                    │
           ▼                                    ▼
    ┌──────────────┐                  ┌──────────────────┐
    │  Simulation  │                  │  Test Vectors    │
    │  (compile &  │                  │  (deterministic  │
    │   execute)   │                  │   computation)   │
    └──────┬───────┘                  └────────┬─────────┘
           │                                    │
           └──────────────┬─────────────────────┘
                          ▼
                 ┌─────────────────┐
                 │   Comparison    │
                 │  (automated     │
                 │   validation)   │
                 └─────────────────┘
```

The two paths are **completely independent** and only converge at the final comparison step. The Golden Model does not use, call, or depend on any LLM. It is a pure mathematical computation based on well-defined algebraic structures.
