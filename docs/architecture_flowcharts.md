# Architecture Flowcharts

This page provides copy-ready Mermaid diagrams capturing the core architecture and flows of the ECG Diagnosis system in this repository.

- Repository components referenced: `src/ci/preprocess/`, `src/ci/model/`, `src/ci/fuzzy/`, `src/ci/evaluation/`, `src/ci/deployment/`, `deploy_package/rest_api_wrapper.py`, `ifog_simulation/ECGDiagnosisSimulation.java`, and `scripts/deployment_simulation.py`.
- You can view these diagrams directly on GitHub/GitLab or in VS Code with Mermaid support. For export, use Mermaid CLI or an online Mermaid renderer.

## 1) End-to-end ECG diagnosis pipeline
References: `README.md`, `src/ci/preprocess/`, `src/ci/model/`, `src/ci/fuzzy/`, `src/ci/evaluation/`

```mermaid
flowchart TD
  A[ECG Input\n30s @ 360Hz] --> B[Sliding Window\n10s, 50% overlap]
  B --> C[Preprocessing\nSTFT -> Mel spectrogram\n(64 mels)]
  C --> D[Feature Extraction\nResNet18-based\n(Pytorch or ONNX INT8)]
  D --> E[Aggregation + Classification\n(512-d embeddings)]
  E --> F[Fuzzy Decision Engine\nMamdani + centroid\nPSO-optimized MFs]
  F --> G[Diagnosis + Risk Score\nTop-1 class, confidence, risk level]

  classDef s fill:#eef,stroke:#88f
  classDef p fill:#efe,stroke:#4a4
  classDef m fill:#ffe,stroke:#aa4
  classDef f fill:#fee,stroke:#a44

  class A s
  class B,C p
  class D,E m
  class F f
```

## 2) Deployment tiers and integration (Edge/Fog/Cloud + REST + iFogSim)
References: `README.md` (Edge Deployment Architecture), `deploy_package/rest_api_wrapper.py`, `ifog_simulation/ECGDiagnosisSimulation.java`, `docs/JAVA_IFOGSIM_INTEGRATION.md`

```mermaid
flowchart LR
  subgraph Sensor
    S[ECG Sensor\nData acquisition]
  end

  subgraph Edge_Gateway
    P[Preprocessing\nSpectrograms]
    FX[Feature Extraction\nONNX Runtime (INT8)]
    FU[Fuzzy Logic\nRisk assessment]
    API[Flask REST API\n/diagnose, /batch_diagnose]
  end

  subgraph Fog_Node
    FXF[Feature Extraction\n(PyTorch/ONNX)]
    FUF[Fuzzy / Aggregation]
  end

  subgraph Cloud_Server
    AN[Analytics & Storage\nModel registry, reports]
  end

  subgraph Simulation_Tools
    SIM[Python Deployment Simulator\nscripts/deployment_simulation.py]
    IFS[iFogSim Java\nECGDiagnosisSimulation.java]
  end

  S -- ECG stream --> P
  P --> FX
  FX --> FU
  FU --> API
  API --> AN

  P -. optional offload .-> FXF
  FXF -.-> FUF
  FUF -.-> AN

  IFS -. HTTP POST /diagnose .-> API
  SIM --> AN

  classDef t fill:#eef,stroke:#55f
  classDef e fill:#efe,stroke:#4a4
  classDef f fill:#ffd,stroke:#aa4
  classDef c fill:#fee,stroke:#a44
  classDef s fill:#f0f0f0,stroke:#888

  class S t
  class P,FX,FU,API e
  class FXF,FUF f
  class AN c
  class SIM,IFS s
```

## 3) Edge conversion (E6) pipeline: export, quantize, parity, benchmark
References: `demo/demo_e6.py`, `scripts/edge_convert.py`, `src/ci/deployment/onnx_utils.py`, `docs/E6_IMPLEMENTATION_STATUS.md`

```mermaid
flowchart LR
  PT[PyTorch Model\nResNet18 FeatureExtractor] --> EX[Export to ONNX\nexport_pytorch_to_onnx()]
  EX --> CK[Validate ONNX\nchecker + ORT load]
  EX --> PAR[Parity Check\ncompare_pytorch_vs_onnx()]
  EX --> Q[Dynamic INT8 Quantization\nquantize_onnx_dynamic_model()]
  PT --> BPT[Benchmark PyTorch\nbenchmark_model()]
  Q --> BONNX[Benchmark ONNX\nbenchmark_onnx()]
  CK --> SUM[Summary & Artifacts]
  PAR --> SUM
  BPT --> SUM
  BONNX --> SUM

  classDef n fill:#eef,stroke:#55f
  classDef a fill:#efe,stroke:#4a4
  classDef b fill:#ffd,stroke:#aa4

  class PT n
  class EX,Q a
  class CK,PAR,BPT,BONNX,SUM b
```

---

### Viewing tips

- GitHub and modern IDEs render Mermaid automatically in `.md` files.
- For offline export to PNG/SVG, you can use Mermaid CLI:

```bash
# install mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# export an individual diagram (example: pipeline)
mmdc -i docs/architecture_flowcharts.md -o pipeline.png
```
