# EarlyBird – Semantic Project Architecture

Automated application core synthesis tool that derives cohesive software architecture components for the **EarlyBird Management System (EBMS)** directly from the high-dimensional vector space topology of its functional requirements.

Includes an embedded **Axum REST API** and interactive **Visual Workbench** (built with Tailwind, Apache ECharts, and Mermaid.js), running with zero external web runtimes on both Windows and CachyOS Linux (with AMD Radeon RX 9070 XT ROCm / Vulkan acceleration).

---

## 🚀 Quick Start

### On Windows
Double-click `run.bat` or run:
```cmd
run.bat
```
This automatically starts the Qdrant vector database (via Docker if installed) and launches the interactive visual workbench on **http://localhost:3030**.

### On Linux (CachyOS / Arch / Ubuntu)
Make executable and run:
```bash
chmod +x run.sh
./run.sh
```
*(On CachyOS with an AMD Radeon RX 9070 XT, it detects `/opt/rocm` and enables ROCm acceleration).*

---

## 🌟 Key Features

1. **Instruction-Tuned Semantic Embeddings:**
   - Powered by **Qwen3-Embedding-0.6B** (1024 dimensions) and **BGE-Small-EN-v1.5** (384 dimensions) via Rust Candle / FastEmbed.
   - Applies an architectural domain responsibility instruction prompt to avoid syntax-only clustering.
2. **Qdrant Vector Database Integration:**
   - Ingests requirement vectors and executes server-side HNSW distance matrix extraction via gRPC.
   - Includes automatic fallback to in-memory vector topology if Docker is offline.
3. **Dual Clustering Verification:**
   - **Primary:** Louvain Modularity Community Detection on Qdrant's HNSW graph ($k$-NN similarity topology).
   - **Verification:** Average-Linkage Hierarchical Agglomerative Clustering ($k=3..14$).
   - Calculates Silhouette Score and Adjusted Rand Index (ARI).
4. **Algorithmic Component Naming (No Local LLM Required):**
   - Class-based TF-IDF (c-TF-IDF) domain keyword extraction.
   - Most representative medoid requirement discovery.
   - Cosine centroid projection against architectural candidate roles.
5. **Interactive 4-Tab Workbench:**
   - **Tab 1 (Requirements & Ingestion):** Manage 44 normalized requirements or paste custom inputs; model picker.
   - **Tab 2 (2D Semantic Topology):** Full-screen ECharts 2D PCA projection scatter canvas, real-time threshold & $k$ tuning sliders with **sub-50ms live re-clustering**.
   - **Tab 3 (Discovered Components):** Card deck detailing proposed components, cohesion metrics, keywords, and allocated requirements.
   - **Tab 4 (Diagrams & Export):** Live interactive Mermaid component diagram, Markdown specification preview, and download buttons for `components.md` and `clusters.svg`.

---

## 📂 Repository Structure

```
SemanticProjectArchitecture/
├── earlybird-arch/              # Pure Rust application core
│   ├── src/
│   │   ├── cluster/             # Louvain graph, agglomerative & metrics
│   │   ├── embed.rs             # Candle / FastEmbed vector inference
│   │   ├── store.rs             # Qdrant vector database gRPC client
│   │   ├── label.rs             # c-TF-IDF keywords & medoid candidate labeling
│   │   ├── report.rs            # Markdown specification & SVG PCA renderer
│   │   ├── server.rs            # Axum 0.8 REST API & rust-embed handler
│   │   └── main.rs              # CLI entrypoint (embed, ingest, cluster, run, serve)
│   ├── frontend/dist/           # Embedded Single-Page Visual Workbench
│   └── Cargo.toml               # Rust dependency definitions
├── data/
│   ├── requirements.yaml        # 44 normalized requirements & domain rules
│   └── requirements.raw.yaml    # Original extracted statements
├── out/                         # Precomputed artifacts
│   ├── components.md            # Generated architecture proposal report
│   ├── clusters.svg             # 2D PCA scatter visualization SVG
│   ├── embeddings.qwen3-0.6b.json # Precomputed 1024-d vectors
│   ├── clusters.qwen3-0.6b.json   # Proposed component definitions
│   └── candidate_embeddings.qwen3-0.6b.json
├── docker-compose.yml           # Local Qdrant container definition
├── run.bat                      # Windows one-click launcher
├── run.sh                       # Linux launcher (ROCm / AMD GPU ready)
├── requirements_normalized.md   # Traceability mapping against PDF
└── Case Study Early Bird Requirements V150.pdf
```

---

## 🛠️ CLI Pipeline Commands

You can also run the pipeline from the command line without the web UI:

```bash
cd earlybird-arch

# 1. Run complete end-to-end pipeline (Embed -> Ingest -> Cluster -> Label -> Report)
cargo run --release -- run --model qwen3-0.6b

# 2. Launch workbench server
cargo run --release -- serve --port 3030

# 3. Fast agglomerative clustering check
cargo run --release -- cluster --method agglo --k 8
```

---

## ⚡ AMD Radeon RX 9070 XT on CachyOS Linux

To run larger models like **Qwen3-Embedding-8B** (4096 dimensions) on AMD ROCm:

```bash
# 1. Install ROCm SDK on CachyOS
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk

# 2. Launch workbench with GPU acceleration
./run.sh
```
