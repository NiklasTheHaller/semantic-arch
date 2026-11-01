# Semantic Project Architecture – Implementation Roadmap & TODOs

> **Goal:** Extend the EarlyBird Semantic Architecture tool from a CLI pipeline into an interactive, cross-platform visual workbench running on both Windows and CachyOS Linux (with AMD Radeon RX 9070 XT GPU acceleration).

---

## Phase 1: Embedded Web Server & REST API Backend

Create an HTTP API in `earlybird-arch` using Axum so the engine can power an interactive UI.

- [x] Add `axum`, `tower-http`, and `rust-embed` dependencies to `earlybird-arch/Cargo.toml`.
- [x] Implement `earlybird-arch serve [--port 3030]` CLI subcommand.
- [x] Create endpoint `GET /api/system`:
  - [x] Return detected OS (Windows / Linux).
  - [x] Return detected hardware compute (CPU vs AMD GPU / ROCm / Vulkan).
  - [x] Check and report Qdrant database connectivity.
- [x] Create endpoint `GET /api/models`:
  - [x] Return available models (`qwen3-0.6b`, `qwen3-8b`, `bge-small`) and their download/cache status.
- [x] Create endpoint `GET /api/pipeline/current`:
  - [x] Return current 2D PCA projected points, clusters, components, and silhouette metrics.
- [x] Create endpoint `POST /api/pipeline/run`:
  - [x] Accept custom requirements input (raw text, YAML, or Markdown).
  - [x] Accept model selection and instruction prompt toggle.
  - [x] Accept clustering method (`graph` / `agglomerative`) and parameters (similarity threshold, target $k$).
  - [x] Sub-50ms fast re-clustering on precomputed vectors.
  - [x] Return complete result: cluster assignments, 2D coordinates, keywords, medoids, cohesion, and silhouette scores.
- [x] Create endpoints `GET /api/export/markdown` and `GET /api/export/svg`:
  - [x] Stream `components.md` and `clusters.svg` downloads.

---

## Phase 2: Interactive 2D Visual Frontend (Embedded SPA)

Build a clean, responsive single-page web application embedded directly inside the Rust binary.

- [x] Set up embedded static asset directory (`frontend/dist`) served via `rust-embed`.
- [x] Add auto-launch browser helper (`open::that("http://localhost:3030")`) on server start.
- [x] **Requirements Input View:**
  - [x] Text area / custom input for requirements (supports Markdown table or YAML).
  - [x] Load EarlyBird preset button.
- [x] **Controls & Parameter Tuning Panel:**
  - [x] Model selector dropdown (Qwen3-0.6B, Qwen3-8B, BGE-Small).
  - [x] Clustering algorithm toggle:
    - [x] Qdrant HNSW Graph (Louvain): slider for similarity edge threshold ($0.30 - 0.75$).
    - [x] Agglomerative Hierarchical: slider for target component count $k$ ($3 - 14$).
  - [x] Live "Fast Re-Cluster" button (~10-50ms using precomputed vectors and candidate embeddings).
- [x] **Interactive 2D Canvas (Apache ECharts):**
  - [x] Render 2D scatter plot (power-iteration PCA projection).
  - [x] Color-code points by assigned architectural component.
  - [x] Hover tooltip: display Requirement ID, source text, PDF section, and component.
  - [x] Interactive data zoom and legend toggle.
- [x] **Architecture Component Cards (Right Pane):**
  - [x] Real-time card deck for each discovered component.
  - [x] Display assigned component name, responsibility statement, cohesion score.
  - [x] Display top c-TF-IDF keywords as badges.
  - [x] Display representative medoid requirement.
  - [x] Allocated requirement tags.
- [x] **Export Actions:**
  - [x] "Export Report" (`components.md`).
  - [x] "Mermaid Diagram" modal with copy to clipboard for software architecture docs.

---

## Phase 3: AMD GPU Acceleration & High-Capacity Models (CachyOS Linux)

Harness the AMD Radeon RX 9070 XT on CachyOS for top-tier embedding models.

- [ ] Add support for **Qwen3-Embedding-8B** (MTEB leader, 4096 dimensions, 32K context).
- [ ] Enable GPU acceleration backends in Candle / FastEmbed:
  - [ ] Linux (CachyOS): ROCm / HIP compute integration (`rocm-hip-sdk`).
  - [ ] Cross-Platform fallback: Vulkan compute backend.
- [ ] Add model quantization support (GGUF / Q4 / Q8) to optimize memory and speed:
  - [ ] Keep 8B model footprint under ~6 GB VRAM.
  - [ ] Sub-second inference across 50–500 requirements.
- [ ] Benchmark comparison report:
  - [ ] Measure CPU vs AMD 9070 XT throughput (tokens/sec).
  - [ ] Measure cluster quality difference (Qwen3-0.6B vs Qwen3-8B).

---

## Phase 4: Cross-Platform Packaging & Portability

Ensure seamless execution on both Windows and Linux.

- [x] Single self-contained binary packaging with embedded HTML/JS/CSS assets (no Node.js or Python runtime required by end-user).
- [x] In-memory topological fallback when Qdrant vector database is offline.
- [x] Create simple startup scripts:
  - [x] `run.bat` for Windows.
  - [x] `run.sh` for CachyOS / Linux.
- [x] Cross-platform path resolution for execution from any working directory.
- [ ] Verify test suite on CachyOS Linux machine with AMD Radeon RX 9070 XT.
