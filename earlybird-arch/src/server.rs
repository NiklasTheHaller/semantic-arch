use anyhow::{Context, Result};
use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{header, StatusCode, Uri},
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use qdrant_client::Qdrant;
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

use crate::cluster::{
    cluster_agglomerative, cluster_via_louvain,
    metrics::{adjusted_rand_index, cosine_similarity, silhouette_score},
};
use crate::embed::{embed_requirements, EmbeddedItem, ModelType};
use crate::label::{
    compute_component_edges, label_clusters, ClusterComponent, ComponentEdge, NamingMode,
};
use crate::report::{generate_cluster_svg, generate_markdown_report, pca_2d};
use crate::requirements::{load_requirements, Requirement};
use crate::store::{connect_qdrant, update_point_clusters, MatrixPair};

#[derive(RustEmbed)]
#[folder = "frontend/dist/"]
struct Asset;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub id: String,
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub cluster_id: usize,
    pub source_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDataResponse {
    pub project_name: String,
    pub naming_mode: String,
    pub points: Vec<Point2D>,
    pub components: Vec<ClusterComponent>,
    pub edges: Vec<ComponentEdge>,
    pub silhouette: f32,
    pub pairs_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemStatus {
    pub os: String,
    pub hardware: String,
    pub qdrant_status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub dims: usize,
    pub cached: bool,
    pub hardware_target: String,
}

#[derive(Debug, Deserialize)]
pub struct PipelineRunRequest {
    pub model: String,
    pub instruction: bool,
    pub method: String, // "graph" or "agglo"
    pub threshold: Option<f32>,
    pub target_k: Option<usize>,
    pub fast_recluster: bool,
    pub custom_input: Option<String>,
    pub naming_mode: Option<String>,
    pub project_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub id: String,
    pub name: String,
    pub timestamp: String,
    pub model: String,
    pub instruction: bool,
    pub method: String,
    #[serde(default)]
    pub naming_mode: Option<String>,
    #[serde(default)]
    pub project_name: Option<String>,
    pub num_requirements: usize,
    pub num_components: usize,
    pub silhouette: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSnapshot {
    pub id: String,
    pub name: String,
    pub timestamp: String,
    pub model: String,
    pub instruction: bool,
    pub method: String,
    #[serde(default)]
    pub naming_mode: Option<String>,
    #[serde(default)]
    pub project_name: Option<String>,
    pub num_requirements: usize,
    pub num_components: usize,
    pub silhouette: f32,
    pub data: PipelineDataResponse,
}

#[derive(Debug, Deserialize)]
pub struct SaveRunPayload {
    pub name: String,
    pub method: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CompareQuery {
    pub run1: String,
    pub run2: String,
}

#[derive(Debug, Serialize)]
pub struct RequirementShift {
    pub id: String,
    pub text: String,
    pub cluster_1: String,
    pub cluster_2: String,
    pub changed: bool,
}

#[derive(Debug, Serialize)]
pub struct CompareResult {
    pub run1_id: String,
    pub run1_name: String,
    pub run2_id: String,
    pub run2_name: String,
    pub ari_score: f32,
    pub total_count: usize,
    pub identical_count: usize,
    pub shifted_count: usize,
    pub shifts: Vec<RequirementShift>,
}

pub struct PipelineSession {
    pub model_type: ModelType,
    pub instruction: bool,
    pub naming_mode: NamingMode,
    pub project_name: String,
    pub active_dataset: String,
    pub items: Vec<EmbeddedItem>,
    pub components: Vec<ClusterComponent>,
    pub edges: Vec<ComponentEdge>,
    pub labels: Vec<usize>,
    pub points_2d: Vec<(f32, f32)>,
    pub silhouette: f32,
    pub pairs: Vec<MatrixPair>,
    pub qdrant_client: Option<Qdrant>,
    pub qdrant_address: Option<String>,
    pub collection: String,
}

#[derive(Clone)]
pub struct AppState {
    pub session: Arc<RwLock<PipelineSession>>,
    pub out_dir: PathBuf,
    pub input_path: PathBuf,
}

pub async fn start_server(
    port: u16,
    no_open: bool,
    out_dir: PathBuf,
    input_path: PathBuf,
) -> Result<()> {
    println!("\nInitializing Semantic Architecture Workbench session...");

    let model_type = ModelType::Qwen3_06B;
    let cache_file = out_dir.join(format!("embeddings.normalized.{}.json", model_type));

    // 1. Load normalized requirements and verified cache
    println!("Loading normalized requirements from {:?}", input_path);
    let reqs = load_requirements(&input_path)?;
    let cached = embed_requirements(&reqs, model_type, true, &cache_file)?;

    let items = cached.items;
    let embeddings: Vec<Vec<f32>> = items.iter().map(|it| it.embedding.clone()).collect();
    let n = embeddings.len();

    // 2. Precompute pairwise similarity matrix
    let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            pairs.push(MatrixPair {
                p1: i as u64,
                p2: j as u64,
                score: sim,
            });
        }
    }

    // 3. Initial Louvain graph clustering at default 0.45
    let labels = cluster_via_louvain(n, &pairs, 0.45);
    let silhouette = silhouette_score(&embeddings, &labels);
    let points_2d = pca_2d(&embeddings);

    // 4. Algorithmically synthesize components (SoftwareArchetype mode, no hardcoded candidates)
    let naming_mode = NamingMode::SoftwareArchetype;
    let project_name = "EarlyBird".to_string();
    let components = label_clusters(&items, &labels, naming_mode)?;
    let edges = compute_component_edges(&items, &labels, &components);

    let initial_points: Vec<Point2D> = items
        .iter()
        .enumerate()
        .map(|(i, it)| Point2D {
            id: it.id.clone(),
            text: it.text.clone(),
            x: points_2d.get(i).map(|p| p.0).unwrap_or(0.0),
            y: points_2d.get(i).map(|p| p.1).unwrap_or(0.0),
            cluster_id: labels.get(i).copied().unwrap_or(0),
            source_ids: it.source_ids.clone(),
        })
        .collect();
    let initial_components = components.clone();
    let initial_edges = edges.clone();
    let initial_pairs_count = pairs.len();
    let initial_num_items = items.len();

    // 5. Test Qdrant connectivity
    println!("Testing Qdrant vector database connectivity...");
    let (qdrant_client, qdrant_address) = match connect_qdrant(None).await {
        Ok(c) => (Some(c), Some("Connected (localhost:6334)".to_string())),
        Err(e) => {
            println!("Note: Qdrant vector DB is offline or not reachable ({}). Running with in-memory vector index.", e);
            (None, None)
        }
    };

    let session = PipelineSession {
        model_type,
        instruction: true,
        naming_mode,
        project_name: project_name.clone(),
        active_dataset: "normalized".to_string(),
        items,
        components,
        edges,
        labels,
        points_2d,
        silhouette,
        pairs,
        qdrant_client,
        qdrant_address,
        collection: "earlybird_requirements".to_string(),
    };

    let app_state = AppState {
        session: Arc::new(RwLock::new(session)),
        out_dir: out_dir.clone(),
        input_path: input_path.clone(),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/system", get(get_system_status))
        .route("/api/models", get(get_models))
        .route("/api/pipeline/current", get(get_current_pipeline))
        .route("/api/pipeline/run", post(run_pipeline))
        .route("/api/export/markdown", get(export_markdown))
        .route("/api/export/svg", get(export_svg))
        .route("/api/runs", get(list_runs))
        .route("/api/runs/save", post(save_run))
        .route("/api/runs/load/{id}", post(load_run))
        .route("/api/runs/{id}", delete(delete_run))
        .route("/api/runs/compare", get(compare_runs))
        .fallback(static_handler)
        .layer(cors)
        .with_state(app_state.clone());

    // Auto-seed initial baseline run if runs folder is empty
    let runs_dir = out_dir.join("runs");
    let _ = std::fs::create_dir_all(&runs_dir);
    let is_runs_empty = std::fs::read_dir(&runs_dir)
        .map(|mut d| d.next().is_none())
        .unwrap_or(true);

    if is_runs_empty {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let base_snapshot = RunSnapshot {
            id: format!("baseline_{}", now_secs),
            name: "Baseline: Normalized (Qwen3-0.6B + Instruct)".to_string(),
            timestamp: format!("{}", now_secs),
            model: "qwen3-0.6b".to_string(),
            instruction: true,
            method: "graph".to_string(),
            naming_mode: Some("archetype".to_string()),
            project_name: Some(project_name.clone()),
            num_requirements: initial_num_items,
            num_components: initial_components.len(),
            silhouette,
            data: PipelineDataResponse {
                project_name: project_name.clone(),
                naming_mode: "archetype".to_string(),
                points: initial_points,
                components: initial_components,
                edges: initial_edges,
                silhouette,
                pairs_count: initial_pairs_count,
            },
        };

        if let Ok(json_str) = serde_json::to_string_pretty(&base_snapshot) {
            let _ = std::fs::write(runs_dir.join(format!("{}.json", base_snapshot.id)), json_str);
        }
    }

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await
        .with_context(|| format!("Failed to bind server to port {}", port))?;

    let local_url = format!("http://localhost:{}", port);
    println!("\n============================================================");
    println!("   EarlyBird Semantic Architecture Workbench");
    println!("   Running at: {}", local_url);
    println!("   Network:    http://0.0.0.0:{}", port);
    println!("   Press Ctrl+C to terminate");
    println!("============================================================\n");

    if !no_open {
        let url_to_open = local_url.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;
            if let Err(err) = open::that(&url_to_open) {
                eprintln!("Could not auto-open browser: {}", err);
            }
        });
    }

    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
}

async fn get_system_status(State(state): State<AppState>) -> Json<SystemStatus> {
    let session = state.session.read().await;

    let os_name = match std::env::consts::OS {
        "windows" => "Windows 11 (x86_64)",
        "linux" => "Linux (CachyOS)",
        "macos" => "macOS",
        other => other,
    };

    let hardware_info = if cfg!(target_os = "linux") {
        "AMD Radeon RX 9070 XT (ROCm 6.x) • AVX2 CPU"
    } else {
        "CPU AVX2/FMA Vectorized • (ROCm available on CachyOS)"
    };

    let qdrant_status = session
        .qdrant_address
        .clone()
        .unwrap_or_else(|| "In-Memory Topology Mode (Docker Qdrant Standby)".to_string());

    Json(SystemStatus {
        os: os_name.to_string(),
        hardware: hardware_info.to_string(),
        qdrant_status,
    })
}

async fn get_models() -> Json<Vec<ModelInfo>> {
    Json(vec![
        ModelInfo {
            id: "qwen3-0.6b".to_string(),
            name: "Qwen3-Embedding-0.6B (1024-d, Instruction-aware)".to_string(),
            dims: 1024,
            cached: true,
            hardware_target: "CPU Vectorized / ROCm".to_string(),
        },
        ModelInfo {
            id: "bge-small".to_string(),
            name: "BGE-Small-EN-v1.5 (384-d, ONNX Fast)".to_string(),
            dims: 384,
            cached: true,
            hardware_target: "ONNX Runtime Fast".to_string(),
        },
        ModelInfo {
            id: "qwen3-8b".to_string(),
            name: "Qwen3-Embedding-8B (4096-d, AMD 9070XT ROCm)".to_string(),
            dims: 4096,
            cached: false,
            hardware_target: "AMD 9070 XT (ROCm / Vulkan)".to_string(),
        },
    ])
}

async fn get_current_pipeline(State(state): State<AppState>) -> Json<PipelineDataResponse> {
    let session = state.session.read().await;

    let points = session
        .items
        .iter()
        .enumerate()
        .map(|(i, it)| Point2D {
            id: it.id.clone(),
            text: it.text.clone(),
            x: session.points_2d.get(i).map(|p| p.0).unwrap_or(0.0),
            y: session.points_2d.get(i).map(|p| p.1).unwrap_or(0.0),
            cluster_id: session.labels.get(i).copied().unwrap_or(0),
            source_ids: it.source_ids.clone(),
        })
        .collect();

    Json(PipelineDataResponse {
        project_name: session.project_name.clone(),
        naming_mode: session.naming_mode.as_str().to_string(),
        points,
        components: session.components.clone(),
        edges: session.edges.clone(),
        silhouette: session.silhouette,
        pairs_count: session.pairs.len(),
    })
}

async fn run_pipeline(
    State(state): State<AppState>,
    Json(payload): Json<PipelineRunRequest>,
) -> Result<Json<PipelineDataResponse>, (StatusCode, String)> {
    let mut session = state.session.write().await;

    let naming_mode = NamingMode::from_str_opt(payload.naming_mode.as_deref());
    if let Some(ref p) = payload.project_name {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            session.project_name = trimmed.to_string();
        }
    }
    session.naming_mode = naming_mode;

    let target_model = payload
        .model
        .parse::<ModelType>()
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let (is_custom, dataset_key) = match payload.custom_input {
        Some(ref raw) if !raw.trim().is_empty() => (true, format!("custom_{}", raw.trim().len())),
        _ => (false, "normalized".to_string()),
    };

    let need_reembed = !payload.fast_recluster
        || session.model_type != target_model
        || session.instruction != payload.instruction
        || session.active_dataset != dataset_key;

    if need_reembed {
        let (reqs, cache_file) = if is_custom {
            let raw = payload.custom_input.as_deref().unwrap_or("").trim();
            let parsed_reqs = if let Ok(parsed) = serde_yaml::from_str::<Vec<Requirement>>(raw) {
                parsed
            } else {
                raw.lines()
                    .filter(|l| !l.trim().is_empty())
                    .enumerate()
                    .map(|(idx, line)| Requirement {
                        id: format!("REQ-{:02}", idx + 1),
                        source_ids: vec![idx as u32 + 1],
                        section: "Custom Requirements".to_string(),
                        kind: "functional".to_string(),
                        text: line
                            .trim()
                            .trim_start_matches(|c: char| c == '-' || c == '*' || c == '.' || c.is_numeric())
                            .trim()
                            .to_string(),
                    })
                    .collect()
            };
            (parsed_reqs, state.out_dir.join(format!("embeddings.custom.{}.json", target_model)))
        } else {
            let norm_reqs = load_requirements(&state.input_path)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            (norm_reqs, state.out_dir.join(format!("embeddings.normalized.{}.json", target_model)))
        };

        let cached = embed_requirements(&reqs, target_model, payload.instruction, &cache_file)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        session.items = cached.items;
        session.model_type = target_model;
        session.instruction = payload.instruction;
        session.active_dataset = dataset_key;

        // Recompute pairwise similarity matrix
        let embeddings: Vec<Vec<f32>> = session.items.iter().map(|it| it.embedding.clone()).collect();
        let n = embeddings.len();
        let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
                pairs.push(MatrixPair {
                    p1: i as u64,
                    p2: j as u64,
                    score: sim,
                });
            }
        }
        session.pairs = pairs;
        session.points_2d = pca_2d(&embeddings);
    }

    let embeddings: Vec<Vec<f32>> = session.items.iter().map(|it| it.embedding.clone()).collect();
    let n = embeddings.len();

    // Perform clustering according to method
    let labels = match payload.method.as_str() {
        "agglo" => {
            let k = payload.target_k.unwrap_or(8).clamp(2, n);
            cluster_agglomerative(&embeddings, k)
        }
        _ => {
            let threshold = payload.threshold.unwrap_or(0.45);
            cluster_via_louvain(n, &session.pairs, threshold)
        }
    };

    let silhouette = silhouette_score(&embeddings, &labels);
    let components = label_clusters(&session.items, &labels, naming_mode)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let edges = compute_component_edges(&session.items, &labels, &components);

    // Optionally update Qdrant points payload if connected
    if let Some(ref client) = session.qdrant_client {
        let mut updates = Vec::new();
        for (idx, &cid) in labels.iter().enumerate() {
            let comp_name = components
                .iter()
                .find(|c| c.cluster_id == cid)
                .map(|c| c.name.as_str())
                .unwrap_or("Unknown");
            updates.push((idx as u64, cid, comp_name));
        }
        let _ = update_point_clusters(client, &session.collection, &updates).await;
    }

    session.labels = labels.clone();
    session.components = components.clone();
    session.edges = edges.clone();
    session.silhouette = silhouette;

    // Generate output files in background/disk
    let _ = std::fs::create_dir_all(&state.out_dir);
    let md_path = state.out_dir.join("components.md");
    let _ = generate_markdown_report(
        &components,
        &session.items,
        &session.project_name,
        &session.model_type.to_string(),
        silhouette,
        components.len(),
        silhouette,
        1.0,
        &md_path,
    );
    let svg_path = state.out_dir.join("clusters.svg");
    let _ = generate_cluster_svg(&session.items, &labels, &components, &session.project_name, &svg_path);

    let points = session
        .items
        .iter()
        .enumerate()
        .map(|(i, it)| Point2D {
            id: it.id.clone(),
            text: it.text.clone(),
            x: session.points_2d.get(i).map(|p| p.0).unwrap_or(0.0),
            y: session.points_2d.get(i).map(|p| p.1).unwrap_or(0.0),
            cluster_id: labels[i],
            source_ids: it.source_ids.clone(),
        })
        .collect();

    Ok(Json(PipelineDataResponse {
        project_name: session.project_name.clone(),
        naming_mode: naming_mode.as_str().to_string(),
        points,
        components,
        edges,
        silhouette,
        pairs_count: session.pairs.len(),
    }))
}

async fn export_markdown(State(state): State<AppState>) -> impl IntoResponse {
    let md_path = state.out_dir.join("components.md");
    if let Ok(content) = std::fs::read_to_string(&md_path) {
        (
            [
                (header::CONTENT_TYPE, "text/markdown; charset=utf-8"),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"components.md\"",
                ),
            ],
            content,
        )
            .into_response()
    } else {
        (StatusCode::NOT_FOUND, "Report not yet generated").into_response()
    }
}

async fn export_svg(State(state): State<AppState>) -> impl IntoResponse {
    let svg_path = state.out_dir.join("clusters.svg");
    if let Ok(content) = std::fs::read_to_string(&svg_path) {
        (
            [
                (header::CONTENT_TYPE, "image/svg+xml"),
                (
                    header::CONTENT_DISPOSITION,
                    "inline; filename=\"clusters.svg\"",
                ),
            ],
            content,
        )
            .into_response()
    } else {
        (StatusCode::NOT_FOUND, "SVG not yet generated").into_response()
    }
}

async fn static_handler(uri: Uri) -> impl IntoResponse {
    let mut path = uri.path().trim_start_matches('/').to_string();
    if path.is_empty() {
        path = "index.html".to_string();
    }

    match Asset::get(&path) {
        Some(content) => {
            let content_type = match path.rsplit('.').next() {
                Some("html") => "text/html; charset=utf-8",
                Some("js") => "application/javascript; charset=utf-8",
                Some("css") => "text/css; charset=utf-8",
                Some("svg") => "image/svg+xml",
                Some("json") => "application/json",
                _ => "application/octet-stream",
            };
            ([(header::CONTENT_TYPE, content_type)], content.data).into_response()
        }
        None => {
            if let Some(content) = Asset::get("index.html") {
                ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], content.data).into_response()
            } else {
                (StatusCode::NOT_FOUND, "404 Not Found").into_response()
            }
        }
    }
}

async fn list_runs(State(state): State<AppState>) -> Json<Vec<RunSummary>> {
    let runs_dir = state.out_dir.join("runs");
    let mut list = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&runs_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(snap) = serde_json::from_str::<RunSnapshot>(&content) {
                        list.push(RunSummary {
                            id: snap.id,
                            name: snap.name,
                            timestamp: snap.timestamp,
                            model: snap.model,
                            instruction: snap.instruction,
                            method: snap.method,
                            naming_mode: snap.naming_mode,
                            project_name: snap.project_name,
                            num_requirements: snap.num_requirements,
                            num_components: snap.num_components,
                            silhouette: snap.silhouette,
                        });
                    }
                }
            }
        }
    }
    // Sort descending by id
    list.sort_by(|a, b| b.id.cmp(&a.id));
    Json(list)
}

async fn save_run(
    State(state): State<AppState>,
    Json(payload): Json<SaveRunPayload>,
) -> Result<Json<RunSnapshot>, (StatusCode, String)> {
    let session = state.session.read().await;

    let runs_dir = state.out_dir.join("runs");
    std::fs::create_dir_all(&runs_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let id = format!("run_{}", now_secs);

    let points: Vec<Point2D> = session
        .items
        .iter()
        .enumerate()
        .map(|(i, it)| Point2D {
            id: it.id.clone(),
            text: it.text.clone(),
            x: session.points_2d.get(i).map(|p| p.0).unwrap_or(0.0),
            y: session.points_2d.get(i).map(|p| p.1).unwrap_or(0.0),
            cluster_id: session.labels.get(i).copied().unwrap_or(0),
            source_ids: it.source_ids.clone(),
        })
        .collect();

    let data = PipelineDataResponse {
        project_name: session.project_name.clone(),
        naming_mode: session.naming_mode.as_str().to_string(),
        points,
        components: session.components.clone(),
        edges: session.edges.clone(),
        silhouette: session.silhouette,
        pairs_count: session.pairs.len(),
    };

    let snapshot = RunSnapshot {
        id: id.clone(),
        name: if payload.name.trim().is_empty() {
            format!("Run ({})", session.model_type)
        } else {
            payload.name
        },
        timestamp: format!("{}", now_secs),
        model: session.model_type.to_string(),
        instruction: session.instruction,
        method: payload.method.unwrap_or_else(|| "graph".to_string()),
        naming_mode: Some(session.naming_mode.as_str().to_string()),
        project_name: Some(session.project_name.clone()),
        num_requirements: session.items.len(),
        num_components: session.components.len(),
        silhouette: session.silhouette,
        data,
    };

    let file_path = runs_dir.join(format!("{}.json", id));
    let json_str = serde_json::to_string_pretty(&snapshot)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    std::fs::write(&file_path, json_str)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(snapshot))
}

async fn load_run(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<PipelineDataResponse>, (StatusCode, String)> {
    let runs_dir = state.out_dir.join("runs");
    let file_path = runs_dir.join(format!("{}.json", id));
    if !file_path.exists() {
        return Err((StatusCode::NOT_FOUND, "Snapshot not found".to_string()));
    }
    let content = std::fs::read_to_string(&file_path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let snap: RunSnapshot = serde_json::from_str(&content)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut session = state.session.write().await;
    session.components = snap.data.components.clone();
    session.edges = snap.data.edges.clone();
    session.silhouette = snap.data.silhouette;
    session.labels = snap.data.points.iter().map(|p| p.cluster_id).collect();
    session.points_2d = snap.data.points.iter().map(|p| (p.x, p.y)).collect();
    if !snap.data.project_name.is_empty() {
        session.project_name = snap.data.project_name.clone();
    }
    session.naming_mode = NamingMode::from_str_opt(Some(&snap.data.naming_mode));

    // Sync items text/id if point counts match
    if session.items.len() == snap.data.points.len() {
        for (i, p) in snap.data.points.iter().enumerate() {
            session.items[i].id = p.id.clone();
            session.items[i].text = p.text.clone();
            session.items[i].source_ids = p.source_ids.clone();
        }
    }

    // Update markdown report and cluster svg
    let md_path = state.out_dir.join("components.md");
    let _ = generate_markdown_report(
        &session.components,
        &session.items,
        &session.project_name,
        &session.model_type.to_string(),
        session.silhouette,
        session.components.len(),
        session.silhouette,
        1.0,
        &md_path,
    );
    let svg_path = state.out_dir.join("clusters.svg");
    let _ = generate_cluster_svg(&session.items, &session.labels, &session.components, &session.project_name, &svg_path);

    Ok(Json(snap.data))
}

async fn delete_run(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let file_path = state.out_dir.join("runs").join(format!("{}.json", id));
    if file_path.exists() {
        std::fs::remove_file(&file_path)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn compare_runs(
    State(state): State<AppState>,
    Query(query): Query<CompareQuery>,
) -> Result<Json<CompareResult>, (StatusCode, String)> {
    let runs_dir = state.out_dir.join("runs");
    let path1 = runs_dir.join(format!("{}.json", query.run1));
    let path2 = runs_dir.join(format!("{}.json", query.run2));

    let content1 = std::fs::read_to_string(&path1)
        .map_err(|_| (StatusCode::NOT_FOUND, format!("Run 1 '{}' not found", query.run1)))?;
    let content2 = std::fs::read_to_string(&path2)
        .map_err(|_| (StatusCode::NOT_FOUND, format!("Run 2 '{}' not found", query.run2)))?;

    let snap1: RunSnapshot = serde_json::from_str(&content1)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let snap2: RunSnapshot = serde_json::from_str(&content2)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Create maps from requirement ID to component name
    let comp_map1: std::collections::HashMap<usize, String> = snap1
        .data
        .components
        .iter()
        .map(|c| (c.cluster_id, c.name.clone()))
        .collect();
    let comp_map2: std::collections::HashMap<usize, String> = snap2
        .data
        .components
        .iter()
        .map(|c| (c.cluster_id, c.name.clone()))
        .collect();

    let mut map1_by_req: std::collections::HashMap<String, (usize, String, String)> = std::collections::HashMap::new();
    for p in &snap1.data.points {
        let comp_name = comp_map1.get(&p.cluster_id).cloned().unwrap_or_else(|| "Unknown".into());
        map1_by_req.insert(p.id.clone(), (p.cluster_id, comp_name, p.text.clone()));
    }

    let mut labels1 = Vec::new();
    let mut labels2 = Vec::new();
    let mut shifts = Vec::new();
    let mut identical_count = 0;
    let mut shifted_count = 0;

    for p2 in &snap2.data.points {
        if let Some((c1_id, c1_name, text)) = map1_by_req.get(&p2.id) {
            let c2_name = comp_map2.get(&p2.cluster_id).cloned().unwrap_or_else(|| "Unknown".into());
            labels1.push(*c1_id);
            labels2.push(p2.cluster_id);

            let changed = c1_name != &c2_name;
            if changed {
                shifted_count += 1;
            } else {
                identical_count += 1;
            }

            shifts.push(RequirementShift {
                id: p2.id.clone(),
                text: text.clone(),
                cluster_1: c1_name.clone(),
                cluster_2: c2_name,
                changed,
            });
        }
    }

    let ari_score = if labels1.len() == labels2.len() && !labels1.is_empty() {
        adjusted_rand_index(&labels1, &labels2)
    } else {
        0.0
    };

    Ok(Json(CompareResult {
        run1_id: snap1.id,
        run1_name: snap1.name,
        run2_id: snap2.id,
        run2_name: snap2.name,
        ari_score,
        total_count: shifts.len(),
        identical_count,
        shifted_count,
        shifts,
    }))
}
