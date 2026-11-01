use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::{header, StatusCode, Uri},
    response::IntoResponse,
    routing::{get, post},
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
    metrics::{cosine_similarity, silhouette_score},
};
use crate::embed::{embed_requirements, EmbeddedItem, ModelType};
use crate::label::{
    label_clusters_with_candidates, precompute_candidate_embeddings, ClusterComponent,
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
    pub points: Vec<Point2D>,
    pub components: Vec<ClusterComponent>,
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
}

pub struct PipelineSession {
    pub model_type: ModelType,
    pub instruction: bool,
    pub items: Vec<EmbeddedItem>,
    pub components: Vec<ClusterComponent>,
    pub labels: Vec<usize>,
    pub points_2d: Vec<(f32, f32)>,
    pub silhouette: f32,
    pub pairs: Vec<MatrixPair>,
    pub candidate_embeddings: Vec<Vec<f32>>,
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
    let cache_file = out_dir.join(format!("embeddings.{}.json", model_type));

    // 1. Load or compute requirements embeddings
    let cached = if cache_file.exists() {
        println!("Loading cached embeddings from {:?}", cache_file);
        let content = std::fs::read_to_string(&cache_file)?;
        let data: crate::embed::CachedEmbeddings = serde_json::from_str(&content)?;
        data
    } else {
        println!("Computing initial embeddings for {:?}", input_path);
        let reqs = load_requirements(&input_path)?;
        embed_requirements(&reqs, model_type, true, &cache_file)?
    };

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

    // 4. Precompute candidate component embeddings for fast matching
    let cand_cache_path = out_dir.join(format!("candidate_embeddings.{}.json", model_type));
    let candidate_embeddings = if cand_cache_path.exists() {
        println!("Loading cached candidate embeddings from {:?}", cand_cache_path);
        let content = std::fs::read_to_string(&cand_cache_path)?;
        serde_json::from_str::<Vec<Vec<f32>>>(&content)?
    } else {
        println!("Precomputing candidate architecture embeddings...");
        let embs = precompute_candidate_embeddings(model_type)?;
        let json = serde_json::to_string_pretty(&embs)?;
        let _ = std::fs::write(&cand_cache_path, json);
        embs
    };
    let components = label_clusters_with_candidates(&items, &labels, &candidate_embeddings)?;

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
        items,
        components,
        labels,
        points_2d,
        silhouette,
        pairs,
        candidate_embeddings,
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
        .fallback(static_handler)
        .layer(cors)
        .with_state(app_state);

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
        points,
        components: session.components.clone(),
        silhouette: session.silhouette,
        pairs_count: session.pairs.len(),
    })
}

async fn run_pipeline(
    State(state): State<AppState>,
    Json(payload): Json<PipelineRunRequest>,
) -> Result<Json<PipelineDataResponse>, (StatusCode, String)> {
    let mut session = state.session.write().await;

    let target_model = payload
        .model
        .parse::<ModelType>()
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let need_reembed = !payload.fast_recluster
        || session.model_type != target_model
        || session.instruction != payload.instruction
        || payload.custom_input.is_some();

    if need_reembed {
        // Parse custom input or default requirements
        let reqs = if let Some(ref raw) = payload.custom_input {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                load_requirements(&state.input_path)
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            } else if let Ok(parsed) = serde_yaml::from_str::<Vec<Requirement>>(trimmed) {
                parsed
            } else {
                // Parse line by line
                trimmed
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .enumerate()
                    .map(|(idx, line)| Requirement {
                        id: format!("REQ-{:02}", idx + 1),
                        source_ids: vec![idx as u32 + 1],
                        section: "Custom Requirements".to_string(),
                        kind: "functional".to_string(),
                        text: line.trim().trim_start_matches(|c: char| c == '-' || c == '*' || c == '.' || c.is_numeric()).trim().to_string(),
                    })
                    .collect()
            }
        } else {
            load_requirements(&state.input_path)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        };

        let cache_file = state.out_dir.join(format!("embeddings.{}.json", target_model));
        let cached = embed_requirements(&reqs, target_model, payload.instruction, &cache_file)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        session.items = cached.items;
        session.model_type = target_model;
        session.instruction = payload.instruction;

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

        // Precompute or load candidate embeddings for this model
        let cand_cache_path = state.out_dir.join(format!("candidate_embeddings.{}.json", target_model));
        session.candidate_embeddings = if cand_cache_path.exists() {
            let content = std::fs::read_to_string(&cand_cache_path)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            serde_json::from_str::<Vec<Vec<f32>>>(&content)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        } else {
            let embs = precompute_candidate_embeddings(target_model)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            let _ = std::fs::write(&cand_cache_path, serde_json::to_string_pretty(&embs).unwrap_or_default());
            embs
        };
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
    let components = label_clusters_with_candidates(&session.items, &labels, &session.candidate_embeddings)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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
    session.silhouette = silhouette;

    // Generate output files in background/disk
    let _ = std::fs::create_dir_all(&state.out_dir);
    let md_path = state.out_dir.join("components.md");
    let _ = generate_markdown_report(
        &components,
        &session.items,
        &session.model_type.to_string(),
        silhouette,
        components.len(),
        silhouette,
        1.0,
        &md_path,
    );
    let svg_path = state.out_dir.join("clusters.svg");
    let _ = generate_cluster_svg(&session.items, &labels, &components, &svg_path);

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
        points,
        components,
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
