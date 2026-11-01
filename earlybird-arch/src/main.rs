use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod cluster;
mod embed;
mod label;
mod report;
mod requirements;
mod server;
mod store;

use cluster::{
    adjusted_rand_index, cluster_via_louvain, find_best_k_agglomerative, silhouette_score,
};
use embed::{embed_requirements, ModelType};
use label::label_clusters;
use report::{generate_cluster_svg, generate_markdown_report};
use requirements::load_requirements;
use store::{
    connect_qdrant, get_distance_matrix_pairs, setup_collection, update_point_clusters,
    upsert_requirements,
};

#[derive(Parser)]
#[command(name = "earlybird-arch")]
#[command(about = "EarlyBird Semantic Project Architecture: Requirements Embedding, Qdrant Clustering, and Core Architecture Proposal")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generates high-dimensional embeddings for normalized requirements
    Embed {
        #[arg(short, long, default_value = "qwen3-0.6b")]
        model: String,

        #[arg(long, default_value_t = true)]
        instruction: bool,

        #[arg(short, long, default_value = "../data/requirements.yaml")]
        input: PathBuf,

        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Ingests requirements embeddings into Qdrant vector database
    Ingest {
        #[arg(short, long, default_value = "qwen3-0.6b")]
        model: String,

        #[arg(short, long, default_value = "earlybird_requirements")]
        collection: String,

        #[arg(long)]
        qdrant_url: Option<String>,
    },

    /// Clusters requirements using Qdrant distance matrix API or agglomerative linkage
    Cluster {
        #[arg(short, long, default_value = "qwen3-0.6b")]
        model: String,

        #[arg(long, default_value = "graph")]
        method: String,

        #[arg(short, long)]
        k: Option<usize>,

        #[arg(short, long, default_value = "earlybird_requirements")]
        collection: String,

        #[arg(long)]
        qdrant_url: Option<String>,
    },

    /// Runs the complete end-to-end pipeline: Embed -> Ingest -> Cluster -> Label -> Report
    Run {
        #[arg(short, long, default_value = "qwen3-0.6b")]
        model: String,

        #[arg(long, default_value_t = true)]
        instruction: bool,

        #[arg(short, long, default_value = "../data/requirements.yaml")]
        input: PathBuf,

        #[arg(short, long, default_value = "earlybird_requirements")]
        collection: String,

        #[arg(long)]
        qdrant_url: Option<String>,

        #[arg(short, long, default_value = "../out")]
        out_dir: PathBuf,
    },

    /// Launches the interactive visual workbench web server
    Serve {
        #[arg(short, long, default_value_t = 3030)]
        port: u16,

        #[arg(long, default_value_t = false)]
        no_open: bool,

        #[arg(short, long, default_value = "../out")]
        out_dir: PathBuf,

        #[arg(short, long, default_value = "../data/requirements.yaml")]
        input: PathBuf,
    },
}

fn resolve_input_path(path: &std::path::Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }
    for candidate in &[
        "data/requirements.yaml",
        "../data/requirements.yaml",
        "SoftwareArchitecture/SemanticProjectArchitecture/data/requirements.yaml",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return p;
        }
    }
    path.to_path_buf()
}

fn resolve_out_dir(path: &std::path::Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }
    for candidate in &[
        "out",
        "../out",
        "SoftwareArchitecture/SemanticProjectArchitecture/out",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return p;
        }
    }
    path.to_path_buf()
}

fn configure_environment() {
    for candidate in &[
        "SoftwareArchitecture/SemanticProjectArchitecture/earlybird-arch/.fastembed_cache",
        "earlybird-arch/.fastembed_cache",
        ".fastembed_cache",
        "../.fastembed_cache",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            if let Ok(abs) = p.canonicalize() {
                // Strip Windows UNC prefix if present so C:\ paths work cleanly with all libraries
                let path_str = abs.to_string_lossy().trim_start_matches(r"\\?\").to_string();
                unsafe {
                    std::env::set_var("HF_HOME", &path_str);
                    std::env::set_var("HF_HUB_CACHE", &path_str);
                    std::env::set_var("FASTEMBED_CACHE_DIR", &path_str);
                    std::env::set_var("HF_HUB_OFFLINE", "1");
                }
                println!("Configured offline model cache at {}", path_str);
                break;
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    configure_environment();
    let cli = Cli::parse();

    match cli.command {
        Commands::Embed {
            model,
            instruction,
            input,
            output,
        } => {
            let input = resolve_input_path(&input);
            let model_type: ModelType = model.parse()?;
            let reqs = load_requirements(&input)?;
            let out_file = output.unwrap_or_else(|| {
                let out_dir = resolve_out_dir(&PathBuf::from("../out"));
                out_dir.join(format!("embeddings.{}.json", model_type))
            });
            let _ = embed_requirements(&reqs, model_type, instruction, &out_file)?;
            println!("Embedding completed successfully.");
        }

        Commands::Ingest {
            model,
            collection,
            qdrant_url,
        } => {
            let model_type: ModelType = model.parse()?;
            let out_dir = resolve_out_dir(&PathBuf::from("../out"));
            let cache_file = out_dir.join(format!("embeddings.{}.json", model_type));
            if !cache_file.exists() {
                anyhow::bail!("Cached embeddings {:?} not found. Run 'embed' first.", cache_file);
            }
            let content = std::fs::read_to_string(&cache_file)?;
            let cached: embed::CachedEmbeddings = serde_json::from_str(&content)?;

            let client = connect_qdrant(qdrant_url.as_deref()).await?;
            setup_collection(&client, &collection, cached.dimensions).await?;
            upsert_requirements(&client, &collection, &cached.items).await?;
            println!("Ingestion into Qdrant completed successfully.");
        }

        Commands::Cluster {
            model,
            method: _,
            k: _,
            collection: _,
            qdrant_url: _,
        } => {
            println!("To run the integrated clustering and report pipeline, use 'earlybird-arch run --model {}'", model);
        }

        Commands::Run {
            model,
            instruction,
            input,
            collection,
            qdrant_url,
            out_dir,
        } => {
            let input = resolve_input_path(&input);
            let out_dir = resolve_out_dir(&out_dir);

            println!("============================================================");
            println!("   EarlyBird Semantic Architecture Pipeline Starting");
            println!("============================================================");

            let model_type: ModelType = model.parse()?;
            std::fs::create_dir_all(&out_dir)?;

            // 1. Load requirements
            println!("\n[1/6] Loading curated requirements from {:?}", input);
            let reqs = load_requirements(&input)?;
            println!("Loaded {} normalized requirements and domain rules.", reqs.len());

            // 2. Generate / load embeddings
            println!("\n[2/6] Generating embeddings (model: {}, instruction: {})...", model_type, instruction);
            let cache_file = out_dir.join(format!("embeddings.{}.json", model_type));
            let cached = embed_requirements(&reqs, model_type, instruction, &cache_file)?;

            // 3. Ingest into Qdrant
            println!("\n[3/6] Connecting to Qdrant vector database and ingesting points...");
            let client = connect_qdrant(qdrant_url.as_deref()).await?;
            setup_collection(&client, &collection, cached.dimensions).await?;
            upsert_requirements(&client, &collection, &cached.items).await?;

            // 4. Query Qdrant Distance Matrix API
            println!("\n[4/6] Querying Qdrant HNSW Distance Matrix API for pairwise similarities...");
            let sample_size = cached.items.len();
            let limit_neighbors = 8; // Top 8 nearest semantic neighbors per requirement
            let pairs = get_distance_matrix_pairs(&client, &collection, sample_size, limit_neighbors).await?;

            // 5. Cluster vectors
            println!("\n[5/6] Performing vector clustering and evaluation...");
            // A. Primary: Louvain Community Detection on Qdrant Distance Matrix Graph
            let graph_labels = cluster_via_louvain(cached.items.len(), &pairs, 0.45);
            let embeddings: Vec<Vec<f32>> = cached.items.iter().map(|it| it.embedding.clone()).collect();
            let graph_silhouette = silhouette_score(&embeddings, &graph_labels);

            let mut unique_graph_clusters = graph_labels.clone();
            unique_graph_clusters.sort_unstable();
            unique_graph_clusters.dedup();
            println!(
                "  -> Qdrant Graph Clustering: {} components discovered (Silhouette: {:.3})",
                unique_graph_clusters.len(),
                graph_silhouette
            );

            // B. Secondary / Verification: Hierarchical Agglomerative Clustering
            let (best_agglo_k, agglo_labels, agglo_silhouette) =
                find_best_k_agglomerative(&embeddings, 6, 10);
            println!(
                "  -> Agglomerative Clustering: best k={} (Silhouette: {:.3})",
                best_agglo_k, agglo_silhouette
            );

            // C. Adjusted Rand Index
            let ari = adjusted_rand_index(&graph_labels, &agglo_labels);
            println!("  -> Adjusted Rand Index (Graph vs Agglomerative): {:.3}", ari);

            // Choose primary labels for component naming
            let active_labels = graph_labels;

            // 6. Label Clusters & Architectural Components
            println!("\n[6/6] Naming clusters, extracting c-TF-IDF keywords, and generating reports...");
            let components = label_clusters(&cached.items, &active_labels, model_type)?;

            // Update Qdrant point payloads with cluster assignments
            let mut payload_updates = Vec::new();
            for (idx, &cid) in active_labels.iter().enumerate() {
                let comp_name = components
                    .iter()
                    .find(|c| c.cluster_id == cid)
                    .map(|c| c.name.as_str())
                    .unwrap_or("Unknown");
                payload_updates.push((idx as u64, cid, comp_name));
            }
            update_point_clusters(&client, &collection, &payload_updates).await?;

            // Save clusters json
            let clusters_json_path = out_dir.join(format!("clusters.{}.json", model_type));
            let clusters_json = serde_json::to_string_pretty(&components)?;
            std::fs::write(&clusters_json_path, clusters_json)?;
            println!("Saved cluster components data to {:?}", clusters_json_path);

            // Generate components.md
            let report_md_path = out_dir.join("components.md");
            generate_markdown_report(
                &components,
                &cached.items,
                &model_type.to_string(),
                graph_silhouette,
                best_agglo_k,
                agglo_silhouette,
                ari,
                &report_md_path,
            )?;

            // Generate clusters.svg
            let report_svg_path = out_dir.join("clusters.svg");
            generate_cluster_svg(&cached.items, &active_labels, &components, &report_svg_path)?;

            println!("\n============================================================");
            println!("   Pipeline Finished Successfully!");
            println!("   Components Report: {:?}", report_md_path);
            println!("   2D Cluster Plot:   {:?}", report_svg_path);
            println!("============================================================");
        }

        Commands::Serve {
            port,
            no_open,
            out_dir,
            input,
        } => {
            let input = resolve_input_path(&input);
            let out_dir = resolve_out_dir(&out_dir);
            server::start_server(port, no_open, out_dir, input).await?;
        }
    }

    Ok(())
}
