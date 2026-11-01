use anyhow::{Context, Result};
use candle_core::{DType, Device};
use fastembed::{EmbeddingModel, Qwen3TextEmbedding, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::requirements::Requirement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    Qwen3_06B,
    BgeSmall,
}

impl std::str::FromStr for ModelType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "qwen3" | "qwen3-0.6b" | "qwen3_06b" => Ok(ModelType::Qwen3_06B),
            "bge" | "bge-small" | "bgesmall" => Ok(ModelType::BgeSmall),
            other => anyhow::bail!("Unknown model '{}'. Supported: qwen3-0.6b, bge-small", other),
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Qwen3_06B => write!(f, "qwen3-0.6b"),
            ModelType::BgeSmall => write!(f, "bge-small"),
        }
    }
}

pub const QWEN3_INSTRUCTION: &str = "Instruct: Identify the software responsibility (business capability) described by this requirement, for grouping requirements into cohesive software components.\nQuery: ";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedItem {
    pub id: String,
    pub source_ids: Vec<u32>,
    pub section: String,
    pub kind: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEmbeddings {
    pub model: String,
    pub use_instruction: bool,
    pub dimensions: usize,
    pub items: Vec<EmbeddedItem>,
}

fn l2_normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn embed_requirements(
    reqs: &[Requirement],
    model_type: ModelType,
    use_instruction: bool,
    cache_path: &Path,
) -> Result<CachedEmbeddings> {
    if cache_path.exists() {
        if let Ok(content) = std::fs::read_to_string(cache_path) {
            if let Ok(cached) = serde_json::from_str::<CachedEmbeddings>(&content) {
                if cached.model == model_type.to_string()
                    && cached.use_instruction == use_instruction
                    && cached.items.len() == reqs.len()
                {
                    println!("Loaded {} cached embeddings from {:?}", cached.items.len(), cache_path);
                    return Ok(cached);
                }
            }
        }
    }

    println!("Generating embeddings using model: {} (instruction: {})...", model_type, use_instruction);

    let texts_to_embed: Vec<String> = reqs
        .iter()
        .map(|r| {
            if use_instruction && model_type == ModelType::Qwen3_06B {
                format!("{}{}", QWEN3_INSTRUCTION, r.text)
            } else {
                r.text.clone()
            }
        })
        .collect();

    let text_refs: Vec<&str> = texts_to_embed.iter().map(|s| s.as_str()).collect();

    let raw_embeddings = match model_type {
        ModelType::Qwen3_06B => {
            println!("Initializing Qwen3-Embedding-0.6B via Candle CPU backend...");
            let model = Qwen3TextEmbedding::from_hf(
                "Qwen/Qwen3-Embedding-0.6B",
                &Device::Cpu,
                DType::F32,
                512,
            )
            .context("Failed to load Qwen3-Embedding-0.6B")?;
            model.embed(&text_refs).context("Failed to generate Qwen3 embeddings")?
        }
        ModelType::BgeSmall => {
            println!("Initializing BGE-Small-EN-v1.5 via ONNX Runtime...");
            let mut model = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::BGESmallENV15))
                .context("Failed to load BGE-Small-EN-v1.5")?;
            model.embed(text_refs, None).context("Failed to generate BGE-Small embeddings")?
        }
    };

    let mut items = Vec::with_capacity(reqs.len());
    for (req, mut emb) in reqs.iter().zip(raw_embeddings.into_iter()) {
        l2_normalize(&mut emb);
        items.push(EmbeddedItem {
            id: req.id.clone(),
            source_ids: req.source_ids.clone(),
            section: req.section.clone(),
            kind: req.kind.clone(),
            text: req.text.clone(),
            embedding: emb,
        });
    }

    let dim = items.first().map(|it| it.embedding.len()).unwrap_or(0);
    let cached = CachedEmbeddings {
        model: model_type.to_string(),
        use_instruction,
        dimensions: dim,
        items,
    };

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&cached)?;
    std::fs::write(cache_path, json)?;
    println!("Saved {} embeddings to {:?}", cached.items.len(), cache_path);

    Ok(cached)
}

pub fn embed_single_texts(texts: &[&str], model_type: ModelType) -> Result<Vec<Vec<f32>>> {
    match model_type {
        ModelType::Qwen3_06B => {
            let model = Qwen3TextEmbedding::from_hf(
                "Qwen/Qwen3-Embedding-0.6B",
                &Device::Cpu,
                DType::F32,
                512,
            )?;
            let mut embs = model.embed(texts)?;
            for emb in embs.iter_mut() {
                l2_normalize(emb);
            }
            Ok(embs)
        }
        ModelType::BgeSmall => {
            let mut model = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::BGESmallENV15))?;
            let mut embs = model.embed(texts.to_vec(), None)?;
            for emb in embs.iter_mut() {
                l2_normalize(emb);
            }
            Ok(embs)
        }
    }
}
