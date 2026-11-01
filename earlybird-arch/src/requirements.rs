use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Requirement {
    pub id: String,
    pub source_ids: Vec<u32>,
    pub section: String,
    pub kind: String,
    pub text: String,
}

pub fn load_requirements(path: &Path) -> Result<Vec<Requirement>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read requirements file at {:?}", path))?;
    let reqs: Vec<Requirement> = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse YAML from {:?}", path))?;
    Ok(reqs)
}
