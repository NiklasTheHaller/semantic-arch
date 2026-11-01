use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    point_id::PointIdOptions, points_selector::PointsSelectorOneOf, CreateCollectionBuilder,
    Distance, PointId, PointStruct, PointsIdsList, SearchMatrixPointsBuilder,
    SetPayloadPointsBuilder, UpsertPointsBuilder, Value, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use std::collections::HashMap;

use crate::embed::EmbeddedItem;

#[derive(Debug, Clone)]
pub struct MatrixPair {
    pub p1: u64,
    pub p2: u64,
    pub score: f32,
}

pub async fn connect_qdrant(explicit_url: Option<&str>) -> Result<Qdrant> {
    let candidate_urls = if let Some(url) = explicit_url {
        vec![url.to_string()]
    } else if let Ok(env_url) = std::env::var("QDRANT_URL") {
        vec![env_url]
    } else {
        vec![
            "http://127.0.0.1:6334".to_string(),
            "http://192.168.0.213:6334".to_string(),
            "http://localhost:6334".to_string(),
        ]
    };

    for url in &candidate_urls {
        if let Ok(client) = Qdrant::from_url(url).build() {
            if client.health_check().await.is_ok() {
                println!("Connected to Qdrant at {}", url);
                return Ok(client);
            }
        }
    }

    anyhow::bail!(
        "Failed to connect to Qdrant at any candidate address: {:?}",
        candidate_urls
    );
}

pub async fn setup_collection(
    client: &Qdrant,
    collection_name: &str,
    dimensions: usize,
) -> Result<()> {
    if client.collection_exists(collection_name).await? {
        println!("Recreating existing collection '{}'...", collection_name);
        client.delete_collection(collection_name).await?;
    }

    client
        .create_collection(
            CreateCollectionBuilder::new(collection_name)
                .vectors_config(VectorParamsBuilder::new(dimensions as u64, Distance::Cosine)),
        )
        .await
        .context("Failed to create Qdrant collection")?;

    println!(
        "Collection '{}' created (dims: {}, distance: Cosine)",
        collection_name, dimensions
    );
    Ok(())
}

pub async fn upsert_requirements(
    client: &Qdrant,
    collection_name: &str,
    items: &[EmbeddedItem],
) -> Result<()> {
    let mut points = Vec::with_capacity(items.len());

    for (idx, item) in items.iter().enumerate() {
        let mut payload: HashMap<String, Value> = HashMap::new();
        payload.insert("req_id".to_string(), item.id.clone().into());
        payload.insert("section".to_string(), item.section.clone().into());
        payload.insert("kind".to_string(), item.kind.clone().into());
        payload.insert("text".to_string(), item.text.clone().into());
        let source_ids_values: Vec<Value> = item
            .source_ids
            .iter()
            .map(|&id| (id as i64).into())
            .collect();
        payload.insert("source_ids".to_string(), source_ids_values.into());

        points.push(PointStruct::new(
            PointId {
                point_id_options: Some(PointIdOptions::Num(idx as u64)),
            },
            item.embedding.clone(),
            payload,
        ));
    }

    client
        .upsert_points(UpsertPointsBuilder::new(collection_name, points))
        .await
        .context("Failed to upsert points into Qdrant")?;

    println!(
        "Upserted {} points with metadata payload into Qdrant collection '{}'",
        items.len(),
        collection_name
    );
    Ok(())
}

pub async fn get_distance_matrix_pairs(
    client: &Qdrant,
    collection_name: &str,
    sample_size: usize,
    neighbors_limit: usize,
) -> Result<Vec<MatrixPair>> {
    println!(
        "Querying Qdrant Distance Matrix API (sample: {}, limit: {})...",
        sample_size, neighbors_limit
    );

    let request = SearchMatrixPointsBuilder::new(collection_name)
        .sample(sample_size as u64)
        .limit(neighbors_limit as u64);

    let response = client
        .search_matrix_pairs(request)
        .await
        .context("Qdrant search_matrix_pairs failed")?;

    let mut pairs = Vec::new();
    if let Some(result) = response.result {
        for pair in result.pairs {
            let p1 = match pair.a {
                Some(PointId {
                    point_id_options: Some(PointIdOptions::Num(n)),
                }) => n,
                _ => continue,
            };
            let p2 = match pair.b {
                Some(PointId {
                    point_id_options: Some(PointIdOptions::Num(n)),
                }) => n,
                _ => continue,
            };
            pairs.push(MatrixPair {
                p1,
                p2,
                score: pair.score,
            });
        }
    }

    println!(
        "Retrieved {} pairwise similarity edges from Qdrant HNSW distance matrix",
        pairs.len()
    );
    Ok(pairs)
}

pub async fn update_point_clusters(
    client: &Qdrant,
    collection_name: &str,
    updates: &[(u64, usize, &str)],
) -> Result<()> {
    for (point_idx, cluster_id, cluster_name) in updates {
        let mut payload: HashMap<String, Value> = HashMap::new();
        payload.insert("cluster_id".to_string(), (*cluster_id as i64).into());
        payload.insert("cluster_name".to_string(), (*cluster_name).into());

        let selector = PointsSelectorOneOf::Points(PointsIdsList {
            ids: vec![PointId {
                point_id_options: Some(PointIdOptions::Num(*point_idx)),
            }],
        });

        let req = SetPayloadPointsBuilder::new(collection_name, payload).points_selector(selector);

        client.set_payload(req).await.context(format!(
            "Failed to update cluster payload for point {}",
            point_idx
        ))?;
    }

    println!(
        "Updated cluster metadata in Qdrant point payloads for {} requirements",
        updates.len()
    );
    Ok(())
}
