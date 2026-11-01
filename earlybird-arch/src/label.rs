use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::cluster::metrics::cosine_similarity;
use crate::embed::{embed_single_texts, EmbeddedItem, ModelType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterComponent {
    pub cluster_id: usize,
    pub name: String,
    pub responsibility: String,
    pub cohesion: f32,
    pub medoid_id: String,
    pub medoid_text: String,
    pub keywords: Vec<String>,
    pub requirement_ids: Vec<String>,
    pub source_pdf_ids: Vec<u32>,
}

pub struct CandidateArchitecture {
    pub name: &'static str,
    pub responsibility: &'static str,
}

pub const CANDIDATE_COMPONENTS: &[CandidateArchitecture] = &[
    CandidateArchitecture {
        name: "Product Catalog & Search",
        responsibility: "Manages catalog products, prepackaged breakfast compositions, unit pricing, and anonymous/authenticated product search.",
    },
    CandidateArchitecture {
        name: "Customer Identity & Account Management",
        responsibility: "Customer identification, account number validation with checksums, authentication, and blacklist enforcement.",
    },
    CandidateArchitecture {
        name: "Shopping Cart & Order Placement",
        responsibility: "Shopping cart composition, follow-up order blueprint copying, order validation, and order number assignment.",
    },
    CandidateArchitecture {
        name: "Order Lifecycle & Cancellation",
        responsibility: "Order tracking status inquiry, cancellation validation against packaging stage, and immutability management.",
    },
    CandidateArchitecture {
        name: "Fulfilment & Packing Operations",
        responsibility: "Order assembly by packing clerks, delivery clerk assignment, bag labeling, and duplicate numbered invoice generation.",
    },
    CandidateArchitecture {
        name: "Delivery Logistics & Itinerary",
        responsibility: "Multi-order itinerary calculation, driver route printouts, and smartphone browser-based delivery confirmation.",
    },
    CandidateArchitecture {
        name: "SMS Ordering Gateway",
        responsibility: "Inbound SMS message syntax parsing for customer orders and cancellations, and automated SMS confirmation replies.",
    },
    CandidateArchitecture {
        name: "External Payment Integration",
        responsibility: "Formatting and dispatching expected-payment interface records to the external payment system upon packaging completion.",
    },
    CandidateArchitecture {
        name: "Management Reporting & Analytics",
        responsibility: "Automated nightly managerial business reports compiling daily orders, revenues, clerks, and customer logistics.",
    },
    CandidateArchitecture {
        name: "Role & Permission Authorization",
        responsibility: "User role assignment and authorization across customer, packing clerk, delivery clerk, and manager personas.",
    },
];

const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should",
    "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    // Domain stop words to avoid uninformative keywords
    "system", "order", "orders", "customer", "customers", "clerk", "clerks", "lets", "let", "per",
    "number", "e.g", "also", "using", "placed", "one", "two", "without", "new",
];

fn extract_words(text: &str) -> Vec<String> {
    let stopset: HashSet<&str> = STOPWORDS.iter().copied().collect();
    text.split(|c: char| !c.is_alphanumeric())
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= 3 && !stopset.contains(s.as_str()) && !s.chars().all(|c| c.is_numeric()))
        .collect()
}

pub fn extract_cluster_keywords(
    cluster_docs: &HashMap<usize, Vec<&str>>,
    top_n: usize,
) -> HashMap<usize, Vec<String>> {
    let mut doc_words: HashMap<usize, Vec<String>> = HashMap::new();
    let mut word_doc_freq: HashMap<String, usize> = HashMap::new();

    for (&cluster_id, texts) in cluster_docs {
        let mut words = Vec::new();
        let mut unique_in_cluster = HashSet::new();
        for text in texts {
            for w in extract_words(text) {
                words.push(w.clone());
                unique_in_cluster.insert(w);
            }
        }
        for w in unique_in_cluster {
            *word_doc_freq.entry(w).or_insert(0) += 1;
        }
        doc_words.insert(cluster_id, words);
    }

    let num_clusters = cluster_docs.len() as f32;
    let mut result = HashMap::new();

    for (&cluster_id, words) in &doc_words {
        let total_terms = words.len().max(1) as f32;
        let mut tf_map = HashMap::new();
        for w in words {
            *tf_map.entry(w).or_insert(0usize) += 1;
        }

        let mut scores: Vec<(String, f32)> = tf_map
            .into_iter()
            .map(|(w, count)| {
                let tf = (count as f32) / total_terms;
                let df = word_doc_freq.get(w).copied().unwrap_or(1) as f32;
                let idf = ((1.0 + num_clusters) / (1.0 + df)).ln() + 1.0;
                let score = tf * idf;
                (w.clone(), score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_words: Vec<String> = scores.into_iter().take(top_n).map(|(w, _)| w).collect();
        result.insert(cluster_id, top_words);
    }

    result
}

pub fn precompute_candidate_embeddings(model_type: ModelType) -> Result<Vec<Vec<f32>>> {
    let candidate_texts: Vec<String> = CANDIDATE_COMPONENTS
        .iter()
        .map(|c| format!("{}: {}", c.name, c.responsibility))
        .collect();
    let candidate_text_refs: Vec<&str> = candidate_texts.iter().map(|s| s.as_str()).collect();
    embed_single_texts(&candidate_text_refs, model_type)
}

pub fn label_clusters_with_candidates(
    items: &[EmbeddedItem],
    labels: &[usize],
    candidate_embeddings: &[Vec<f32>],
) -> Result<Vec<ClusterComponent>> {
    assert_eq!(items.len(), labels.len());

    let mut cluster_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &lbl) in labels.iter().enumerate() {
        cluster_indices.entry(lbl).or_default().push(idx);
    }

    // Prepare texts per cluster for c-TF-IDF
    let mut cluster_docs: HashMap<usize, Vec<&str>> = HashMap::new();
    for (&cid, indices) in &cluster_indices {
        let texts: Vec<&str> = indices.iter().map(|&i| items[i].text.as_str()).collect();
        cluster_docs.insert(cid, texts);
    }
    let cluster_keywords = extract_cluster_keywords(&cluster_docs, 5);

    let mut components = Vec::new();

    // Track used candidate names to avoid duplicate assignment if possible
    let mut used_candidates = HashSet::new();

    let mut sorted_cids: Vec<usize> = cluster_indices.keys().copied().collect();
    sorted_cids.sort_unstable();

    for cid in sorted_cids {
        let indices = &cluster_indices[&cid];
        let n_c = indices.len();

        // 1. Compute Centroid
        let dim = items[0].embedding.len();
        let mut centroid = vec![0.0f32; dim];
        for &i in indices {
            for (d, val) in items[i].embedding.iter().enumerate() {
                centroid[d] += val;
            }
        }
        for d in 0..dim {
            centroid[d] /= n_c as f32;
        }

        // 2. Intra-cluster cohesion (mean pairwise similarity)
        let mut sim_sum = 0.0f32;
        let mut sim_pairs = 0usize;
        for i_idx in 0..n_c {
            for j_idx in i_idx + 1..n_c {
                let u = indices[i_idx];
                let v = indices[j_idx];
                sim_sum += cosine_similarity(&items[u].embedding, &items[v].embedding);
                sim_pairs += 1;
            }
        }
        let cohesion = if sim_pairs > 0 {
            sim_sum / (sim_pairs as f32)
        } else {
            1.0
        };

        // 3. Find Medoid (point with highest sum of similarities to all members)
        let mut best_medoid = indices[0];
        let mut max_medoid_sim = -1.0f32;
        for &u in indices {
            let sum_sim: f32 = indices
                .iter()
                .map(|&v| cosine_similarity(&items[u].embedding, &items[v].embedding))
                .sum();
            if sum_sim > max_medoid_sim {
                max_medoid_sim = sum_sim;
                best_medoid = u;
            }
        }

        // 4. Match against Candidate Components
        let mut candidate_sims: Vec<(usize, f32)> = candidate_embeddings
            .iter()
            .enumerate()
            .map(|(c_idx, c_emb)| (c_idx, cosine_similarity(&centroid, c_emb)))
            .collect();
        candidate_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Prefer unused candidates if similarity is competitive
        let best_candidate_idx = candidate_sims
            .iter()
            .find(|(idx, _)| !used_candidates.contains(idx))
            .map(|(idx, _)| *idx)
            .unwrap_or(candidate_sims[0].0);

        used_candidates.insert(best_candidate_idx);
        let matched = &CANDIDATE_COMPONENTS[best_candidate_idx];

        let mut req_ids: Vec<String> = indices.iter().map(|&i| items[i].id.clone()).collect();
        req_ids.sort();

        let mut source_pdf_ids: Vec<u32> = indices
            .iter()
            .flat_map(|&i| items[i].source_ids.clone())
            .collect();
        source_pdf_ids.sort_unstable();
        source_pdf_ids.dedup();

        components.push(ClusterComponent {
            cluster_id: cid,
            name: matched.name.to_string(),
            responsibility: matched.responsibility.to_string(),
            cohesion,
            medoid_id: items[best_medoid].id.clone(),
            medoid_text: items[best_medoid].text.clone(),
            keywords: cluster_keywords.get(&cid).cloned().unwrap_or_default(),
            requirement_ids: req_ids,
            source_pdf_ids,
        });
    }

    Ok(components)
}

pub fn label_clusters(
    items: &[EmbeddedItem],
    labels: &[usize],
    model_type: ModelType,
) -> Result<Vec<ClusterComponent>> {
    let candidate_embeddings = precompute_candidate_embeddings(model_type)?;
    label_clusters_with_candidates(items, labels, &candidate_embeddings)
}
