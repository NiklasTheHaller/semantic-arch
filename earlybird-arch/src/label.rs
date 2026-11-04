use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::cluster::metrics::cosine_similarity;
use crate::embed::EmbeddedItem;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamingMode {
    #[serde(rename = "archetype", alias = "SoftwareArchetype")]
    SoftwareArchetype,
    #[serde(rename = "inferred", alias = "PureInferred")]
    PureInferred,
}

impl Default for NamingMode {
    fn default() -> Self {
        NamingMode::SoftwareArchetype
    }
}

impl NamingMode {
    pub fn from_str_opt(s: Option<&str>) -> Self {
        match s {
            Some(m)
                if m.eq_ignore_ascii_case("inferred")
                    || m.eq_ignore_ascii_case("pureinferred")
                    || m.eq_ignore_ascii_case("data_driven") =>
            {
                NamingMode::PureInferred
            }
            _ => NamingMode::SoftwareArchetype,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            NamingMode::SoftwareArchetype => "archetype",
            NamingMode::PureInferred => "inferred",
        }
    }
}

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
    #[serde(default)]
    pub archetype: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentEdge {
    pub from: usize,
    pub to: usize,
    pub strength: f32,
    pub label: String,
}

/// Generic, domain-agnostic English stop words and requirements specification boilerplates
const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "won't", "would", "wouldn't", "you",
    "your", "yours", "yourself", "yourselves",
    // Generic requirements meta-words (avoid uninformative cluster labels)
    "system", "application", "software", "shall", "must", "should", "will", "would", "able",
    "allows", "allow", "allowing", "allowed", "provides", "provide", "providing", "user", "users",
    "also", "using", "use", "used", "uses", "given", "based", "e.g", "i.e", "etc", "per", "one",
    "two", "three", "four", "five", "well", "like", "many", "various", "different", "number",
    "lets", "let", "without", "new", "current", "currently", "way", "placed", "parts",
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

fn title_case(word: &str) -> String {
    let lower = word.to_lowercase();
    match lower.as_str() {
        "sms" => "SMS".to_string(),
        "api" => "API".to_string(),
        "ui" => "UI".to_string(),
        "id" => "ID".to_string(),
        "pdf" => "PDF".to_string(),
        "url" => "URL".to_string(),
        "db" => "DB".to_string(),
        "iot" => "IoT".to_string(),
        "http" => "HTTP".to_string(),
        _ => {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        }
    }
}

fn format_keyword_phrase(keywords: &[String], max_count: usize) -> String {
    let selected: Vec<String> = keywords
        .iter()
        .take(max_count)
        .map(|k| title_case(k))
        .collect();

    match selected.len() {
        0 => "Core".to_string(),
        1 => selected[0].clone(),
        2 => format!("{} & {}", selected[0], selected[1]),
        _ => format!("{}, {} & {}", selected[0], selected[1], selected[2]),
    }
}

/// Identifies action verb stems in sentences and maps them to operational nouns
const ACTION_STEMS: &[(&str, &str)] = &[
    ("validat", "Validation"),
    ("authenticat", "Authentication"),
    ("verif", "Verification"),
    ("calculat", "Calculation"),
    ("assembl", "Assembly"),
    ("pack", "Packaging"),
    ("deliver", "Delivery"),
    ("itinerar", "Routing"),
    ("rout", "Routing"),
    ("dispatch", "Dispatching"),
    ("cancel", "Cancellation"),
    ("search", "Search & Discovery"),
    ("query", "Inquiry"),
    ("inqui", "Inquiry"),
    ("report", "Reporting"),
    ("print", "Document Generation"),
    ("generat", "Generation"),
    ("transfer", "Transfer & Sync"),
    ("translat", "Processing"),
    ("pars", "Parsing & Ingestion"),
    ("track", "Tracking"),
    ("monitor", "Monitoring"),
    ("coordinat", "Coordination"),
    ("schedul", "Scheduling"),
    ("process", "Processing"),
    ("execut", "Execution"),
    ("notif", "Notification"),
    ("integrat", "Integration"),
    ("synchroniz", "Synchronization"),
    ("authoriz", "Authorization"),
    ("stor", "Storage"),
    ("record", "Recording"),
    ("configur", "Configuration"),
    ("ingest", "Ingestion"),
    ("transmitt", "Transmission"),
    ("transmit", "Transmission"),
    ("allocat", "Allocation"),
    ("publish", "Publishing"),
    ("subscrib", "Subscription"),
    ("pay", "Payment Processing"),
    ("bill", "Billing"),
];

fn extract_action_noun(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    for &(stem, noun) in ACTION_STEMS {
        if lower.contains(stem) {
            return Some(noun);
        }
    }
    None
}

/// Structural software engineering archetypes (Pattern Mode)
struct ArchetypeDef {
    kind: &'static str,
    suffix: &'static str,
    role_description: &'static str,
    tokens: &'static [&'static str],
}

const ARCHETYPES: &[ArchetypeDef] = &[
    ArchetypeDef {
        kind: "Gateway",
        suffix: "Integration Gateway",
        role_description: "External protocol adaptation, message ingestion, format translation, and third-party integration",
        tokens: &[
            "sms", "message", "messages", "gateway", "adapter", "api", "http", "external",
            "interface", "remote", "socket", "protocol", "format", "transmit", "transmission",
            "integration", "device", "sensor", "inbound", "outbound", "webhook", "network", "third",
        ],
    },
    ArchetypeDef {
        kind: "Repository",
        suffix: "Catalog & Repository",
        role_description: "Persistent entity storage, catalog definition, inventory indexing, and structured lookup queries",
        tokens: &[
            "catalog", "repository", "store", "database", "db", "record", "records", "entity",
            "entities", "item", "items", "inventory", "archive", "persist", "persistence",
            "lookup", "registry", "product", "products", "unit", "price", "pricing", "composite",
        ],
    },
    ArchetypeDef {
        kind: "Manager",
        suffix: "Workflow & Lifecycle Manager",
        role_description: "Finite state machine, business process orchestration, stage progression, and status tracking",
        tokens: &[
            "workflow", "lifecycle", "state", "status", "manager", "orchestrat", "pipeline",
            "progress", "stage", "cancel", "cancellation", "transition", "step", "process",
            "itinerary", "route", "routing", "dispatch", "fulfill", "fulfillment", "assembly",
            "pack", "packing", "deliver", "delivery", "clerk",
        ],
    },
    ArchetypeDef {
        kind: "Policy",
        suffix: "Verification & Policy Engine",
        role_description: "Identity verification, security credential validation, role permissions, and domain invariant enforcement",
        tokens: &[
            "auth", "authenticate", "authentication", "authoriz", "authorization", "role",
            "roles", "permission", "permissions", "password", "login", "token", "checksum",
            "blacklist", "whitelist", "verif", "verification", "validat", "validation",
            "policy", "security", "access", "rule", "rules", "guard",
        ],
    },
    ArchetypeDef {
        kind: "Analytics",
        suffix: "Reporting & Analytics Service",
        role_description: "Aggregating operational metrics, managerial audit logging, business intelligence, and scheduled reports",
        tokens: &[
            "report", "reports", "reporting", "analytic", "analytics", "metric", "metrics",
            "statistic", "statistics", "nightly", "daily", "audit", "history", "summary",
            "aggregate", "dashboard", "managerial",
        ],
    },
    ArchetypeDef {
        kind: "Service",
        suffix: "Domain Service",
        role_description: "Core business logic, domain transaction coordination, and business capability execution",
        tokens: &[
            "service", "operation", "action", "function", "business", "cart", "blueprint",
            "order", "orders", "placement", "checkout", "compose", "composition",
        ],
    },
];

fn classify_archetype(cluster_texts: &[&str], keywords: &[String]) -> &'static ArchetypeDef {
    let mut scores = vec![0usize; ARCHETYPES.len()];

    for text in cluster_texts {
        let words = extract_words(text);
        for w in words {
            for (i, arch) in ARCHETYPES.iter().enumerate() {
                if arch.tokens.iter().any(|&t| w.contains(t) || t.contains(&w)) {
                    scores[i] += 1;
                }
            }
        }
    }

    for kw in keywords {
        let kw_lower = kw.to_lowercase();
        for (i, arch) in ARCHETYPES.iter().enumerate() {
            if arch.tokens.iter().any(|&t| kw_lower.contains(t) || t.contains(&kw_lower)) {
                // c-TF-IDF keywords weighted higher
                scores[i] += 3;
            }
        }
    }

    let mut best_idx = 5; // Default to "Domain Service"
    let mut max_score = 0;
    for (i, &sc) in scores.iter().enumerate() {
        if sc > max_score {
            max_score = sc;
            best_idx = i;
        }
    }

    &ARCHETYPES[best_idx]
}

/// Synthesizes a component name and responsibility dynamically without hardcoded candidates
fn synthesize_component(
    cluster_texts: &[&str],
    keywords: &[String],
    medoid_id: &str,
    medoid_text: &str,
    req_count: usize,
    mode: NamingMode,
) -> (String, String, String) {
    let keyword_phrase = format_keyword_phrase(keywords, 2);
    let action_noun = extract_action_noun(medoid_text)
        .or_else(|| {
            for t in cluster_texts {
                if let Some(act) = extract_action_noun(t) {
                    return Some(act);
                }
            }
            None
        })
        .unwrap_or("Management");

    match mode {
        NamingMode::SoftwareArchetype => {
            let arch = classify_archetype(cluster_texts, keywords);
            let name = format!("{} {}", keyword_phrase, arch.suffix);
            let responsibility = format!(
                "{}. Encapsulates {} requirements (e.g. [{}]: \"{}\").",
                arch.role_description, req_count, medoid_id, medoid_text
            );
            (name, responsibility, arch.kind.to_string())
        }
        NamingMode::PureInferred => {
            let name = format!("{} {} Component", keyword_phrase, action_noun);
            let responsibility = format!(
                "Cohesive cluster of {} requirements centered on operational {} of {}. Primary medoid [{}]: \"{}\".",
                req_count, action_noun.to_lowercase(), keywords.join(", "), medoid_id, medoid_text
            );
            (name, responsibility, "Inferred".to_string())
        }
    }
}

/// Discovers inter-component edges by measuring cross-cluster semantic coupling
pub fn compute_component_edges(
    items: &[EmbeddedItem],
    labels: &[usize],
    components: &[ClusterComponent],
) -> Vec<ComponentEdge> {
    let mut cluster_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &lbl) in labels.iter().enumerate() {
        cluster_indices.entry(lbl).or_default().push(idx);
    }

    let cids: Vec<usize> = components.iter().map(|c| c.cluster_id).collect();
    let num_comp = cids.len();
    let mut raw_edges = Vec::new();

    for i in 0..num_comp {
        for j in (i + 1)..num_comp {
            let cid_a = cids[i];
            let cid_b = cids[j];
            let idxs_a = match cluster_indices.get(&cid_a) {
                Some(v) => v,
                None => continue,
            };
            let idxs_b = match cluster_indices.get(&cid_b) {
                Some(v) => v,
                None => continue,
            };

            let mut sum_sim = 0.0f32;
            let mut count = 0usize;
            for &u in idxs_a {
                for &v in idxs_b {
                    sum_sim += cosine_similarity(&items[u].embedding, &items[v].embedding);
                    count += 1;
                }
            }

            if count > 0 {
                let mean_sim = sum_sim / (count as f32);
                raw_edges.push((cid_a, cid_b, mean_sim));
            }
        }
    }

    // Sort edges by similarity strength descending
    raw_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Keep top meaningful couplings (at most 2 * num_comp, and >= 0.20 similarity)
    let max_edges = (num_comp * 2).max(4);
    raw_edges
        .into_iter()
        .filter(|&(_, _, sim)| sim >= 0.20)
        .take(max_edges)
        .map(|(from, to, sim)| ComponentEdge {
            from,
            to,
            strength: sim,
            label: format!("coupling: {:.2}", sim),
        })
        .collect()
}

pub fn label_clusters(
    items: &[EmbeddedItem],
    labels: &[usize],
    mode: NamingMode,
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
    let mut sorted_cids: Vec<usize> = cluster_indices.keys().copied().collect();
    sorted_cids.sort_unstable();

    for cid in sorted_cids {
        let indices = &cluster_indices[&cid];
        let n_c = indices.len();
        let texts = &cluster_docs[&cid];

        // 1. Intra-cluster cohesion (mean pairwise similarity)
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

        // 2. Find Medoid (point with highest sum of similarities to all members)
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

        let kw = cluster_keywords.get(&cid).cloned().unwrap_or_default();
        let (name, responsibility, archetype_kind) = synthesize_component(
            texts,
            &kw,
            &items[best_medoid].id,
            &items[best_medoid].text,
            n_c,
            mode,
        );

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
            name,
            responsibility,
            cohesion,
            medoid_id: items[best_medoid].id.clone(),
            medoid_text: items[best_medoid].text.clone(),
            keywords: kw,
            requirement_ids: req_ids,
            source_pdf_ids,
            archetype: Some(archetype_kind),
        });
    }

    Ok(components)
}
