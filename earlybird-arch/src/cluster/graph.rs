use petgraph::graph::UnGraph;
use std::collections::{HashMap, HashSet};

use crate::store::MatrixPair;

pub fn cluster_via_louvain(num_points: usize, pairs: &[MatrixPair], min_similarity_threshold: f32) -> Vec<usize> {
    let mut graph = UnGraph::<usize, f32>::new_undirected();
    let mut node_indices = Vec::with_capacity(num_points);

    for i in 0..num_points {
        node_indices.push(graph.add_node(i));
    }

    let mut total_weight = 0.0f32;
    let mut added_edges = HashSet::new();

    for pair in pairs {
        if pair.p1 as usize >= num_points || pair.p2 as usize >= num_points {
            continue;
        }
        if pair.p1 == pair.p2 {
            continue;
        }
        if pair.score < min_similarity_threshold {
            continue;
        }

        let edge_key = if pair.p1 < pair.p2 {
            (pair.p1, pair.p2)
        } else {
            (pair.p2, pair.p1)
        };

        if added_edges.insert(edge_key) {
            let u = node_indices[pair.p1 as usize];
            let v = node_indices[pair.p2 as usize];
            let weight = pair.score;
            graph.add_edge(u, v, weight);
            total_weight += weight;
        }
    }

    if total_weight < 1e-6 {
        // Fallback if no edges meet threshold: all in separate or single cluster
        return (0..num_points).collect();
    }

    let m2 = 2.0 * total_weight;

    // Node degrees (sum of incident edge weights)
    let mut k_degree = vec![0.0f32; num_points];
    // Neighbor adjacency map: node -> Vec<(neighbor_node, weight)>
    let mut adj = vec![Vec::new(); num_points];

    for edge in graph.edge_indices() {
        if let Some((u, v)) = graph.edge_endpoints(edge) {
            let u_idx = graph[u];
            let v_idx = graph[v];
            let w = *graph.edge_weight(edge).unwrap();
            k_degree[u_idx] += w;
            k_degree[v_idx] += w;
            adj[u_idx].push((v_idx, w));
            adj[v_idx].push((u_idx, w));
        }
    }

    // Initialize each node in its own community
    let mut community: Vec<usize> = (0..num_points).collect();
    // Sum of degrees in each community: comm_id -> sum_tot
    let mut sigma_tot: Vec<f32> = k_degree.clone();

    let max_iterations = 50;
    for _iter in 0..max_iterations {
        let mut moved = false;

        for i in 0..num_points {
            let current_comm = community[i];
            let k_i = k_degree[i];

            if k_i < 1e-6 {
                continue; // Isolated node
            }

            // Weights from i to neighboring communities
            let mut comm_weights: HashMap<usize, f32> = HashMap::new();
            for &(neighbor, w) in &adj[i] {
                let neighbor_comm = community[neighbor];
                *comm_weights.entry(neighbor_comm).or_insert(0.0) += w;
            }

            // Remove i from current community
            let _k_i_in_curr = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
            sigma_tot[current_comm] -= k_i;

            let mut best_comm = current_comm;
            let mut best_delta_q = 0.0f32;

            for (&candidate_comm, &k_i_in) in &comm_weights {
                let sig_tot = sigma_tot[candidate_comm];
                // Delta Q for moving i into candidate_comm
                // delta_q = k_i_in / (2m) - (sig_tot * k_i) / (2m^2)
                let delta_q = (k_i_in / m2) - ((sig_tot * k_i) / (m2 * m2));
                if delta_q > best_delta_q {
                    best_delta_q = delta_q;
                    best_comm = candidate_comm;
                }
            }

            // Insert i into best_comm
            community[i] = best_comm;
            sigma_tot[best_comm] += k_i;

            if best_comm != current_comm {
                moved = true;
            }
        }

        if !moved {
            break;
        }
    }

    // Remap community IDs to dense 0..num_clusters
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 0usize;
    let mut result_labels = vec![0usize; num_points];

    for i in 0..num_points {
        let comm = community[i];
        let label = *label_map.entry(comm).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        result_labels[i] = label;
    }

    result_labels
}
