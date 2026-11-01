use crate::cluster::metrics::{cosine_distance, silhouette_score};

pub fn cluster_agglomerative(points: &[Vec<f32>], target_k: usize) -> Vec<usize> {
    let n = points.len();
    if n <= target_k {
        return (0..n).collect();
    }

    // Precompute pairwise distances
    let mut dist = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d = cosine_distance(&points[i], &points[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Each cluster holds a list of original point indices
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > target_k {
        let mut min_dist = f32::INFINITY;
        let mut best_pair = (0, 1);

        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                // Average linkage distance between cluster i and cluster j
                let mut sum_d = 0.0f32;
                for &u in &clusters[i] {
                    for &v in &clusters[j] {
                        sum_d += dist[u][v];
                    }
                }
                let avg_d = sum_d / ((clusters[i].len() * clusters[j].len()) as f32);
                if avg_d < min_dist {
                    min_dist = avg_d;
                    best_pair = (i, j);
                }
            }
        }

        // Merge cluster best_pair.1 into best_pair.0
        let (c1, c2) = best_pair;
        let mut merged = clusters.swap_remove(c2);
        clusters[c1].append(&mut merged);
    }

    let mut labels = vec![0usize; n];
    for (cluster_id, members) in clusters.iter().enumerate() {
        for &m in members {
            labels[m] = cluster_id;
        }
    }

    labels
}

pub fn find_best_k_agglomerative(
    points: &[Vec<f32>],
    min_k: usize,
    max_k: usize,
) -> (usize, Vec<usize>, f32) {
    let mut best_k = min_k;
    let mut best_labels = Vec::new();
    let mut best_score = -2.0f32;

    for k in min_k..=max_k {
        let labels = cluster_agglomerative(points, k);
        let score = silhouette_score(points, &labels);
        if score > best_score {
            best_score = score;
            best_k = k;
            best_labels = labels;
        }
    }

    (best_k, best_labels, best_score)
}
