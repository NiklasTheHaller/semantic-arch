pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 1e-12 && norm_b > 1e-12 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    (1.0 - cosine_similarity(a, b)).max(0.0)
}

/// Computes the mean Silhouette Score for a set of points and their cluster labels.
/// Range: [-1.0, 1.0], where higher indicates denser, better-separated clusters.
pub fn silhouette_score(points: &[Vec<f32>], labels: &[usize]) -> f32 {
    let n = points.len();
    if n <= 1 {
        return 0.0;
    }

    let mut unique_labels: Vec<usize> = labels.to_vec();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    if unique_labels.len() <= 1 || unique_labels.len() == n {
        return 0.0;
    }

    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let label_i = labels[i];
        let mut same_cluster_dist_sum = 0.0f32;
        let mut same_cluster_count = 0usize;

        // Distances to other clusters
        let mut other_cluster_dist_sum = std::collections::HashMap::<usize, f32>::new();
        let mut other_cluster_counts = std::collections::HashMap::<usize, usize>::new();

        for j in 0..n {
            if i == j {
                continue;
            }
            let dist = cosine_distance(&points[i], &points[j]);
            if labels[j] == label_i {
                same_cluster_dist_sum += dist;
                same_cluster_count += 1;
            } else {
                *other_cluster_dist_sum.entry(labels[j]).or_insert(0.0) += dist;
                *other_cluster_counts.entry(labels[j]).or_insert(0) += 1;
            }
        }

        let a_i = if same_cluster_count > 0 {
            same_cluster_dist_sum / (same_cluster_count as f32)
        } else {
            0.0
        };

        let mut b_i = f32::INFINITY;
        for (lbl, sum_dist) in other_cluster_dist_sum {
            let count = other_cluster_counts[&lbl];
            if count > 0 {
                let mean_dist = sum_dist / (count as f32);
                if mean_dist < b_i {
                    b_i = mean_dist;
                }
            }
        }

        let s_i = if a_i.max(b_i) > 1e-12 {
            (b_i - a_i) / a_i.max(b_i)
        } else {
            0.0
        };

        scores.push(s_i);
    }

    scores.iter().sum::<f32>() / (n as f32)
}

/// Computes the Adjusted Rand Index (ARI) between two clusterings.
/// Range: [-1.0, 1.0], where 1.0 indicates identical clusterings and 0.0 indicates random chance.
pub fn adjusted_rand_index(labels1: &[usize], labels2: &[usize]) -> f32 {
    assert_eq!(labels1.len(), labels2.len());
    let n = labels1.len();
    if n <= 1 {
        return 1.0;
    }

    let mut contingency = std::collections::HashMap::<(usize, usize), usize>::new();
    let mut sum_a = std::collections::HashMap::<usize, usize>::new();
    let mut sum_b = std::collections::HashMap::<usize, usize>::new();

    for (&l1, &l2) in labels1.iter().zip(labels2.iter()) {
        *contingency.entry((l1, l2)).or_insert(0) += 1;
        *sum_a.entry(l1).or_insert(0) += 1;
        *sum_b.entry(l2).or_insert(0) += 1;
    }

    let comb2 = |k: usize| -> f64 {
        if k < 2 {
            0.0
        } else {
            (k as f64) * ((k - 1) as f64) / 2.0
        }
    };

    let sum_comb_c: f64 = contingency.values().map(|&v| comb2(v)).sum();
    let sum_comb_a: f64 = sum_a.values().map(|&v| comb2(v)).sum();
    let sum_comb_b: f64 = sum_b.values().map(|&v| comb2(v)).sum();
    let total_comb = comb2(n);

    let expected_index = (sum_comb_a * sum_comb_b) / total_comb;
    let max_index = (sum_comb_a + sum_comb_b) / 2.0;

    let denominator = max_index - expected_index;
    if denominator.abs() < 1e-12 {
        1.0
    } else {
        ((sum_comb_c - expected_index) / denominator) as f32
    }
}
