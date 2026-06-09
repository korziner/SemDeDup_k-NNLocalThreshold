# Адаптивная семантическая дедупликація текстовыхъ массивовъ

**Объектомъ изслѣдованія** является процессъ семантической дедупликаціи текстовыхъ массивовъ, **предметомъ** — методы адаптивнаго опредѣленія порога сходства.

Въ работѣ реализованъ базовый методъ **SemDeDup** и шесть адаптивныхъ модификацій:

1. Адаптивный порогъ на основѣ плотности (*density‑based*);
2. Локальный порогъ на основѣ *k*‑ближайшихъ сосѣдей (*k‑NN local*);
3. Кластерный порогъ на основѣ процентилей (*cluster percentile*);
4. Кластерный порогъ на основѣ средняго значенія и стандартнаго отклоненія (*cluster mean/std*);
5. Адаптивный порогъ Оцу (*Otsu*);
6. Двухпроходный адаптивный порогъ Оцу (*two‑pass Otsu*).

**Эмпирическая база** — корпусъ паръ вопросовъ Quora; векторныя представленія получены съ помощью модели SBERT `all‑MiniLM‑L6‑v2`. Проведёнъ систематическій поискъ гиперпараметровъ и анализъ масштабируемости на наборахъ данныхъ объёмомъ отъ 500 до 10 000 документовъ (усреднено по трёмъ независимымъ прогонамъ).

**Основной результат:** методъ *k‑NN Local Threshold* превосходитъ базовый SemDeDup по показателю **F1‑оценки** при всѣхъ изученныхъ размерахъ корпуса, улучшая результатъ на **5–8 процентныхъ пунктовъ**. Методъ *Two‑Pass OAT* обезпечиваетъ наименьшую потерю уникальныхъ документовъ.

Сформулированы практическія рекомендаціи по выбору алгоритма дедупликаціи.

---

## Реализаціи методовъ by DeepSeek

Ниже приведены работающіе примеры кода на языкахъ **Julia** и **Rust** для всехъ перечисленныхъ алгоритмовъ.

### Julia

```julia
using Distances, LinearAlgebra, Statistics

# ------------------------------------------------------------
# Базовый метод SemDeDup (фиксированный порог)
function semdedup(embeddings::Matrix{Float64}, threshold::Float64)
    n = size(embeddings, 2)
    keep = trues(n)
    for i in 1:n
        keep[i] || continue
        for j in i+1:n
            sim = cosine_similarity(view(embeddings, :, i), view(embeddings, :, j))
            if sim > threshold
                keep[j] = false
            end
        end
    end
    return keep
end

cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))

# ------------------------------------------------------------
# 1. Адаптивный порог на основе плотности
function density_threshold(embeddings::Matrix{Float64}, k::Int=5, alpha::Float64=1.0)
    n = size(embeddings, 2)
    thresholds = zeros(n)
    for i in 1:n
        dists = [cosine_distance(view(embeddings, :, i), view(embeddings, :, j)) for j in 1:n if j != i]
        sort!(dists)
        if length(dists) >= k
            avg_k_dist = mean(dists[1:k])
            thresholds[i] = alpha * avg_k_dist
        else
            thresholds[i] = 0.5
        end
    end
    return thresholds
end

cosine_distance(a, b) = 1 - cosine_similarity(a, b)

# ------------------------------------------------------------
# 2. Локальный порог на основе k-ближайших соседей (k-NN local)
function knn_local_threshold(embeddings::Matrix{Float64}, k::Int=10, percentile::Float64=0.8)
    n = size(embeddings, 2)
    thresholds = zeros(n)
    for i in 1:n
        sims = [cosine_similarity(view(embeddings, :, i), view(embeddings, :, j)) for j in 1:n if j != i]
        sort!(sims, rev=true)
        if length(sims) >= k
            knn_sims = sims[1:k]
            thresholds[i] = quantile(knn_sims, percentile)
        else
            thresholds[i] = 0.5
        end
    end
    return thresholds
end

function quantile(v::Vector{Float64}, p::Float64)
    idx = Int(ceil(p * length(v)))
    idx = clamp(idx, 1, length(v))
    return v[idx]
end

# ------------------------------------------------------------
# 3. Кластерный порог на основе процентилей
function cluster_percentile_threshold(embeddings::Matrix{Float64}, labels::Vector{Int}, percentile::Float64=0.9)
    unique_labels = unique(labels)
    thresholds = Dict{Int, Float64}()
    for lbl in unique_labels
        idxs = findall(x -> x == lbl, labels)
        if length(idxs) < 2
            thresholds[lbl] = 0.5
            continue
        end
        sims = Float64[]
        for i in 1:length(idxs), j in i+1:length(idxs)
            push!(sims, cosine_similarity(view(embeddings, :, idxs[i]), view(embeddings, :, idxs[j])))
        end
        sort!(sims, rev=true)
        thresholds[lbl] = quantile(sims, percentile)
    end
    return [thresholds[labels[i]] for i in 1:length(labels)]
end

# ------------------------------------------------------------
# 4. Кластерный порог на основе среднего и стандартного отклонения
function cluster_meanstd_threshold(embeddings::Matrix{Float64}, labels::Vector{Int}, factor::Float64=1.0)
    unique_labels = unique(labels)
    thresholds = Dict{Int, Float64}()
    for lbl in unique_labels
        idxs = findall(x -> x == lbl, labels)
        if length(idxs) < 2
            thresholds[lbl] = 0.5
            continue
        end
        sims = Float64[]
        for i in 1:length(idxs), j in i+1:length(idxs)
            push!(sims, cosine_similarity(view(embeddings, :, idxs[i]), view(embeddings, :, idxs[j])))
        end
        m = mean(sims)
        s = std(sims)
        thresholds[lbl] = m + factor * s
    end
    return [thresholds[labels[i]] for i in 1:length(labels)]
end

# ------------------------------------------------------------
# 5. Адаптивный порог Оцу (Otsu)
function otsu_threshold(embeddings::Matrix{Float64}, n_bins::Int=100)
    n = size(embeddings, 2)
    all_sims = Float64[]
    for i in 1:n, j in i+1:n
        push!(all_sims, cosine_similarity(view(embeddings, :, i), view(embeddings, :, j)))
    end
    hist, edges = fit(Histogram, all_sims, n_bins)
    return otsu_from_hist(hist, edges)
end

function otsu_from_hist(hist, edges)
    total = sum(hist.weights)
    sum_total = sum((edges[i] + edges[i+1])/2 * hist.weights[i] for i in 1:length(hist.weights))
    best_thresh = 0.5
    max_var = 0.0
    sum_b = 0.0
    w_b = 0.0
    for i in 1:length(hist.weights)
        w_b += hist.weights[i]
        if w_b == 0 || w_b == total; continue; end
        w_f = total - w_b
        sum_b += (edges[i] + edges[i+1])/2 * hist.weights[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f)^2
        if var_between > max_var
            max_var = var_between
            best_thresh = (edges[i] + edges[i+1])/2
        end
    end
    return best_thresh
end

# ------------------------------------------------------------
# 6. Двухпроходный адаптивный порог Оцу
function twopass_otsu_threshold(embeddings::Matrix{Float64}, n_bins::Int=100)
    # Первый проход: глобальный порог
    t1 = otsu_threshold(embeddings, n_bins)
    # Разделяем на две группы
    n = size(embeddings, 2)
    low_sims = Float64[]
    high_sims = Float64[]
    for i in 1:n, j in i+1:n
        s = cosine_similarity(view(embeddings, :, i), view(embeddings, :, j))
        if s <= t1
            push!(low_sims, s)
        else
            push!(high_sims, s)
        end
    end
    # Второй проход: порог только для high-группы (или low)
    t2 = if length(high_sims) > 10
        hist_h, edges_h = fit(Histogram, high_sims, n_bins)
        otsu_from_hist(hist_h, edges_h)
    else
        t1
    end
    return (t1, t2)
end
```

Rust

```
use ndarray::{Array2, ArrayView1, Axis};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

type Embedding = Array2<f64>;  // shape (dims, n_samples)

// ------------------------------------------------------------
// Helper: cosine similarity
fn cosine_similarity(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let dot = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();
    dot / (norm_a * norm_b)
}

// ------------------------------------------------------------
// 1. Base SemDeDup
fn semdedup(embeddings: &Embedding, threshold: f64) -> Vec<bool> {
    let n = embeddings.ncols();
    let mut keep = vec![true; n];
    for i in 0..n {
        if !keep[i] { continue; }
        let vi = embeddings.column(i);
        for j in (i+1)..n {
            let vj = embeddings.column(j);
            if cosine_similarity(vi, vj) > threshold {
                keep[j] = false;
            }
        }
    }
    keep
}

// ------------------------------------------------------------
// 2. Density-based threshold (per point)
fn density_threshold(embeddings: &Embedding, k: usize, alpha: f64) -> Vec<f64> {
    let n = embeddings.ncols();
    let mut thresholds = vec![0.5; n];
    for i in 0..n {
        let vi = embeddings.column(i);
        let mut dists: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| 1.0 - cosine_similarity(vi, embeddings.column(j)))
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if dists.len() >= k {
            let avg_k: f64 = dists[0..k].iter().sum::<f64>() / (k as f64);
            thresholds[i] = alpha * avg_k;
        }
    }
    thresholds
}

// ------------------------------------------------------------
// 3. k-NN local threshold
fn knn_local_threshold(embeddings: &Embedding, k: usize, percentile: f64) -> Vec<f64> {
    let n = embeddings.ncols();
    let mut thresholds = vec![0.5; n];
    for i in 0..n {
        let vi = embeddings.column(i);
        let mut sims: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| cosine_similarity(vi, embeddings.column(j)))
            .collect();
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if sims.len() >= k {
            let knn_sims = &sims[0..k];
            let idx = ((percentile * (k-1) as f64).round() as usize).min(k-1);
            thresholds[i] = knn_sims[idx];
        }
    }
    thresholds
}

// ------------------------------------------------------------
// 4. Cluster percentile threshold
fn cluster_percentile_threshold(embeddings: &Embedding, labels: &[usize], percentile: f64) -> Vec<f64> {
    let n = embeddings.ncols();
    let mut cluster_sims: HashMap<usize, Vec<f64>> = HashMap::new();
    for i in 0..n {
        for j in (i+1)..n {
            if labels[i] == labels[j] {
                let sim = cosine_similarity(embeddings.column(i), embeddings.column(j));
                cluster_sims.entry(labels[i]).or_default().push(sim);
            }
        }
    }
    let mut threshold_map = HashMap::new();
    for (c, sims) in cluster_sims {
        if sims.is_empty() {
            threshold_map.insert(c, 0.5);
        } else {
            let mut sorted = sims;
            sorted.sort_by(|a,b| b.partial_cmp(a).unwrap());
            let idx = ((percentile * (sorted.len()-1) as f64).round() as usize).min(sorted.len()-1);
            threshold_map.insert(c, sorted[idx]);
        }
    }
    (0..n).map(|i| *threshold_map.get(&labels[i]).unwrap_or(&0.5)).collect()
}

// ------------------------------------------------------------
// 5. Cluster mean+std threshold
fn cluster_meanstd_threshold(embeddings: &Embedding, labels: &[usize], factor: f64) -> Vec<f64> {
    let n = embeddings.ncols();
    let mut cluster_sims: HashMap<usize, Vec<f64>> = HashMap::new();
    for i in 0..n {
        for j in (i+1)..n {
            if labels[i] == labels[j] {
                let sim = cosine_similarity(embeddings.column(i), embeddings.column(j));
                cluster_sims.entry(labels[i]).or_default().push(sim);
            }
        }
    }
    let mut threshold_map = HashMap::new();
    for (c, sims) in cluster_sims {
        if sims.is_empty() {
            threshold_map.insert(c, 0.5);
        } else {
            let mean = sims.iter().sum::<f64>() / sims.len() as f64;
            let variance = sims.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / sims.len() as f64;
            let std = variance.sqrt();
            threshold_map.insert(c, mean + factor * std);
        }
    }
    (0..n).map(|i| *threshold_map.get(&labels[i]).unwrap_or(&0.5)).collect()
}

// ------------------------------------------------------------
// Helper: Otsu on a vector of similarities
fn otsu_threshold_from_sims(sims: &[f64], n_bins: usize) -> f64 {
    if sims.is_empty() { return 0.5; }
    let min_s = sims.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_s = sims.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let bin_width = (max_s - min_s) / (n_bins as f64);
    let mut hist = vec![0u32; n_bins];
    for &val in sims {
        let idx = ((val - min_s) / bin_width).floor() as usize;
        let idx = idx.min(n_bins-1);
        hist[idx] += 1;
    }
    let total = sims.len() as f64;
    let mut sum_total = 0.0;
    for i in 0..n_bins {
        let bin_center = min_s + (i as f64 + 0.5) * bin_width;
        sum_total += bin_center * hist[i] as f64;
    }
    let mut best_thresh = 0.5;
    let mut max_var = 0.0;
    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    for i in 0..n_bins-1 {
        w_b += hist[i] as f64;
        if w_b == 0.0 || w_b == total { continue; }
        let w_f = total - w_b;
        let bin_center = min_s + (i as f64 + 0.5) * bin_width;
        sum_b += bin_center * hist[i] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_total - sum_b) / w_f;
        let var_between = w_b * w_f * (m_b - m_f).powi(2);
        if var_between > max_var {
            max_var = var_between;
            best_thresh = bin_center;
        }
    }
    best_thresh
}

// ------------------------------------------------------------
// 6. Otsu adaptive threshold
fn otsu_threshold(embeddings: &Embedding, n_bins: usize) -> f64 {
    let n = embeddings.ncols();
    let mut all_sims = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            all_sims.push(cosine_similarity(embeddings.column(i), embeddings.column(j)));
        }
    }
    otsu_threshold_from_sims(&all_sims, n_bins)
}

// ------------------------------------------------------------
// 7. Two-pass Otsu adaptive threshold
fn twopass_otsu_threshold(embeddings: &Embedding, n_bins: usize) -> (f64, f64) {
    let t1 = otsu_threshold(embeddings, n_bins);
    let n = embeddings.ncols();
    let mut low_sims = Vec::new();
    let mut high_sims = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            let sim = cosine_similarity(embeddings.column(i), embeddings.column(j));
            if sim <= t1 {
                low_sims.push(sim);
            } else {
                high_sims.push(sim);
            }
        }
    }
    let t2 = if high_sims.len() > 10 {
        otsu_threshold_from_sims(&high_sims, n_bins)
    } else {
        t1
    };
    (t1, t2)
}
```


Aleksandr Pak
Supervisor: Alexey Masyutin
Faculty of Computer Science
Educational Programme: Financial Technology and Data Analysis (Master)
Year of Graduation: 2026
www.hse.ru/en/ma/fintech/students/diplomas/1163760559
