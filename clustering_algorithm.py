# Use HDBSCAN and geographic data to cluster counties into groups affected by similar events.

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import hdbscan
from collections import Counter

# parameters
GEO_WEIGHT_ALPHA = 0.10                # geographic weight
GEOGRAPHIC_CUTOFF_KM = 500             # tighter soft distance limit
MIN_CLUSTER_SIZE = 20                 # increase to reduce fragmentation
MIN_SAMPLES = 10                      # increase to reduce sparse cores
ABSORPTION_ITERATE = True             # enable iterative absorption
ABSORPTION_MAJORITY_THRESHOLD = 0.8   # fraction of neighbors needed to dominate
MIN_FINAL_CLUSTER_SIZE = 15           # minimum size for clusters after absorption

# 1.) Load disaster data
df = pd.read_csv('county_disaster_timeline_2.csv') # binary disaster or other metric timeline (edit by timeseries format below)
disaster_cols = df.columns.difference(['fips', 'month_year'])
county_vectors = df.groupby('fips')[disaster_cols].max().astype(bool)

# 2.) Compute Jaccard distances
jaccard_dist = pairwise_distances(county_vectors.values, metric='jaccard')
jaccard_df = pd.DataFrame(jaccard_dist, index=county_vectors.index, columns=county_vectors.index)

# 3.) Load geographic distance matrix
geo_matrix = pd.read_csv('pairwise_county_distances_km.csv', index_col=0)
geo_matrix.columns = geo_matrix.columns.astype(str)
geo_matrix.index = geo_matrix.index.astype(str)
fips_list = county_vectors.index.astype(str)
geo_matrix = geo_matrix.reindex(index=fips_list, columns=fips_list)

# 4.) Blend Jaccard + normalized geo penalty
max_geo = geo_matrix.to_numpy().max()
geo_penalty = geo_matrix / max_geo
geo_penalty = geo_penalty.fillna(1.0)
geo_mask = geo_matrix <= GEOGRAPHIC_CUTOFF_KM

blended_dist = (1 - GEO_WEIGHT_ALPHA) * jaccard_df.values + GEO_WEIGHT_ALPHA * geo_penalty.values
blended_dist[~geo_mask.values] = 0.99
np.fill_diagonal(blended_dist, 0.0)

# 5.) HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    metric='precomputed',
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES
)
cluster_labels = clusterer.fit_predict(blended_dist)

# 6.) Assign initial clusters
cluster_df = pd.DataFrame({'fips': county_vectors.index.astype(str), 'cluster': cluster_labels})
cluster_map = dict(zip(cluster_df['fips'], cluster_df['cluster']))

# 7.) Load adjacency list
adj_df = pd.read_csv('county_adjacency_list.csv') # insert adjacency list here and edit columns names below
adj_df['fips'] = adj_df['fips'].astype(str)
adj_df['neighbor_fips'] = adj_df['neighbor_fips'].astype(str)
adjacency_map = adj_df.groupby('fips')['neighbor_fips'].apply(set).to_dict()

# 8.) Iterative absorption (relaxed)
total_absorbed = 0
adjusted_map = cluster_map.copy()
iteration = 0
changed = True

while ABSORPTION_ITERATE and changed:
    changed = False
    iteration += 1
    print(f"\U0001F501 Absorption iteration {iteration}...")

    new_map = adjusted_map.copy()
    absorbed_count = 0

    for fips in adjusted_map:
        current_cluster = adjusted_map.get(fips, -1)
        neighbors = adjacency_map.get(fips, set())

        neighbor_clusters = [
            adjusted_map.get(n, -1) for n in neighbors
            if adjusted_map.get(n, -1) != -1
        ]

        if not neighbor_clusters:
            continue

        cluster_counter = Counter(neighbor_clusters)
        dominant_cluster, count = cluster_counter.most_common(1)[0]
        ratio = count / len(neighbor_clusters)

        if ratio >= ABSORPTION_MAJORITY_THRESHOLD and dominant_cluster != current_cluster:
            new_map[fips] = dominant_cluster
            changed = True
            absorbed_count += 1

    adjusted_map = new_map
    total_absorbed += absorbed_count
    print(f"↪️  Absorbed {absorbed_count} counties in iteration {iteration}\n")

# 9.) Final absorption pass (strict)
print("Final enclave absorption pass (100% neighbor agreement)...")
final_map = adjusted_map.copy()
final_absorbed = 0

for fips in adjusted_map:
    current_cluster = adjusted_map.get(fips, -1)
    neighbors = adjacency_map.get(fips, set())
    neighbor_clusters = [adjusted_map.get(n, -1) for n in neighbors if adjusted_map.get(n, -1) != -1]

    if neighbor_clusters:
        unique_clusters = set(neighbor_clusters)
        if len(unique_clusters) == 1:
            sole_cluster = unique_clusters.pop()
            if sole_cluster != current_cluster:
                final_map[fips] = sole_cluster
                final_absorbed += 1

adjusted_map = final_map
print(f"✅ Final enclave pass absorbed {final_absorbed} additional counties.\n")

# 10.) Save results with post-absorption cleanup
cluster_df['adjusted_cluster'] = cluster_df['fips'].map(adjusted_map)

# 11.) Merge clusters smaller than MIN_FINAL_CLUSTER_SIZE into noise
counts = Counter(cluster_df['adjusted_cluster'])
cluster_df['adjusted_cluster'] = cluster_df['adjusted_cluster'].apply(
    lambda x: -1 if counts[x] < MIN_FINAL_CLUSTER_SIZE else x
)

cluster_df.to_csv('county_clusters_strict_softgeo_absorbed.csv', index=False)

# 12.) Print a summary
label_counts = Counter(cluster_df['adjusted_cluster'])
total_counties = len(cluster_df)
noise_count = label_counts.get(-1, 0)
noise_percent = 100 * noise_count / total_counties
if -1 in label_counts:
    del label_counts[-1]
sorted_clusters = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

print("\u2705 Clustering complete and adjusted. Saved to county_clusters_strict_softgeo_absorbed.csv.")
print(f"\U0001F4CA Total clusters (excluding noise): {len(sorted_clusters)}")
print(f"\u2757 Noise: {noise_count} counties ({noise_percent:.2f}%)")
print(f"\U0001F44D Total counties absorbed via relaxed rule: {total_absorbed}")
print(f"🧩 Additional counties absorbed via strict enclave rule: {final_absorbed}\n")

print("\U0001F53D Top 3 largest clusters:")
for i, (label, size) in enumerate(sorted_clusters[:3], 1):
    print(f"  {i}. Cluster {label}: {size} counties")

print("\n\U0001F53D Smallest 3 non-noise clusters:")
for i, (label, size) in enumerate(sorted_clusters[-3:], 1):
    print(f"  {i}. Cluster {label}: {size} counties")