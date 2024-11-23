import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from kmeans_config import MAX_ITER, N_CLUSTERS, N_FEATURES_LIST, N_SAMPLES

# Create necessary directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class ExperimentResult:
    sequential_time: float
    cuda_time: float
    speedup: float


def generate_dataset(
    n_samples: int, n_features: int, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset for clustering."""
    print(
        f"Generating dataset with {n_samples} samples, {n_features} features, and {n_clusters} clusters..."
    )
    x, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
    )
    return x, y


def save_dataset(x: np.ndarray, filename: str) -> None:
    """Save dataset to CSV file."""
    filepath = DATA_DIR / filename
    print(f"Saving dataset to '{filepath}'...")
    np.savetxt(filepath, x, delimiter=",")


def run_kmeans_cuda(input_file: str, k: int, max_iter: int) -> ExperimentResult:
    """Run k-means with different thread counts and collect results."""
    input_path = str(DATA_DIR / input_file)

    cmd = ["./cmake-build-release/kmeans_cuda", input_path, str(k), str(max_iter)]
    print(f"Executing command: {' '.join(cmd)}")
    output = subprocess.check_output(cmd).decode()
    print(f"Results:\n{output}")

    # Parse timing results
    lines = output.strip().split("\n")
    sequential_time = float(lines[0].split(": ")[1][:-2])
    cuda_time = float(lines[1].split(": ")[1][:-2])
    speedup = float(lines[2].split(": ")[1][:-1])

    return ExperimentResult(
        sequential_time=sequential_time, cuda_time=cuda_time, speedup=speedup
    )


def plot_execution_times(
    results: Dict[int, ExperimentResult], output_prefix: str
) -> None:
    """Plot execution times comparison between sequential and CUDA implementations."""
    dimensions = list(results.keys())
    sequential_times = [r.sequential_time for r in results.values()]
    cuda_times = [r.cuda_time for r in results.values()]

    plt.figure(figsize=(12, 7))

    # Plot times
    plt.plot(
        dimensions,
        sequential_times,
        marker="o",
        label="Sequential",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        dimensions, cuda_times, marker="s", label="CUDA", linewidth=2, markersize=8
    )

    plt.xlabel("Number of Dimensions")
    plt.ylabel("Execution Time (ms)")
    plt.title("K-means Clustering Execution Times Comparison")
    plt.grid(True)
    plt.legend()

    # Use log scale if the difference is too large
    if max(sequential_times) / min(cuda_times) > 100:
        plt.yscale("log")

    plt.savefig(RESULTS_DIR / f"{output_prefix}_execution_times.png")
    plt.close()


def plot_speedup(results: Dict[int, ExperimentResult], output_prefix: str) -> None:
    """Plot speedup achieved by CUDA implementation."""
    dimensions = list(results.keys())
    speedups = [r.speedup for r in results.values()]

    plt.figure(figsize=(12, 7))

    plt.plot(dimensions, speedups, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Speedup (x)")
    plt.title("CUDA K-means Speedup Analysis")
    plt.grid(True)

    plt.savefig(RESULTS_DIR / f"{output_prefix}_speedup.png")
    plt.close()


def plot_kmeans_clustering(x: np.ndarray, labels: np.ndarray, n_features: int) -> None:
    """Plot visual clustering results."""
    if n_features not in [2, 3]:
        return  # Only plot 2D and 3D data

    output_path = RESULTS_DIR / f"kmeans_clustering_{n_features}d.png"
    print(f"Plotting K-means clustering to '{output_path}'...")

    if n_features == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="viridis")
        plt.title(f"K-means Clustering ({N_CLUSTERS} clusters)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    else:  # 3D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=labels, cmap="viridis")
        ax.set_title(f"K-means Clustering ({N_CLUSTERS} clusters)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        plt.colorbar(scatter)

    plt.savefig(output_path)
    plt.close()


def main():
    """Run experiments for CUDA K-means implementation"""
    results = {}

    for n_features in N_FEATURES_LIST:
        print(f"\nRunning experiments for {n_features} dimensions...")

        # Generate and save dataset
        x, y = generate_dataset(N_SAMPLES, n_features, N_CLUSTERS)
        dataset_file = f"dataset_{n_features}d.csv"
        save_dataset(x, dataset_file)

        # Run CUDA implementation
        result = run_kmeans_cuda(dataset_file, N_CLUSTERS, MAX_ITER)
        results[n_features] = result

        # Plot clustering visualization for 2D and 3D
        if n_features <= 3:
            plot_kmeans_clustering(x, y, n_features)
    # Plot comparative results
    plot_execution_times(results, "cuda_kmeans")
    plot_speedup(results, "cuda_kmeans")

    # Save results to CSV
    results_df = pd.DataFrame.from_dict(
        {k: vars(v) for k, v in results.items()}, orient="index"
    )
    results_df.index.name = "dimensions"
    results_df.to_csv(RESULTS_DIR / "cuda_kmeans_results.csv")


if __name__ == "__main__":
    main()
