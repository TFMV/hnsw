#!/bin/bash

# Benchmark runner for hybrid index
# This script runs benchmarks and generates visualizations

# Set working directory to the script location
cd "$(dirname "$0")"

# Create output directory
BENCHMARK_DIR="benchmark_results"
mkdir -p "$BENCHMARK_DIR"

# Run benchmarks
echo "Running benchmarks..."
go test -bench=. -benchmem -cpu=1 -timeout=30m > "$BENCHMARK_DIR/benchmark_results.txt"

# Run recall benchmarks separately with more iterations
echo "Running recall benchmarks..."
go test -bench=BenchmarkRecall -benchtime=10x -cpu=1 > "$BENCHMARK_DIR/recall_results.txt"

# Run latency benchmarks separately
echo "Running latency benchmarks..."
go test -bench=BenchmarkQueryLatency -benchtime=5x -cpu=1 > "$BENCHMARK_DIR/latency_results.txt"

# Generate visualization script
cat > "$BENCHMARK_DIR/visualize.py" << 'EOF'
#!/usr/bin/env python3
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Function to parse benchmark results
def parse_benchmark_results(filename):
    results = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith("Benchmark"):
            parts = line.strip().split()
            if len(parts) >= 4:
                name = parts[0]
                ops = float(parts[1])
                ns_per_op = float(parts[2])
                
                # Extract additional metrics
                metrics = {}
                for i in range(3, len(parts), 2):
                    if i+1 < len(parts) and parts[i].endswith(':'):
                        metric_name = parts[i][:-1]
                        metric_value = float(parts[i+1])
                        metrics[metric_name] = metric_value
                
                # Parse benchmark name
                name_parts = name[len("Benchmark"):].split('/')
                
                if len(name_parts) >= 2:
                    benchmark_type = name_parts[0]
                    index_type = name_parts[1]
                    
                    # Parse additional parameters
                    params = {}
                    if benchmark_type in ["HybridIndex", "Scale"]:
                        if len(name_parts) >= 4:
                            params["num_vectors"] = int(name_parts[2])
                            params["dimension"] = int(name_parts[3])
                            if len(name_parts) >= 5:
                                params["dataset_type"] = name_parts[4]
                    elif benchmark_type == "Dimension":
                        if len(name_parts) >= 3:
                            params["dimension"] = int(name_parts[2])
                    elif benchmark_type == "Build":
                        if len(name_parts) >= 3:
                            params["num_vectors"] = int(name_parts[2])
                    
                    result = {
                        "benchmark_type": benchmark_type,
                        "index_type": index_type,
                        "ops": ops,
                        "ns_per_op": ns_per_op,
                        **params,
                        **metrics
                    }
                    results.append(result)
    
    return pd.DataFrame(results)

# Parse benchmark results
results_file = "benchmark_results.txt"
recall_file = "recall_results.txt"
latency_file = "latency_results.txt"

if os.path.exists(results_file):
    df = parse_benchmark_results(results_file)
    
    # Plot search performance by dataset size
    if "num_vectors" in df.columns and "index_type" in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter for main benchmark results
        df_search = df[df["benchmark_type"] == "HybridIndex"].copy()
        if not df_search.empty:
            pivot = df_search.pivot(index="num_vectors", columns="index_type", values="ns_per_op")
            ax = pivot.plot(marker='o', loglog=True)
            plt.title("Search Performance by Dataset Size")
            plt.xlabel("Number of Vectors (log scale)")
            plt.ylabel("Nanoseconds per Operation (log scale)")
            plt.grid(True, which="both", ls="--")
            plt.savefig("search_performance_by_size.png", dpi=300, bbox_inches="tight")
        
        # Plot build time
        df_build = df[df["benchmark_type"] == "Build"].copy()
        if not df_build.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="index_type", y="ns_per_op", data=df_build)
            plt.title("Index Build Time")
            plt.xlabel("Index Type")
            plt.ylabel("Nanoseconds per Operation")
            plt.yscale("log")
            plt.savefig("build_time.png", dpi=300, bbox_inches="tight")
        
        # Plot dimensionality impact
        df_dim = df[df["benchmark_type"] == "Dimension"].copy()
        if not df_dim.empty:
            plt.figure(figsize=(12, 8))
            pivot = df_dim.pivot(index="dimension", columns="index_type", values="ns_per_op")
            ax = pivot.plot(marker='o', loglog=True)
            plt.title("Search Performance by Dimensionality")
            plt.xlabel("Dimension (log scale)")
            plt.ylabel("Nanoseconds per Operation (log scale)")
            plt.grid(True, which="both", ls="--")
            plt.savefig("search_performance_by_dimension.png", dpi=300, bbox_inches="tight")
        
        # Plot scalability
        df_scale = df[df["benchmark_type"] == "Scale"].copy()
        if not df_scale.empty:
            plt.figure(figsize=(12, 8))
            pivot = df_scale.pivot(index="num_vectors", columns="index_type", values="ns_per_op")
            ax = pivot.plot(marker='o', loglog=True)
            plt.title("Scalability: Search Time vs Dataset Size")
            plt.xlabel("Number of Vectors (log scale)")
            plt.ylabel("Nanoseconds per Operation (log scale)")
            plt.grid(True, which="both", ls="--")
            plt.savefig("scalability.png", dpi=300, bbox_inches="tight")

# Parse recall results
if os.path.exists(recall_file):
    df_recall = parse_benchmark_results(recall_file)
    
    if "recall" in df_recall.columns and "index_type" in df_recall.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="index_type", y="recall", data=df_recall)
        plt.title("Recall Comparison")
        plt.xlabel("Index Type")
        plt.ylabel("Recall (higher is better)")
        plt.ylim(0, 1.0)
        plt.savefig("recall_comparison.png", dpi=300, bbox_inches="tight")

# Parse latency results
if os.path.exists(latency_file):
    df_latency = parse_benchmark_results(latency_file)
    
    if "p50_ns" in df_latency.columns and "p95_ns" in df_latency.columns and "p99_ns" in df_latency.columns:
        plt.figure(figsize=(12, 8))
        
        # Reshape data for grouped bar chart
        percentiles = ["p50_ns", "p95_ns", "p99_ns"]
        index_types = df_latency["index_type"].unique()
        
        data = []
        for idx_type in index_types:
            row = df_latency[df_latency["index_type"] == idx_type].iloc[0]
            for p in percentiles:
                data.append({
                    "index_type": idx_type,
                    "percentile": p.replace("_ns", ""),
                    "latency_ns": row[p]
                })
        
        df_plot = pd.DataFrame(data)
        
        # Plot grouped bar chart
        sns.barplot(x="index_type", y="latency_ns", hue="percentile", data=df_plot)
        plt.title("Query Latency Distribution")
        plt.xlabel("Index Type")
        plt.ylabel("Latency (ns)")
        plt.yscale("log")
        plt.savefig("latency_distribution.png", dpi=300, bbox_inches="tight")

print("Visualizations generated!")
EOF

# Make visualization script executable
chmod +x "$BENCHMARK_DIR/visualize.py"

# Check if Python and required packages are available
if command -v python3 &>/dev/null; then
    echo "Checking Python dependencies..."
    REQUIRED_PACKAGES="pandas matplotlib seaborn numpy"
    MISSING_PACKAGES=""
    
    for package in $REQUIRED_PACKAGES; do
        if ! python3 -c "import $package" &>/dev/null; then
            MISSING_PACKAGES="$MISSING_PACKAGES $package"
        fi
    done
    
    if [ -n "$MISSING_PACKAGES" ]; then
        echo "Missing Python packages:$MISSING_PACKAGES"
        echo "Install them with: pip install$MISSING_PACKAGES"
        echo "Then run: python3 $BENCHMARK_DIR/visualize.py"
    else
        echo "Generating visualizations..."
        cd "$BENCHMARK_DIR" && python3 visualize.py
        echo "Benchmark results and visualizations available in $BENCHMARK_DIR"
    fi
else
    echo "Python 3 not found. Install Python 3 and required packages to generate visualizations."
    echo "Then run: python3 $BENCHMARK_DIR/visualize.py"
fi

echo "Benchmark complete!" 