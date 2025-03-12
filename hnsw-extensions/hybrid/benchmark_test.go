package hybrid

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/TFMV/hnsw"
)

// BenchmarkConfig holds configuration for benchmarks
type BenchmarkConfig struct {
	NumVectors    int     // Number of vectors in the dataset
	Dimension     int     // Dimension of vectors
	QueryCount    int     // Number of queries to run
	K             int     // Number of nearest neighbors to retrieve
	Seed          int64   // Random seed for reproducibility
	DatasetType   string  // Type of dataset: "random", "clustered", "skewed"
	ClusterCount  int     // Number of clusters for clustered data
	ClusterRadius float32 // Radius of clusters
	SkewFactor    float32 // Skew factor for skewed data
}

// DefaultBenchmarkConfig returns a default benchmark configuration
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		NumVectors:    10000,
		Dimension:     128,
		QueryCount:    100,
		K:             10,
		Seed:          42,
		DatasetType:   "random",
		ClusterCount:  10,
		ClusterRadius: 0.1,
		SkewFactor:    0.8,
	}
}

// generateBenchmarkData generates data for benchmarks based on configuration
func generateBenchmarkData(config BenchmarkConfig) ([][]float32, [][]float32) {
	rand.Seed(config.Seed)

	// Generate dataset
	var vectors [][]float32
	switch config.DatasetType {
	case "random":
		vectors = generateRandomVectors(config.NumVectors, config.Dimension)
	case "clustered":
		vectors = generateClusteredVectors(config.NumVectors, config.Dimension, config.ClusterCount, config.ClusterRadius)
	case "skewed":
		vectors = generateSkewedVectors(config.NumVectors, config.Dimension, config.SkewFactor)
	default:
		vectors = generateRandomVectors(config.NumVectors, config.Dimension)
	}

	// Generate query vectors (always random for fair comparison)
	queries := generateRandomVectors(config.QueryCount, config.Dimension)

	return vectors, queries
}

// generateClusteredVectors generates vectors clustered around centroids
func generateClusteredVectors(count, dimension, clusterCount int, radius float32) [][]float32 {
	vectors := make([][]float32, count)
	centroids := generateRandomVectors(clusterCount, dimension)

	for i := 0; i < count; i++ {
		// Select a random centroid
		centroidIdx := rand.Intn(clusterCount)
		centroid := centroids[centroidIdx]

		// Generate a vector within radius of the centroid
		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			// Random offset within radius
			offset := (rand.Float32()*2 - 1) * radius
			vector[j] = centroid[j] + offset
		}

		// Normalize
		normalizeVector(vector)
		vectors[i] = vector
	}

	return vectors
}

// generateSkewedVectors generates vectors with skewed distribution
func generateSkewedVectors(count, dimension int, skewFactor float32) [][]float32 {
	vectors := make([][]float32, count)

	for i := 0; i < count; i++ {
		vector := make([]float32, dimension)

		// Generate skewed components
		for j := 0; j < dimension; j++ {
			// Apply power law distribution
			value := rand.Float32()
			if rand.Float32() < skewFactor {
				// Skew towards smaller values
				value = value * value
			}
			// Convert to range [-1, 1]
			vector[j] = value*2 - 1
		}

		// Normalize
		normalizeVector(vector)
		vectors[i] = vector
	}

	return vectors
}

// normalizeVector normalizes a vector to unit length
func normalizeVector(vector []float32) {
	var sum float32
	for _, v := range vector {
		sum += v * v
	}
	norm := float32(1.0 / float64(sum))
	for i := range vector {
		vector[i] *= norm
	}
}

// BenchmarkHybridIndex benchmarks the hybrid index with different configurations
func BenchmarkHybridIndex(b *testing.B) {
	benchmarkConfigs := []BenchmarkConfig{
		// Small dataset
		{NumVectors: 1000, Dimension: 128, QueryCount: 100, K: 10, DatasetType: "random"},
		// Medium dataset
		{NumVectors: 10000, Dimension: 128, QueryCount: 100, K: 10, DatasetType: "random"},
		// Large dataset
		{NumVectors: 100000, Dimension: 128, QueryCount: 100, K: 10, DatasetType: "random"},
		// High-dimensional dataset
		{NumVectors: 10000, Dimension: 512, QueryCount: 100, K: 10, DatasetType: "random"},
		// Clustered dataset
		{NumVectors: 10000, Dimension: 128, QueryCount: 100, K: 10, DatasetType: "clustered"},
		// Skewed dataset
		{NumVectors: 10000, Dimension: 128, QueryCount: 100, K: 10, DatasetType: "skewed"},
	}

	for _, config := range benchmarkConfigs {
		// Generate dataset and queries
		vectors, queries := generateBenchmarkData(config)

		// Benchmark different index types
		benchmarkIndexTypes(b, vectors, queries, config)
	}
}

// benchmarkIndexTypes benchmarks different index types with the same data
func benchmarkIndexTypes(b *testing.B, vectors [][]float32, queries [][]float32, config BenchmarkConfig) {
	// Define index types to benchmark
	indexTypes := []IndexType{
		ExactIndexType,
		HNSWIndexType,
		LSHIndexType,
		HybridIndexType,
	}

	for _, indexType := range indexTypes {
		// Create benchmark name
		benchName := fmt.Sprintf("%s/%d/%d/%s", indexType.String(), config.NumVectors, config.Dimension, config.DatasetType)

		b.Run(benchName, func(b *testing.B) {
			// Create index configuration
			indexConfig := DefaultIndexConfig()
			indexConfig.Type = indexType
			indexConfig.Distance = hnsw.CosineDistance

			// Create index
			index, err := NewHybridIndex[int](indexConfig)
			if err != nil {
				b.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Add vectors to index
			for i, vector := range vectors {
				if err := index.Add(i, vector); err != nil {
					b.Fatalf("Failed to add vector: %v", err)
				}
			}

			// Reset timer before search benchmark
			b.ResetTimer()

			// Run benchmark
			for i := 0; i < b.N; i++ {
				queryIdx := i % len(queries)
				query := queries[queryIdx]

				_, err := index.Search(query, config.K)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
			}
		})
	}
}

// String returns a string representation of the index type
func (t IndexType) String() string {
	switch t {
	case ExactIndexType:
		return "Exact"
	case HNSWIndexType:
		return "HNSW"
	case LSHIndexType:
		return "LSH"
	case HybridIndexType:
		return "Hybrid"
	default:
		return "Unknown"
	}
}

// BenchmarkBuildTime benchmarks the time to build different index types
func BenchmarkBuildTime(b *testing.B) {
	config := DefaultBenchmarkConfig()
	vectors, _ := generateBenchmarkData(config)

	indexTypes := []IndexType{
		ExactIndexType,
		HNSWIndexType,
		LSHIndexType,
		HybridIndexType,
	}

	for _, indexType := range indexTypes {
		benchName := fmt.Sprintf("Build/%s/%d", indexType.String(), config.NumVectors)

		b.Run(benchName, func(b *testing.B) {
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Create index configuration
				indexConfig := DefaultIndexConfig()
				indexConfig.Type = indexType

				// Create index
				index, err := NewHybridIndex[int](indexConfig)
				if err != nil {
					b.Fatalf("Failed to create index: %v", err)
				}

				// Add vectors to index
				for j, vector := range vectors {
					if err := index.Add(j, vector); err != nil {
						b.Fatalf("Failed to add vector: %v", err)
					}
				}

				index.Close()
			}
		})
	}
}

// BenchmarkMemoryUsage benchmarks the memory usage of different index types
func BenchmarkMemoryUsage(b *testing.B) {
	// This is a placeholder for memory usage benchmarks
	// In Go, it's challenging to accurately measure memory usage within the benchmark itself
	// Consider using external tools like pprof for memory profiling

	b.Skip("Memory usage benchmarks require external profiling")
}

// BenchmarkRecall benchmarks the recall of different index types
func BenchmarkRecall(b *testing.B) {
	config := DefaultBenchmarkConfig()
	vectors, queries := generateBenchmarkData(config)

	// First, compute ground truth using exact search
	exactConfig := DefaultIndexConfig()
	exactConfig.Type = ExactIndexType
	exactIndex, err := NewHybridIndex[int](exactConfig)
	if err != nil {
		b.Fatalf("Failed to create exact index: %v", err)
	}
	defer exactIndex.Close()

	// Add vectors to exact index
	for i, vector := range vectors {
		if err := exactIndex.Add(i, vector); err != nil {
			b.Fatalf("Failed to add vector to exact index: %v", err)
		}
	}

	// Compute ground truth
	groundTruth := make([][]hnsw.Node[int], len(queries))
	for i, query := range queries {
		results, err := exactIndex.Search(query, config.K)
		if err != nil {
			b.Fatalf("Exact search failed: %v", err)
		}
		groundTruth[i] = results
	}

	// Benchmark recall for different index types
	indexTypes := []IndexType{
		HNSWIndexType,
		LSHIndexType,
		HybridIndexType,
	}

	for _, indexType := range indexTypes {
		benchName := fmt.Sprintf("Recall/%s", indexType.String())

		b.Run(benchName, func(b *testing.B) {
			// Create index
			indexConfig := DefaultIndexConfig()
			indexConfig.Type = indexType
			index, err := NewHybridIndex[int](indexConfig)
			if err != nil {
				b.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Add vectors to index
			for i, vector := range vectors {
				if err := index.Add(i, vector); err != nil {
					b.Fatalf("Failed to add vector: %v", err)
				}
			}

			b.ResetTimer()

			// Measure recall
			var totalRecall float64

			for i := 0; i < b.N; i++ {
				recall := 0.0

				for j, query := range queries {
					results, err := index.Search(query, config.K)
					if err != nil {
						b.Fatalf("Search failed: %v", err)
					}

					// Compute recall
					truth := groundTruth[j]
					truthKeys := make(map[int]struct{})
					for _, node := range truth {
						truthKeys[node.Key] = struct{}{}
					}

					matches := 0
					for _, node := range results {
						if _, found := truthKeys[node.Key]; found {
							matches++
						}
					}

					recall += float64(matches) / float64(len(truth))
				}

				recall /= float64(len(queries))
				totalRecall += recall
			}

			// Report average recall
			if b.N > 0 {
				b.ReportMetric(totalRecall/float64(b.N), "recall")
			}
		})
	}
}

// BenchmarkQueryLatency benchmarks the latency distribution of queries
func BenchmarkQueryLatency(b *testing.B) {
	config := DefaultBenchmarkConfig()
	vectors, queries := generateBenchmarkData(config)

	indexTypes := []IndexType{
		ExactIndexType,
		HNSWIndexType,
		LSHIndexType,
		HybridIndexType,
	}

	for _, indexType := range indexTypes {
		benchName := fmt.Sprintf("Latency/%s", indexType.String())

		b.Run(benchName, func(b *testing.B) {
			// Create index
			indexConfig := DefaultIndexConfig()
			indexConfig.Type = indexType
			index, err := NewHybridIndex[int](indexConfig)
			if err != nil {
				b.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Add vectors to index
			for i, vector := range vectors {
				if err := index.Add(i, vector); err != nil {
					b.Fatalf("Failed to add vector: %v", err)
				}
			}

			// Warm up
			for _, query := range queries[:5] {
				_, err := index.Search(query, config.K)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
			}

			b.ResetTimer()

			// Measure latency
			latencies := make([]time.Duration, 0, len(queries))

			for i := 0; i < b.N; i++ {
				for _, query := range queries {
					start := time.Now()
					_, err := index.Search(query, config.K)
					duration := time.Since(start)

					if err != nil {
						b.Fatalf("Search failed: %v", err)
					}

					latencies = append(latencies, duration)
				}
			}

			// Report latency statistics
			if len(latencies) > 0 {
				// Sort latencies
				sortDurations(latencies)

				// Calculate percentiles
				p50 := latencies[len(latencies)*50/100]
				p95 := latencies[len(latencies)*95/100]
				p99 := latencies[len(latencies)*99/100]

				b.ReportMetric(float64(p50.Nanoseconds()), "p50_ns")
				b.ReportMetric(float64(p95.Nanoseconds()), "p95_ns")
				b.ReportMetric(float64(p99.Nanoseconds()), "p99_ns")
			}
		})
	}
}

// sortDurations sorts a slice of durations in ascending order
func sortDurations(durations []time.Duration) {
	for i := 1; i < len(durations); i++ {
		j := i
		for j > 0 && durations[j-1] > durations[j] {
			durations[j-1], durations[j] = durations[j], durations[j-1]
			j--
		}
	}
}

// BenchmarkScalability benchmarks how performance scales with dataset size
func BenchmarkScalability(b *testing.B) {
	sizes := []int{1000, 10000, 100000}
	dimension := 128
	queryCount := 10
	k := 10

	for _, size := range sizes {
		config := BenchmarkConfig{
			NumVectors:  size,
			Dimension:   dimension,
			QueryCount:  queryCount,
			K:           k,
			DatasetType: "random",
		}

		vectors, queries := generateBenchmarkData(config)

		indexTypes := []IndexType{
			ExactIndexType,
			HNSWIndexType,
			LSHIndexType,
			HybridIndexType,
		}

		for _, indexType := range indexTypes {
			benchName := fmt.Sprintf("Scale/%s/%d", indexType.String(), size)

			b.Run(benchName, func(b *testing.B) {
				// Create index
				indexConfig := DefaultIndexConfig()
				indexConfig.Type = indexType
				index, err := NewHybridIndex[int](indexConfig)
				if err != nil {
					b.Fatalf("Failed to create index: %v", err)
				}
				defer index.Close()

				// Add vectors to index
				for i, vector := range vectors {
					if err := index.Add(i, vector); err != nil {
						b.Fatalf("Failed to add vector: %v", err)
					}
				}

				b.ResetTimer()

				// Run benchmark
				for i := 0; i < b.N; i++ {
					queryIdx := i % len(queries)
					query := queries[queryIdx]

					_, err := index.Search(query, k)
					if err != nil {
						b.Fatalf("Search failed: %v", err)
					}
				}
			})
		}
	}
}

// BenchmarkDimensionalityImpact benchmarks how performance scales with dimensionality
func BenchmarkDimensionalityImpact(b *testing.B) {
	dimensions := []int{32, 128, 512, 1024}
	size := 10000
	queryCount := 10
	k := 10

	for _, dimension := range dimensions {
		config := BenchmarkConfig{
			NumVectors:  size,
			Dimension:   dimension,
			QueryCount:  queryCount,
			K:           k,
			DatasetType: "random",
		}

		vectors, queries := generateBenchmarkData(config)

		indexTypes := []IndexType{
			ExactIndexType,
			HNSWIndexType,
			LSHIndexType,
			HybridIndexType,
		}

		for _, indexType := range indexTypes {
			benchName := fmt.Sprintf("Dimension/%s/%d", indexType.String(), dimension)

			b.Run(benchName, func(b *testing.B) {
				// Create index
				indexConfig := DefaultIndexConfig()
				indexConfig.Type = indexType
				index, err := NewHybridIndex[int](indexConfig)
				if err != nil {
					b.Fatalf("Failed to create index: %v", err)
				}
				defer index.Close()

				// Add vectors to index
				for i, vector := range vectors {
					if err := index.Add(i, vector); err != nil {
						b.Fatalf("Failed to add vector: %v", err)
					}
				}

				b.ResetTimer()

				// Run benchmark
				for i := 0; i < b.N; i++ {
					queryIdx := i % len(queries)
					query := queries[queryIdx]

					_, err := index.Search(query, k)
					if err != nil {
						b.Fatalf("Search failed: %v", err)
					}
				}
			})
		}
	}
}
