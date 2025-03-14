package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
)

const (
	dimensions    = 128
	numVectors    = 10000
	numQueries    = 100
	k             = 10
	numHashTables = 4
	numHashBits   = 8
)

func main() {
	fmt.Println("Adaptive Hybrid Index Example")

	// Initialize random number generator
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Create distance function
	distFunc := hnsw.CosineDistance

	// Create underlying indexes
	exactIndex := hybrid.NewExactIndex[int](distFunc)

	// Create HNSW config
	hnswConfig := hybrid.DefaultIndexConfig()
	hnswConfig.Type = hybrid.HNSWIndexType

	// Create LSH index
	lshIndex := hybrid.NewLSHIndex[int](numHashTables, numHashBits, distFunc)

	// Create adaptive config
	adaptiveConfig := hybrid.DefaultAdaptiveConfig()
	adaptiveConfig.InitialExactThreshold = 1000
	adaptiveConfig.InitialDimThreshold = 100
	adaptiveConfig.ExplorationFactor = 0.2

	// Create a new HNSW graph directly
	hnswGraph, err := hnsw.NewGraphWithConfig[int](
		int(hnswConfig.M),
		hnswConfig.Ml,
		hnswConfig.EfSearch,
		distFunc,
	)
	if err != nil {
		fmt.Printf("Error creating HNSW graph: %v\n", err)
		return
	}

	// Create adaptive hybrid index
	adaptiveIndex := hybrid.NewAdaptiveHybridIndex(
		exactIndex,
		hnswGraph,
		lshIndex,
		distFunc,
		adaptiveConfig,
	)

	// Generate random vectors
	fmt.Println("Generating random vectors...")
	vectors := generateRandomVectorsForAdaptive(numVectors, dimensions)

	// Add vectors to the index
	fmt.Println("Adding vectors to the index...")
	for i, vector := range vectors {
		if err := adaptiveIndex.Add(i, vector); err != nil {
			fmt.Printf("Error adding vector %d: %v\n", i, err)
		}
	}

	// Generate query vectors
	fmt.Println("Generating query vectors...")
	queryVectors := generateRandomVectorsForAdaptive(numQueries, dimensions)

	// Perform searches with different characteristics
	fmt.Println("Performing searches...")

	// First batch: standard queries
	fmt.Println("\nStandard queries:")
	for i := 0; i < 10; i++ {
		query := queryVectors[i]
		keys, _, err := adaptiveIndex.Search(query, k)
		if err != nil {
			fmt.Printf("Error searching: %v\n", err)
			continue
		}
		fmt.Printf("Query %d: Found %d results\n", i, len(keys))
	}

	// Print stats after first batch
	printAdaptiveStats(adaptiveIndex)

	// Second batch: high-dimensional queries (simulate by using only a subset of dimensions)
	fmt.Println("\nHigh-dimensional queries:")
	for i := 10; i < 20; i++ {
		query := generateRandomVectorForAdaptive(dimensions * 2)    // Higher dimensionality
		keys, _, err := adaptiveIndex.Search(query[:dimensions], k) // Use compatible part
		if err != nil {
			fmt.Printf("Error searching: %v\n", err)
			continue
		}
		fmt.Printf("Query %d: Found %d results\n", i, len(keys))
	}

	// Print stats after second batch
	printAdaptiveStats(adaptiveIndex)

	// Third batch: clustered queries (similar vectors)
	fmt.Println("\nClustered queries:")
	baseVector := generateRandomVectorForAdaptive(dimensions)
	for i := 20; i < 30; i++ {
		// Create a query that's a slight variation of the base vector
		query := make([]float32, dimensions)
		copy(query, baseVector)
		// Add small random variations
		for j := range query {
			query[j] += rng.Float32() * 0.1
		}

		keys, _, err := adaptiveIndex.Search(query, k)
		if err != nil {
			fmt.Printf("Error searching: %v\n", err)
			continue
		}
		fmt.Printf("Query %d: Found %d results\n", i, len(keys))
	}

	// Print final stats
	printAdaptiveStats(adaptiveIndex)

	// Reset stats and run a few more queries
	fmt.Println("\nResetting stats and running more queries...")
	adaptiveIndex.ResetStats()

	for i := 30; i < 40; i++ {
		query := queryVectors[i]
		keys, _, err := adaptiveIndex.Search(query, k)
		if err != nil {
			fmt.Printf("Error searching: %v\n", err)
			continue
		}
		fmt.Printf("Query %d: Found %d results\n", i, len(keys))
	}

	// Print stats after reset
	printAdaptiveStats(adaptiveIndex)
}

// generateRandomVectorForAdaptive creates a random unit vector
func generateRandomVectorForAdaptive(dimensions int) []float32 {
	// Use the global random source
	vector := make([]float32, dimensions)

	// Generate random components
	var sumSquares float32
	for i := range vector {
		vector[i] = rand.Float32()*2 - 1 // Random value between -1 and 1
		sumSquares += vector[i] * vector[i]
	}

	// Normalize to unit length
	magnitude := float32(1.0)
	if sumSquares > 0 {
		magnitude = float32(1.0 / float32(sumSquares))
	}

	for i := range vector {
		vector[i] *= magnitude
	}

	return vector
}

// generateRandomVectorsForAdaptive generates multiple random vectors
func generateRandomVectorsForAdaptive(count, dimensions int) [][]float32 {
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = generateRandomVectorForAdaptive(dimensions)
	}
	return vectors
}

// printAdaptiveStats prints the current statistics of the adaptive index
func printAdaptiveStats(index *hybrid.AdaptiveHybridIndex[int]) {
	stats := index.GetStats()

	fmt.Println("\nAdaptive Index Statistics:")
	fmt.Printf("Vector count: %d\n", stats["vector_count"])

	if exactCount, ok := stats["exact_index_count"]; ok {
		fmt.Printf("Exact index count: %d\n", exactCount)
	}

	if hnswCount, ok := stats["hnsw_index_count"]; ok {
		fmt.Printf("HNSW index count: %d\n", hnswCount)
	}

	if lshCount, ok := stats["lsh_index_count"]; ok {
		fmt.Printf("LSH index count: %d\n", lshCount)
	}

	if exactThreshold, ok := stats["exact_threshold"].(int); ok {
		fmt.Printf("Exact threshold: %d\n", exactThreshold)
	}

	if dimThreshold, ok := stats["dimension_threshold"].(int); ok {
		fmt.Printf("Dimension threshold: %d\n", dimThreshold)
	}

	if strategies, ok := stats["strategies"].(map[string]interface{}); ok {
		fmt.Println("\nStrategy Statistics:")
		for strategy, stratStats := range strategies {
			if ss, ok := stratStats.(map[string]interface{}); ok {
				fmt.Printf("  %s:\n", strategy)
				if totalQueries, ok := ss["total_queries"].(int); ok {
					fmt.Printf("    Total queries: %d\n", totalQueries)
				}
				if avgDuration, ok := ss["avg_duration"].(string); ok {
					fmt.Printf("    Avg duration: %s\n", avgDuration)
				}
				if p95Duration, ok := ss["p95_duration"].(string); ok {
					fmt.Printf("    P95 duration: %s\n", p95Duration)
				}
				if avgRecall, ok := ss["avg_recall"].(float64); ok {
					fmt.Printf("    Avg recall: %.4f\n", avgRecall)
				}
				if successRate, ok := ss["success_rate"].(float64); ok {
					fmt.Printf("    Success rate: %.4f\n", successRate)
				}
			}
		}
	}

	if clusters, ok := stats["query_clusters"].(map[string]int); ok && len(clusters) > 0 {
		fmt.Println("\nQuery Clusters:")
		for hash, count := range clusters {
			fmt.Printf("  Cluster %s: %d queries\n", hash, count)
		}
	}

	fmt.Println()
}
