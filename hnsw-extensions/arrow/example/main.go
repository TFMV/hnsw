package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/arrow"
)

const (
	dimensions = 128
	numVectors = 10000
	numQueries = 10
	k          = 10
)

func main() {
	// Create a new index with default configuration
	config := arrow.DefaultArrowGraphConfig()

	// Customize configuration
	config.M = 16
	config.EfSearch = 100
	config.Distance = hnsw.CosineDistance
	config.StorageDir = "arrow_example_data"
	config.NumWorkers = 4

	// Create directory if it doesn't exist
	if err := os.MkdirAll(config.StorageDir, 0755); err != nil {
		fmt.Printf("Error creating directory: %v\n", err)
		return
	}

	// Create index
	index, err := arrow.NewArrowIndex[string](config)
	if err != nil {
		fmt.Printf("Error creating index: %v\n", err)
		return
	}

	// Use explicit close instead of defer to ensure it runs before program exit
	// We'll call this at the end of the program

	// Generate random vectors
	fmt.Println("Generating random vectors...")
	keys := make([]string, numVectors)
	vectors := make([][]float32, numVectors)

	for i := 0; i < numVectors; i++ {
		keys[i] = fmt.Sprintf("vector-%d", i)
		vectors[i] = generateRandomVector(dimensions)
	}

	// Add vectors in batches
	batchSize := 1000
	fmt.Println("Adding vectors to index...")
	start := time.Now()

	for i := 0; i < numVectors; i += batchSize {
		end := i + batchSize
		if end > numVectors {
			end = numVectors
		}

		batchKeys := keys[i:end]
		batchVectors := vectors[i:end]

		errors := index.BatchAdd(batchKeys, batchVectors)
		for j, err := range errors {
			if err != nil {
				fmt.Printf("Error adding vector %s: %v\n", batchKeys[j], err)
			}
		}

		fmt.Printf("Added %d/%d vectors\n", end, numVectors)
	}

	addDuration := time.Since(start)
	fmt.Printf("Added %d vectors in %v (%.2f vectors/sec)\n",
		numVectors, addDuration, float64(numVectors)/addDuration.Seconds())

	// Save the index
	fmt.Println("Saving index...")
	if err := index.Save(); err != nil {
		fmt.Printf("Error saving index: %v\n", err)
		// Close the index before returning
		closeErr := index.Close()
		if closeErr != nil {
			fmt.Printf("Error closing index: %v\n", closeErr)
		}
		return
	}

	// Generate random queries
	fmt.Println("Generating random queries...")
	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = generateRandomVector(dimensions)
	}

	// Search
	fmt.Println("Searching...")
	start = time.Now()

	for i, query := range queries {
		results, err := index.Search(query, k)
		if err != nil {
			fmt.Printf("Error searching: %v\n", err)
			continue
		}

		fmt.Printf("Query %d results:\n", i+1)
		for j, result := range results {
			// Calculate distance between query and result vector
			distance := index.GetDistance()(query, result.Value)
			fmt.Printf("  %d. Key: %s, Distance: %.4f\n", j+1, result.Key, distance)
		}
		fmt.Println()
	}

	searchDuration := time.Since(start)
	fmt.Printf("Performed %d searches in %v (%.2f searches/sec)\n",
		numQueries, searchDuration, float64(numQueries)/searchDuration.Seconds())

	// Get statistics
	stats := index.Stats()
	fmt.Println("Index statistics:")
	fmt.Printf("  Number of vectors: %d\n", stats["num_vectors"])
	fmt.Printf("  Dimensions: %d\n", stats["dimensions"])
	fmt.Printf("  Number of layers: %d\n", stats["num_layers"])
	fmt.Printf("  Average connections per node: %.2f\n", stats["avg_connections_per_node"])

	// Storage statistics
	fmt.Println("Storage statistics:")
	for key, value := range stats {
		if len(key) >= 8 && key[:8] == "storage_" {
			fmt.Printf("  %s: %v\n", key[8:], value)
		}
	}

	// Explicitly close the index to ensure all resources are released
	fmt.Println("Closing index...")
	if err := index.Close(); err != nil {
		fmt.Printf("Error closing index: %v\n", err)
	}

	fmt.Println("Example completed successfully")
}

// generateRandomVector generates a random vector of the given dimension
func generateRandomVector(dim int) []float32 {
	vector := make([]float32, dim)

	// Generate random values
	for i := 0; i < dim; i++ {
		vector[i] = rand.Float32()
	}

	// Normalize the vector
	var sum float32
	for _, v := range vector {
		sum += v * v
	}

	norm := float32(1.0 / float64(sum))
	for i := range vector {
		vector[i] *= norm
	}

	return vector
}
