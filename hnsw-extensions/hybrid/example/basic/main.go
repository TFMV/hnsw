package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/hybrid"
)

func main() {
	// Initialize random number generator
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Create a hybrid index with default configuration
	config := hybrid.DefaultIndexConfig()
	config.Type = hybrid.HybridIndexType
	index, err := hybrid.NewHybridIndex[int](config)
	if err != nil {
		fmt.Printf("Error creating hybrid index: %v\n", err)
		return
	}
	defer index.Close()

	// Generate some random vectors
	numVectors := 1000
	dimension := 128
	vectors := generateRandomVectors(rng, numVectors, dimension)

	// Add vectors to the index
	fmt.Println("Adding vectors to the index...")
	for i := 0; i < numVectors; i++ {
		if err := index.Add(i, vectors[i]); err != nil {
			fmt.Printf("Error adding vector %d: %v\n", i, err)
			return
		}
	}

	// Generate a random query vector
	queryVector := generateRandomVector(rng, dimension)

	// Search for nearest neighbors
	fmt.Println("Searching for nearest neighbors...")
	k := 10
	results, err := index.Search(queryVector, k)
	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
		return
	}

	// Print results
	fmt.Printf("Found %d nearest neighbors:\n", len(results))
	for i, node := range results {
		// Calculate distance using cosine distance
		distance := hnsw.CosineDistance(queryVector, node.Value)
		fmt.Printf("%d. Key: %d, Distance: %.4f\n", i+1, node.Key, distance)
	}

	// Get index statistics
	stats := index.GetStats()
	fmt.Println("\nIndex Statistics:")
	fmt.Printf("Total vectors: %d\n", stats.TotalVectors)
	fmt.Printf("HNSW vectors: %d\n", stats.HNSWVectors)
	fmt.Printf("Exact vectors: %d\n", stats.ExactVectors)
	fmt.Printf("Number of partitions: %d\n", stats.NumPartitions)
	fmt.Printf("Average partition size: %.2f\n", stats.AvgPartitionSize)

	// Get partition statistics
	partitionStats := index.GetPartitionStats()
	if partitionStats != nil {
		fmt.Println("\nPartition Sizes:")
		for i, size := range partitionStats {
			fmt.Printf("Partition %d: %d vectors\n", i, size)
		}
	}

	// Demonstrate deletion
	fmt.Println("\nDeleting some vectors...")
	for i := 0; i < 10; i++ {
		keyToDelete := rand.Intn(numVectors)
		success := index.Delete(keyToDelete)
		fmt.Printf("Deleted key %d: %v\n", keyToDelete, success)
	}

	// Search again after deletion
	fmt.Println("\nSearching again after deletion...")
	results, err = index.Search(queryVector, k)
	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
		return
	}

	fmt.Printf("Found %d nearest neighbors:\n", len(results))
	for i, node := range results {
		// Calculate distance using cosine distance
		distance := hnsw.CosineDistance(queryVector, node.Value)
		fmt.Printf("%d. Key: %d, Distance: %.4f\n", i+1, node.Key, distance)
	}
}

// generateRandomVectors generates a set of random vectors.
func generateRandomVectors(rng *rand.Rand, count, dimension int) [][]float32 {
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = generateRandomVector(rng, dimension)
	}
	return vectors
}

// generateRandomVector generates a random unit vector.
func generateRandomVector(rng *rand.Rand, dimension int) []float32 {
	vector := make([]float32, dimension)
	var sum float32

	// Generate random components
	for i := range vector {
		vector[i] = rng.Float32()*2 - 1
		sum += vector[i] * vector[i]
	}

	// Normalize to unit length
	norm := float32(1.0 / float64(sum))
	for i := range vector {
		vector[i] *= norm
	}

	return vector
}
