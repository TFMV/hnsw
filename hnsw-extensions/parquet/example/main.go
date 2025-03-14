package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/TFMV/hnsw"
	parquet "github.com/TFMV/hnsw/hnsw-extensions/parquet"
)

func main() {
	// Create a temporary directory for the example
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-example")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(tempDir)

	fmt.Printf("Using temporary directory: %s\n", tempDir)

	// Create a new Parquet-based HNSW graph with default configuration
	config := parquet.DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir
	config.M = 16        // Maximum number of connections per node
	config.Ml = 0.25     // Level generation factor
	config.EfSearch = 50 // Size of dynamic candidate list during search
	config.Distance = hnsw.CosineDistance

	// Configure incremental updates
	config.Incremental.MaxChanges = 100         // Trigger compaction after 100 changes
	config.Incremental.MaxAge = 5 * time.Minute // Trigger compaction after 5 minutes

	fmt.Println("Creating Parquet-based HNSW graph with incremental updates...")
	graph, err := parquet.NewParquetGraph[int](config)
	if err != nil {
		panic(err)
	}
	defer graph.Close()

	// Generate random vectors
	fmt.Println("Generating random vectors...")
	dimensions := 128
	numVectors := 1000

	// Add vectors in batches
	batchSize := 100
	for i := 0; i < numVectors/batchSize; i++ {
		fmt.Printf("Adding batch %d/%d...\n", i+1, numVectors/batchSize)

		nodes := make([]hnsw.Node[int], batchSize)
		for j := 0; j < batchSize; j++ {
			nodeId := i*batchSize + j
			nodes[j] = hnsw.Node[int]{
				Key:   nodeId,
				Value: generateRandomVector(dimensions),
			}
		}

		if err := graph.Add(nodes...); err != nil {
			panic(err)
		}
	}

	fmt.Printf("Added %d vectors to the graph\n", graph.Len())

	// Perform a search
	fmt.Println("Performing search...")
	query := generateRandomVector(dimensions)

	start := time.Now()
	results, err := graph.Search(query, 10)
	duration := time.Since(start)

	if err != nil {
		panic(err)
	}

	fmt.Printf("Search completed in %v\n", duration)
	fmt.Println("Search results:")
	for i, result := range results {
		distance := hnsw.CosineDistance(query, result.Value)
		fmt.Printf("  %d. ID: %d, Distance: %.6f\n", i+1, result.Key, distance)
	}

	// Delete some vectors
	fmt.Println("Deleting vectors (using incremental updates)...")
	keysToDelete := []int{50, 150, 250, 350, 450}
	results2 := graph.BatchDelete(keysToDelete)

	fmt.Println("Delete results:")
	for i, key := range keysToDelete {
		fmt.Printf("  Deleted %d: %v\n", key, results2[i])
	}

	fmt.Printf("Graph size after deletion: %d\n", graph.Len())

	// Update some vectors
	fmt.Println("Updating vectors (using incremental updates)...")
	for i := 0; i < 5; i++ {
		key := i*100 + 10
		err = graph.Add(hnsw.Node[int]{
			Key:   key,
			Value: generateRandomVector(dimensions),
		})
		if err != nil {
			panic(err)
		}
		fmt.Printf("  Updated vector with ID: %d\n", key)
	}

	// Demonstrate persistence
	fmt.Println("Demonstrating persistence with incremental updates...")
	fmt.Println("Closing graph (this will flush any pending changes)...")
	if err := graph.Close(); err != nil {
		panic(err)
	}

	fmt.Println("Reopening graph from Parquet files...")
	graph2, err := parquet.NewParquetGraph[int](config)
	if err != nil {
		panic(err)
	}
	defer graph2.Close()

	fmt.Printf("Reopened graph size: %d\n", graph2.Len())

	// Perform another search on the reopened graph
	fmt.Println("Performing search on reopened graph...")

	start = time.Now()
	results, err = graph2.Search(query, 10)
	duration = time.Since(start)

	if err != nil {
		panic(err)
	}

	fmt.Printf("Search completed in %v\n", duration)
	fmt.Println("Search results:")
	for i, result := range results {
		distance := hnsw.CosineDistance(query, result.Value)
		fmt.Printf("  %d. ID: %d, Distance: %.6f\n", i+1, result.Key, distance)
	}

	fmt.Println("Example completed successfully!")
}

// generateRandomVector creates a random normalized vector
func generateRandomVector(dimensions int) []float32 {
	// Initialize random number generator with a seed for reproducibility
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	vector := make([]float32, dimensions)
	for i := range vector {
		vector[i] = r.Float32()
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
