package main

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/TFMV/hnsw/hnsw-extensions/arrow"
)

const (
	benchDimensions = 128
	benchNumVectors = 100000
	benchK          = 10
)

var (
	benchVectors [][]float32
	benchQueries [][]float32
)

func init() {
	// Generate random vectors for benchmarks
	benchVectors = make([][]float32, benchNumVectors)
	for i := 0; i < benchNumVectors; i++ {
		benchVectors[i] = generateRandomVector(benchDimensions)
	}

	// Generate random queries for benchmarks
	benchQueries = make([][]float32, 100)
	for i := 0; i < 100; i++ {
		benchQueries[i] = generateRandomVector(benchDimensions)
	}
}

// setupArrowIndex creates and populates an Arrow index for benchmarking
func setupArrowIndex(b *testing.B) *arrow.ArrowIndex[int] {
	b.Helper()

	// Create a temporary directory for the index
	tempDir := fmt.Sprintf("arrow_benchmark_%d", rand.Int())
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		b.Fatalf("Error creating directory: %v", err)
	}

	// Create index configuration
	config := arrow.DefaultArrowGraphConfig()
	config.Storage.Directory = tempDir
	config.Storage.NumWorkers = 4

	// Create index
	index, err := arrow.NewArrowIndex[int](config)
	if err != nil {
		b.Fatalf("Error creating index: %v", err)
	}

	return index
}

// cleanupArrowIndex closes the index and removes the temporary directory
func cleanupArrowIndex(b *testing.B, index *arrow.ArrowIndex[int]) {
	b.Helper()

	// Get the directory
	dir := index.GetStorageConfig().Directory

	// Close the index
	if err := index.Close(); err != nil {
		b.Fatalf("Error closing index: %v", err)
	}

	// Remove the directory
	if err := os.RemoveAll(dir); err != nil {
		b.Fatalf("Error removing directory: %v", err)
	}
}

// BenchmarkArrowAdd benchmarks adding vectors to the index
func BenchmarkArrowAdd(b *testing.B) {
	// Create index
	index := setupArrowIndex(b)
	defer cleanupArrowIndex(b, index)

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		// Use modulo to cycle through the vectors
		vectorIdx := i % benchNumVectors
		err := index.Add(vectorIdx, benchVectors[vectorIdx])
		if err != nil {
			b.Fatalf("Error adding vector: %v", err)
		}
	}
}

// BenchmarkArrowBatchAdd benchmarks adding vectors in batches
func BenchmarkArrowBatchAdd(b *testing.B) {
	// Create index
	index := setupArrowIndex(b)
	defer cleanupArrowIndex(b, index)

	// Batch size
	batchSize := 100 // Reduced from 1000 to make the benchmark more stable

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		b.StopTimer() // Stop timer while preparing the batch

		// Create batch
		keys := make([]int, batchSize)
		vectors := make([][]float32, batchSize)

		for j := 0; j < batchSize; j++ {
			vectorIdx := (i*batchSize + j) % benchNumVectors
			keys[j] = vectorIdx
			vectors[j] = benchVectors[vectorIdx]
		}

		b.StartTimer() // Resume timer for the actual operation

		// Add batch
		errors := index.BatchAdd(keys, vectors)

		// Check for errors outside the timed section
		b.StopTimer()
		for j, err := range errors {
			if err != nil {
				b.Fatalf("Error adding vector %d: %v", keys[j], err)
			}
		}
		b.StartTimer()
	}
}

// BenchmarkArrowSearch benchmarks searching the index
func BenchmarkArrowSearch(b *testing.B) {
	// Create index
	index := setupArrowIndex(b)
	defer cleanupArrowIndex(b, index)

	// Add vectors to the index
	numVectorsToAdd := 10000
	for i := 0; i < numVectorsToAdd; i++ {
		err := index.Add(i, benchVectors[i])
		if err != nil {
			b.Fatalf("Error adding vector: %v", err)
		}
	}

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		// Use modulo to cycle through the queries
		queryIdx := i % len(benchQueries)
		_, err := index.Search(benchQueries[queryIdx], benchK)
		if err != nil {
			b.Fatalf("Error searching: %v", err)
		}
	}
}

// BenchmarkArrowBatchSearch benchmarks batch searching the index
func BenchmarkArrowBatchSearch(b *testing.B) {
	// Create index
	index := setupArrowIndex(b)
	defer cleanupArrowIndex(b, index)

	// Add vectors to the index
	numVectorsToAdd := 10000
	for i := 0; i < numVectorsToAdd; i++ {
		err := index.Add(i, benchVectors[i])
		if err != nil {
			b.Fatalf("Error adding vector: %v", err)
		}
	}

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		_, err := index.BatchSearch(benchQueries, benchK)
		if err != nil {
			b.Fatalf("Error batch searching: %v", err)
		}
	}
}

// BenchmarkArrowSave benchmarks saving the index
func BenchmarkArrowSave(b *testing.B) {
	// Create index
	index := setupArrowIndex(b)
	defer cleanupArrowIndex(b, index)

	// Add vectors to the index
	numVectorsToAdd := 10000
	for i := 0; i < numVectorsToAdd; i++ {
		err := index.Add(i, benchVectors[i])
		if err != nil {
			b.Fatalf("Error adding vector: %v", err)
		}
	}

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		err := index.Save()
		if err != nil {
			b.Fatalf("Error saving index: %v", err)
		}
	}
}

// BenchmarkArrowLoad benchmarks loading the index
func BenchmarkArrowLoad(b *testing.B) {
	// Create index
	index := setupArrowIndex(b)

	// Add vectors to the index
	numVectorsToAdd := 1000 // Reduced from 10000 to make the benchmark faster
	for i := 0; i < numVectorsToAdd; i++ {
		err := index.Add(i, benchVectors[i])
		if err != nil {
			b.Fatalf("Error adding vector: %v", err)
		}
	}

	// Save the index
	if err := index.Save(); err != nil {
		b.Fatalf("Error saving index: %v", err)
	}

	// Get the directory
	dir := index.GetStorageConfig().Directory

	// Close the index
	if err := index.Close(); err != nil {
		b.Fatalf("Error closing index: %v", err)
	}

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		// Create a new index with the same configuration
		config := arrow.DefaultArrowGraphConfig()
		config.Storage.Directory = dir

		// Create index
		newIndex, err := arrow.NewArrowIndex[int](config)
		if err != nil {
			b.Fatalf("Error creating index: %v", err)
		}

		// Load the index
		err = newIndex.Load()
		if err != nil {
			b.Fatalf("Error loading index: %v", err)
		}

		// Close the index
		if err := newIndex.Close(); err != nil {
			b.Fatalf("Error closing index: %v", err)
		}
	}

	// Clean up
	if err := os.RemoveAll(dir); err != nil {
		b.Fatalf("Error removing directory: %v", err)
	}
}
