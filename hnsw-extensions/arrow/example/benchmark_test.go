package main

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	arrowext "github.com/TFMV/hnsw/hnsw-extensions/arrow"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
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
func setupArrowIndex(b *testing.B) *arrowext.ArrowIndex[int] {
	b.Helper()

	// Create a temporary directory for the index
	tempDir := fmt.Sprintf("arrow_benchmark_%d", rand.Int())
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		b.Fatalf("Error creating directory: %v", err)
	}

	// Create index configuration
	config := arrowext.DefaultArrowGraphConfig()
	config.Storage.Directory = tempDir
	config.Storage.NumWorkers = 4

	// Create index
	index, err := arrowext.NewArrowIndex[int](config)
	if err != nil {
		b.Fatalf("Error creating index: %v", err)
	}

	// Initialize the index by adding a vector with a special key that won't conflict with benchmarks
	// Use a negative key that won't be used in the benchmarks
	specialKey := -1
	if err := index.Add(specialKey, generateRandomVector(benchDimensions)); err != nil {
		b.Fatalf("Error initializing index: %v", err)
	}

	if err := index.Save(); err != nil {
		b.Fatalf("Error saving index: %v", err)
	}

	return index
}

// cleanupArrowIndex closes the index and removes the temporary directory
func cleanupArrowIndex(b *testing.B, index *arrowext.ArrowIndex[int]) {
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
		config := arrowext.DefaultArrowGraphConfig()
		config.Storage.Directory = dir

		// Create index
		newIndex, err := arrowext.NewArrowIndex[int](config)
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

// createArrowRecordBatch creates an Arrow record batch with vectors for benchmarking
func createArrowRecordBatch(batchSize int, startKey int) arrow.Record {
	// Create memory allocator
	pool := memory.NewGoAllocator()

	// Create schema
	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "key", Type: arrow.PrimitiveTypes.Int64},
			{Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)

	// Create record builder
	builder := array.NewRecordBuilder(pool, schema)
	defer builder.Release()

	// Get field builders
	keyBuilder := builder.Field(0).(*array.Int64Builder)
	vectorBuilder := builder.Field(1).(*array.ListBuilder)
	valueBuilder := vectorBuilder.ValueBuilder().(*array.Float32Builder)

	// Add data
	for i := 0; i < batchSize; i++ {
		// Add key with offset to avoid conflicts
		keyBuilder.Append(int64(startKey + i))

		// Add vector
		vectorBuilder.Append(true)
		for j := 0; j < benchDimensions; j++ {
			vectorIdx := (startKey + i) % benchNumVectors
			valueBuilder.Append(benchVectors[vectorIdx][j])
		}
	}

	// Build record
	record := builder.NewRecord()
	return record
}

// BenchmarkArrowAppenderSingle benchmarks appending a single record to the index
func BenchmarkArrowAppenderSingle(b *testing.B) {
	// Create index
	index := setupArrowIndex(b)
	defer cleanupArrowIndex(b, index)

	// Create appender
	appenderConfig := arrowext.DefaultAppenderConfig()
	appender := arrowext.NewArrowAppender[int](index, appenderConfig)

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		// Create a record batch with a single vector and unique key
		record := createArrowRecordBatch(1, i)

		if err := appender.AppendRecord(record); err != nil {
			b.Fatalf("Error appending record: %v", err)
		}

		// Release the record to free memory
		record.Release()
	}
}

// BenchmarkArrowAppenderBatch benchmarks appending a batch of records to the index
func BenchmarkArrowAppenderBatch(b *testing.B) {
	// Create index
	index := setupArrowIndex(b)
	defer cleanupArrowIndex(b, index)

	// Create appender
	appenderConfig := arrowext.DefaultAppenderConfig()
	appender := arrowext.NewArrowAppender[int](index, appenderConfig)

	// Batch size
	batchSize := 100

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		// Create a record batch with multiple vectors and unique keys
		startKey := i * batchSize
		record := createArrowRecordBatch(batchSize, startKey)

		if err := appender.AppendBatch(record); err != nil {
			b.Fatalf("Error appending batch: %v", err)
		}

		// Release the record to free memory
		record.Release()
	}
}

// BenchmarkArrowAppenderStream benchmarks streaming records to the index
func BenchmarkArrowAppenderStream(b *testing.B) {
	// Create index with explicit directory creation
	tempDir := fmt.Sprintf("arrow_benchmark_%d", rand.Int())
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		b.Fatalf("Error creating directory: %v", err)
	}

	// Create index configuration
	config := arrowext.DefaultArrowGraphConfig()
	config.Storage.Directory = tempDir
	config.Storage.NumWorkers = 4

	// Create index
	index, err := arrowext.NewArrowIndex[int](config)
	if err != nil {
		b.Fatalf("Error creating index: %v", err)
	}
	defer func() {
		// Close the index
		if err := index.Close(); err != nil {
			b.Fatalf("Error closing index: %v", err)
		}

		// Remove the directory
		if err := os.RemoveAll(tempDir); err != nil {
			b.Fatalf("Error removing directory: %v", err)
		}
	}()

	// Initialize the storage by adding a vector with a special key
	// Use a negative key that won't be used in the benchmarks
	specialKey := -1
	if err := index.Add(specialKey, generateRandomVector(benchDimensions)); err != nil {
		b.Fatalf("Error initializing index: %v", err)
	}

	// Explicitly save to create all necessary files and directories
	if err := index.Save(); err != nil {
		b.Fatalf("Error saving index: %v", err)
	}

	// Create appender
	appenderConfig := arrowext.DefaultAppenderConfig()
	appender := arrowext.NewArrowAppender[int](index, appenderConfig)

	// Batch size and number of batches
	batchSize := 100
	numBatches := 10

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run the benchmark
	for i := 0; i < b.N; i++ {
		b.StopTimer()

		// Create a channel for streaming records
		recordChan := make(chan arrow.Record, numBatches)

		// Create and send record batches with unique keys
		for j := 0; j < numBatches; j++ {
			// Use positive keys starting from 0
			startKey := i*batchSize*numBatches + j*batchSize
			record := createArrowRecordBatch(batchSize, startKey)
			recordChan <- record
		}
		close(recordChan)

		b.StartTimer()

		// Stream records
		if err := appender.StreamRecords(recordChan); err != nil {
			b.Fatalf("Error streaming records: %v", err)
		}

		// Explicitly flush after streaming to ensure changes are written
		if err := index.Save(); err != nil {
			b.Fatalf("Error saving index after streaming: %v", err)
		}
	}
}
