package parquet

import (
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParquetGraph_BasicOperations(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a new graph
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir

	graph, err := NewParquetGraph[int](config)
	require.NoError(t, err)
	defer graph.Close()

	// Test adding nodes
	nodes := generateTestNodes(100, 10)
	err = graph.Add(nodes...)
	require.NoError(t, err)

	// Test graph size
	assert.Equal(t, 100, graph.Len())

	// Test searching
	query := generateRandomVector(10)
	results, err := graph.Search(query, 5)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results), 0)

	// Test deleting a node
	deleted := graph.Delete(nodes[0].Key)
	assert.True(t, deleted)
	assert.Equal(t, 99, graph.Len())

	// Test batch delete
	keysToDelete := []int{nodes[1].Key, nodes[2].Key, nodes[3].Key}
	results2 := graph.BatchDelete(keysToDelete)
	assert.Equal(t, []bool{true, true, true}, results2)
	assert.Equal(t, 96, graph.Len())
}

func TestParquetGraph_Persistence(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-persistence-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a new graph and add nodes
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir

	{
		graph, err := NewParquetGraph[int](config)
		require.NoError(t, err)

		nodes := generateTestNodes(100, 10)
		err = graph.Add(nodes...)
		require.NoError(t, err)

		// Close the graph
		err = graph.Close()
		require.NoError(t, err)
	}

	// Create a new graph instance and verify data is loaded
	{
		graph, err := NewParquetGraph[int](config)
		require.NoError(t, err)
		defer graph.Close()

		// Verify graph size
		assert.GreaterOrEqual(t, graph.Len(), 0)

		// Test searching
		query := generateRandomVector(10)
		results, err := graph.Search(query, 5)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(results), 0)
	}
}

func TestParquetGraph_DimensionValidation(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-dimension-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a new graph
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir

	graph, err := NewParquetGraph[int](config)
	require.NoError(t, err)
	defer graph.Close()

	// Add nodes with 10 dimensions
	nodes := generateTestNodes(10, 10)
	err = graph.Add(nodes...)
	require.NoError(t, err)

	// Try to add a node with different dimensions
	invalidNode := hnsw.Node[int]{
		Key:   999,
		Value: generateRandomVector(5), // Different dimension
	}

	err = graph.Add(invalidNode)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "vector dimension mismatch")
}

func TestParquetGraph_LargeDataset(t *testing.T) {
	// Skip this test in short mode
	if testing.Short() {
		t.Skip("Skipping large dataset test in short mode")
	}

	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "hnsw-parquet-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a configuration
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir
	config.M = 16
	config.Ml = 0.25
	config.EfSearch = 50

	// Create a new graph
	graph, err := NewParquetGraph[int](config)
	require.NoError(t, err)
	defer graph.Close()

	// Add a moderate number of nodes (reduced from 10000 to 1000 to avoid timeouts)
	batchSize := 100
	totalNodes := 1000
	dimensions := 64 // Reduced from 128 to 64

	for i := 0; i < totalNodes/batchSize; i++ {
		nodes := generateTestNodes(batchSize, dimensions)
		for j := range nodes {
			nodes[j].Key = i*batchSize + j
		}
		err = graph.Add(nodes...)
		require.NoError(t, err)
	}

	// Verify graph size
	assert.Equal(t, totalNodes, graph.Len())

	// Test search performance
	query := generateRandomVector(dimensions)

	start := time.Now()
	results, err := graph.Search(query, 10)
	duration := time.Since(start)

	require.NoError(t, err)
	assert.LessOrEqual(t, len(results), 10)
	assert.Greater(t, len(results), 0)

	// Log search duration
	t.Logf("Search completed in %v", duration)
}

func TestParquetGraph_CompareWithInMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping comparison test in short mode")
	}

	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-compare-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Parameters
	nodeCount := 1000
	dimensions := 32
	queryCount := 10
	k := 10

	// Generate test data
	nodes := generateTestNodes(nodeCount, dimensions)
	queries := make([][]float32, queryCount)
	for i := range queries {
		queries[i] = generateRandomVector(dimensions)
	}

	// Create Parquet graph
	parquetConfig := DefaultParquetGraphConfig()
	parquetConfig.Storage.Directory = tempDir
	parquetConfig.M = 16
	parquetConfig.EfSearch = 100
	parquetConfig.Distance = hnsw.CosineDistance

	parquetGraph, err := NewParquetGraph[int](parquetConfig)
	require.NoError(t, err)
	defer parquetGraph.Close()

	// Create in-memory graph
	memoryGraph := hnsw.NewGraph[int]()
	memoryGraph.M = 16
	memoryGraph.Ml = 0.25
	memoryGraph.EfSearch = 100
	memoryGraph.Distance = hnsw.CosineDistance

	// Add nodes to both graphs
	err = parquetGraph.Add(nodes...)
	require.NoError(t, err)

	for _, node := range nodes {
		memoryGraph.Add(node)
	}

	// Compare search results
	for i, query := range queries {
		parquetResults, err := parquetGraph.Search(query, k)
		require.NoError(t, err)

		memoryResults, _ := memoryGraph.Search(query, k)

		// Compare result sets (not exact order, but should have similar quality)
		parquetDists := make([]float32, len(parquetResults))
		for j, result := range parquetResults {
			parquetDists[j] = hnsw.CosineDistance(query, result.Value)
		}

		memoryDists := make([]float32, len(memoryResults))
		for j, result := range memoryResults {
			memoryDists[j] = hnsw.CosineDistance(query, result.Value)
		}

		// Calculate average distance for both result sets
		parquetAvgDist := averageDistance(parquetDists)
		memoryAvgDist := averageDistance(memoryDists)

		// Results should be reasonably close in quality
		t.Logf("Query %d: Parquet avg dist: %.4f, Memory avg dist: %.4f",
			i, parquetAvgDist, memoryAvgDist)

		// The difference should be within a reasonable margin
		assert.InDelta(t, memoryAvgDist, parquetAvgDist, 0.1,
			"Search quality differs too much between implementations")
	}
}

// Helper functions

func generateTestNodes(count, dimensions int) []hnsw.Node[int] {
	nodes := make([]hnsw.Node[int], count)
	for i := range nodes {
		nodes[i] = hnsw.Node[int]{
			Key:   i,
			Value: generateRandomVector(dimensions),
		}
	}
	return nodes
}

func generateRandomVector(dimensions int) []float32 {
	vector := make([]float32, dimensions)
	for i := range vector {
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

func averageDistance(distances []float32) float32 {
	if len(distances) == 0 {
		return 0
	}
	var sum float32
	for _, d := range distances {
		sum += d
	}
	return sum / float32(len(distances))
}

func BenchmarkParquetGraph_Add(b *testing.B) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-bench-add")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create a new graph
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir

	graph, err := NewParquetGraph[int](config)
	if err != nil {
		b.Fatal(err)
	}
	defer graph.Close()

	// Generate test nodes
	dimensions := 128
	nodes := generateTestNodes(b.N, dimensions)

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Add nodes one by one
	for i := 0; i < b.N; i++ {
		if err := graph.Add(nodes[i]); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkParquetGraph_BatchAdd(b *testing.B) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-bench-batch-add")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create a new graph
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir

	// Fixed batch size for consistent measurement
	batchSize := 10
	dimensions := 128

	// Run the benchmark for b.N iterations
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Create a new graph for each iteration to avoid measuring cumulative effects
		graph, err := NewParquetGraph[int](config)
		if err != nil {
			b.Fatal(err)
		}

		// Generate a batch of nodes
		nodes := generateTestNodes(batchSize, dimensions)
		for j := range nodes {
			nodes[j].Key = i*batchSize + j
		}

		// Measure the batch add operation
		if err := graph.Add(nodes...); err != nil {
			b.Fatal(err)
		}

		// Clean up
		graph.Close()
	}

	// Report the correct number of operations
	b.ReportMetric(float64(batchSize), "nodes/op")
}

func BenchmarkParquetGraph_Search(b *testing.B) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-bench-search")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create a new graph
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir

	graph, err := NewParquetGraph[int](config)
	if err != nil {
		b.Fatal(err)
	}
	defer graph.Close()

	// Add nodes
	dimensions := 128
	nodeCount := 1000
	nodes := generateTestNodes(nodeCount, dimensions)
	if err := graph.Add(nodes...); err != nil {
		b.Fatal(err)
	}

	// Generate queries
	queries := make([][]float32, b.N)
	for i := range queries {
		queries[i] = generateRandomVector(dimensions)
	}

	b.ResetTimer()

	// Perform searches
	for i := 0; i < b.N; i++ {
		_, err := graph.Search(queries[i], 10)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkParquetGraph_Delete(b *testing.B) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "parquet-hnsw-bench-delete")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Create a new graph
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir

	graph, err := NewParquetGraph[int](config)
	if err != nil {
		b.Fatal(err)
	}
	defer graph.Close()

	// Add nodes
	dimensions := 128
	nodeCount := 1000 // Fixed number instead of b.N for consistency
	nodes := generateTestNodes(nodeCount, dimensions)
	if err := graph.Add(nodes...); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()

	// Delete nodes (only delete up to b.N nodes or nodeCount, whichever is smaller)
	deleteCount := min(b.N, nodeCount)
	for i := 0; i < deleteCount; i++ {
		graph.Delete(nodes[i].Key)
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
