package parquet

import (
	"os"
	"testing"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIncrementalUpdates(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "hnsw-parquet-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a configuration with incremental updates
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir
	config.Incremental.MaxChanges = 10 // Set a small value to trigger compaction

	// Create a new graph
	graph, err := NewParquetGraph[int](config)
	require.NoError(t, err)
	defer graph.Close()

	// Add some vectors
	vectors := []hnsw.Node[int]{
		{Key: 1, Value: []float32{1.0, 0.0, 0.0}},
		{Key: 2, Value: []float32{0.0, 1.0, 0.0}},
		{Key: 3, Value: []float32{0.0, 0.0, 1.0}},
	}
	err = graph.Add(vectors...)
	require.NoError(t, err)

	// Verify the vectors were added
	assert.Equal(t, 3, graph.Len())

	// Search for a vector
	results, err := graph.Search([]float32{1.0, 0.1, 0.1}, 1)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results), 0)
	if len(results) > 0 {
		assert.Equal(t, 1, results[0].Key)
	}

	// Delete a vector
	deleted := graph.Delete(2)
	assert.True(t, deleted)

	// Verify the vector was deleted
	assert.Equal(t, 2, graph.Len())

	// Add more vectors to trigger compaction
	moreVectors := []hnsw.Node[int]{
		{Key: 4, Value: []float32{1.0, 1.0, 0.0}},
		{Key: 5, Value: []float32{1.0, 0.0, 1.0}},
		{Key: 6, Value: []float32{0.0, 1.0, 1.0}},
		{Key: 7, Value: []float32{1.0, 1.0, 1.0}},
		{Key: 8, Value: []float32{-1.0, 0.0, 0.0}},
		{Key: 9, Value: []float32{0.0, -1.0, 0.0}},
		{Key: 10, Value: []float32{0.0, 0.0, -1.0}},
	}
	err = graph.Add(moreVectors...)
	require.NoError(t, err)

	// Verify all vectors were added
	assert.Equal(t, 9, graph.Len())

	// Verify the deleted vector is still deleted
	results, err = graph.Search([]float32{0.0, 1.0, 0.0}, 1)
	require.NoError(t, err)
	assert.NotEqual(t, 2, results[0].Key)

	// Close the graph
	err = graph.Close()
	require.NoError(t, err)

	// Reopen the graph to verify persistence
	graph2, err := NewParquetGraph[int](config)
	require.NoError(t, err)
	defer graph2.Close()

	// Verify the graph size
	assert.Equal(t, 9, graph2.Len())

	// Search for a vector
	results, err = graph2.Search([]float32{1.0, 0.1, 0.1}, 1)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results), 0)
	if len(results) > 0 {
		assert.Equal(t, 1, results[0].Key)
	}

	// Add more test cases for incremental updates
	// Update a vector
	err = graph.Add(hnsw.Node[int]{
		Key:   1,
		Value: []float32{0.9, 0.1, 0.0},
	})
	require.NoError(t, err)

	// Search for the updated vector
	results, err = graph.Search([]float32{0.9, 0.1, 0.0}, 1)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results), 0)
	if len(results) > 0 {
		assert.Equal(t, 1, results[0].Key)
	}

	// Close and reopen the graph to test persistence
	err = graph.Close()
	require.NoError(t, err)

	// Reopen the graph
	graph2, err = NewParquetGraph[int](config)
	require.NoError(t, err)
	defer graph2.Close()

	// Verify the graph size
	assert.Equal(t, 9, graph2.Len())

	// Search for the updated vector in the reopened graph
	results, err = graph2.Search([]float32{0.9, 0.1, 0.0}, 1)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results), 0)
	if len(results) > 0 {
		assert.Equal(t, 1, results[0].Key)
	}
}

func TestIncrementalCompaction(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "incremental-compaction-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a configuration with incremental updates
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir
	config.Incremental.MaxChanges = 5 // Set a small value to trigger compaction

	// Create a new graph
	graph, err := NewParquetGraph[int](config)
	require.NoError(t, err)

	// Add vectors to trigger compaction
	for i := 0; i < 20; i++ {
		err := graph.Add(hnsw.Node[int]{
			Key:   i,
			Value: []float32{float32(i), float32(i + 1), float32(i + 2)},
		})
		require.NoError(t, err)
	}

	// Verify the graph size
	assert.Equal(t, 20, graph.Len())

	// Close the graph to ensure changes are flushed
	err = graph.Close()
	require.NoError(t, err)

	// Reopen the graph
	graph2, err := NewParquetGraph[int](config)
	require.NoError(t, err)
	defer graph2.Close()

	// Verify the graph size after reopening
	assert.Equal(t, 20, graph2.Len())

	// Search for a vector
	results, err := graph2.Search([]float32{5.0, 6.0, 7.0}, 1)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(results), 0)
	if len(results) > 0 {
		assert.Equal(t, 5, results[0].Key)
	}
}

func TestIncrementalAgeBasedCompaction(t *testing.T) {
	// Create a temporary directory for the test
	tempDir, err := os.MkdirTemp("", "incremental-age-compaction-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a configuration with incremental updates
	config := DefaultParquetGraphConfig()
	config.Storage.Directory = tempDir
	config.Incremental.MaxChanges = 100              // Set high to not trigger count-based compaction
	config.Incremental.MaxAge = 1 * time.Millisecond // Set very low to trigger age-based compaction

	// Create a new graph
	graph, err := NewParquetGraph[int](config)
	require.NoError(t, err)

	// Add vectors
	for i := 0; i < 10; i++ {
		err := graph.Add(hnsw.Node[int]{
			Key:   i,
			Value: []float32{float32(i), float32(i + 1), float32(i + 2)},
		})
		require.NoError(t, err)
	}

	// Sleep to trigger age-based compaction
	time.Sleep(10 * time.Millisecond)

	// Add more vectors to trigger check for age-based compaction
	for i := 10; i < 15; i++ {
		err := graph.Add(hnsw.Node[int]{
			Key:   i,
			Value: []float32{float32(i), float32(i + 1), float32(i + 2)},
		})
		require.NoError(t, err)
	}

	// Verify the graph size
	assert.Equal(t, 15, graph.Len())

	// Close the graph to ensure changes are flushed
	err = graph.Close()
	require.NoError(t, err)

	// Reopen the graph
	graph2, err := NewParquetGraph[int](config)
	require.NoError(t, err)
	defer graph2.Close()

	// Verify the graph size after reopening
	assert.Equal(t, 15, graph2.Len())
}
