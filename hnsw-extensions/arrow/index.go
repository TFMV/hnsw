package arrow

import (
	"cmp"
	"fmt"
	"sync"

	"github.com/TFMV/hnsw"
)

// ArrowIndex is a high-level interface for the Arrow-based HNSW graph
// It implements the hnsw.VectorIndex interface
type ArrowIndex[K cmp.Ordered] struct {
	graph *ArrowGraph[K]
	mu    sync.RWMutex
}

// NewArrowIndex creates a new Arrow-based HNSW index
func NewArrowIndex[K cmp.Ordered](config ArrowGraphConfig) (*ArrowIndex[K], error) {
	graph, err := NewArrowGraph[K](config)
	if err != nil {
		return nil, fmt.Errorf("failed to create graph: %w", err)
	}

	index := &ArrowIndex[K]{
		graph: graph,
	}

	return index, nil
}

// Add adds a vector to the index
func (idx *ArrowIndex[K]) Add(key K, vector []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.graph.Add(key, vector)
}

// BatchAdd adds multiple vectors to the index
func (idx *ArrowIndex[K]) BatchAdd(keys []K, vectors [][]float32) []error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.graph.BatchAdd(keys, vectors)
}

// Search finds the k nearest neighbors to the query vector
func (idx *ArrowIndex[K]) Search(query []float32, k int) ([]hnsw.Node[K], error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.Search(query, k)
}

// BatchSearch performs multiple searches in parallel
func (idx *ArrowIndex[K]) BatchSearch(queries [][]float32, k int) ([][]hnsw.Node[K], error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.BatchSearch(queries, k)
}

// Delete removes a vector from the index
func (idx *ArrowIndex[K]) Delete(key K) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.graph.Delete(key)
}

// BatchDelete removes multiple vectors from the index
func (idx *ArrowIndex[K]) BatchDelete(keys []K) []bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.graph.BatchDelete(keys)
}

// Save saves the index to disk
func (idx *ArrowIndex[K]) Save() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.graph.Save()
}

// Load loads the index from disk
func (idx *ArrowIndex[K]) Load() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.graph.Load()
}

// Close releases resources used by the index
func (idx *ArrowIndex[K]) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.graph.Close()
}

// Len returns the number of vectors in the index
func (idx *ArrowIndex[K]) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.Len()
}

// Dims returns the dimensionality of vectors in the index
func (idx *ArrowIndex[K]) Dims() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.Dims()
}

// GetConfig returns the configuration of the index
func (idx *ArrowIndex[K]) GetConfig() ArrowGraphConfig {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return ArrowGraphConfig{
		M:        idx.graph.M,
		Ml:       idx.graph.Ml,
		EfSearch: idx.graph.EfSearch,
		Distance: idx.graph.Distance,
		Storage:  idx.graph.storage.config,
	}
}

// SetEfSearch sets the size of the dynamic candidate list during search
func (idx *ArrowIndex[K]) SetEfSearch(efSearch int) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.graph.EfSearch = efSearch
}

// GetEfSearch returns the size of the dynamic candidate list during search
func (idx *ArrowIndex[K]) GetEfSearch() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.EfSearch
}

// GetM returns the maximum number of connections per node
func (idx *ArrowIndex[K]) GetM() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.M
}

// GetMl returns the level generation factor
func (idx *ArrowIndex[K]) GetMl() float64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.Ml
}

// GetDistance returns the distance function
func (idx *ArrowIndex[K]) GetDistance() hnsw.DistanceFunc {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.Distance
}

// GetStorageConfig returns the storage configuration
func (idx *ArrowIndex[K]) GetStorageConfig() ArrowStorageConfig {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.storage.config
}

// GetVector retrieves a vector from the index
func (idx *ArrowIndex[K]) GetVector(key K) ([]float32, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.graph.vectorStore.GetVector(key)
}

// GetVectors retrieves multiple vectors from the index
func (idx *ArrowIndex[K]) GetVectors(keys []K) ([][]float32, []error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	vectors := make([][]float32, len(keys))
	errors := make([]error, len(keys))

	for i, key := range keys {
		vectors[i], errors[i] = idx.graph.vectorStore.GetVector(key)
	}

	return vectors, errors
}

// Optimize optimizes the index for search performance
// This can be called periodically to improve search performance
func (idx *ArrowIndex[K]) Optimize() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Flush any pending changes to disk
	if err := idx.graph.vectorStore.Flush(); err != nil {
		return fmt.Errorf("failed to flush vector store: %w", err)
	}

	// Save the current state of the graph
	if err := idx.graph.Save(); err != nil {
		return fmt.Errorf("failed to save graph: %w", err)
	}

	return nil
}

// Stats returns statistics about the index
func (idx *ArrowIndex[K]) Stats() map[string]interface{} {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	stats := make(map[string]interface{})
	stats["num_vectors"] = idx.graph.nodeCount
	stats["dimensions"] = idx.graph.dimensions
	stats["num_layers"] = len(idx.graph.layers)

	// Calculate average connections per node
	totalConnections := 0
	for _, layer := range idx.graph.layers {
		for _, neighbors := range layer {
			totalConnections += len(neighbors)
		}
	}

	if idx.graph.nodeCount > 0 {
		stats["avg_connections_per_node"] = float64(totalConnections) / float64(idx.graph.nodeCount)
	} else {
		stats["avg_connections_per_node"] = 0.0
	}

	// Add storage stats
	storageStats := idx.graph.storage.Stats()
	for k, v := range storageStats {
		stats["storage_"+k] = v
	}

	return stats
}
