package arrow

import (
	"cmp"
	"fmt"
	"os"
	"sort"
	"sync"

	"github.com/TFMV/hnsw"
)

// SearchResult represents a search result with distance information
type SearchResult[K cmp.Ordered] struct {
	Node     hnsw.Node[K]
	Distance float32
}

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

// SearchWithNegative finds the k nearest neighbors to the query vector while avoiding the negative example
func (idx *ArrowIndex[K]) SearchWithNegative(query, negative []float32, k int, negWeight float32) ([]hnsw.Node[K], error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Get more candidates than needed to allow for reranking
	nodes, err := idx.graph.Search(query, k*2)
	if err != nil {
		return nil, err
	}

	if len(nodes) == 0 {
		return nodes, nil
	}

	// Convert to SearchResult with distance information
	results := make([]SearchResult[K], len(nodes))
	for i, node := range nodes {
		// Calculate original distance to query
		queryDist := idx.graph.Distance(node.Value, query)

		// Calculate distance to negative example
		negDist := idx.graph.Distance(node.Value, negative)

		// Adjust the distance: smaller distance to query is better, larger distance to negative is better
		// So we subtract the negative distance (multiplied by weight) from the original distance
		// This will make nodes that are far from the negative example more favorable
		adjustedDist := queryDist - (negWeight * negDist)

		results[i] = SearchResult[K]{
			Node:     node,
			Distance: adjustedDist,
		}
	}

	// Sort by adjusted distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	// Return top k results
	if len(results) > k {
		results = results[:k]
	}

	// Convert back to Node format
	finalNodes := make([]hnsw.Node[K], len(results))
	for i, result := range results {
		finalNodes[i] = result.Node
	}

	return finalNodes, nil
}

// SearchWithNegatives finds the k nearest neighbors to the query vector while avoiding multiple negative examples
func (idx *ArrowIndex[K]) SearchWithNegatives(query []float32, negatives [][]float32, k int, negWeight float32) ([]hnsw.Node[K], error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Get more candidates than needed to allow for reranking
	nodes, err := idx.graph.Search(query, k*2)
	if err != nil {
		return nil, err
	}

	if len(nodes) == 0 || len(negatives) == 0 {
		return nodes, nil
	}

	// Convert to SearchResult with distance information
	results := make([]SearchResult[K], len(nodes))
	for i, node := range nodes {
		// Calculate original distance to query
		queryDist := idx.graph.Distance(node.Value, query)

		// Calculate average distance to all negative examples
		var totalNegDist float32
		for _, negative := range negatives {
			totalNegDist += idx.graph.Distance(node.Value, negative)
		}
		avgNegDist := totalNegDist / float32(len(negatives))

		// Adjust the distance
		adjustedDist := queryDist - (negWeight * avgNegDist)

		results[i] = SearchResult[K]{
			Node:     node,
			Distance: adjustedDist,
		}
	}

	// Sort by adjusted distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	// Return top k results
	if len(results) > k {
		results = results[:k]
	}

	// Convert back to Node format
	finalNodes := make([]hnsw.Node[K], len(results))
	for i, result := range results {
		finalNodes[i] = result.Node
	}

	return finalNodes, nil
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

	storageConfig := idx.graph.storage.config

	return ArrowGraphConfig{
		M:                 idx.graph.M,
		Ml:                idx.graph.Ml,
		EfSearch:          idx.graph.EfSearch,
		Distance:          idx.graph.Distance,
		StorageDir:        storageConfig.Directory,
		NumWorkers:        storageConfig.NumWorkers,
		BatchSize:         int(storageConfig.BatchSize),
		MemoryPoolSize:    storageConfig.MemoryPoolSize,
		MemoryMap:         storageConfig.MemoryMap,
		FlushInterval:     idx.graph.vectorStore.flushInterval,
		EnableCompression: false, // Default values for now
		CompressionType:   "zstd",
		CompressionLevel:  3,
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

	// Basic stats
	stats["num_vectors"] = idx.graph.nodeCount
	stats["dimensions"] = idx.graph.dimensions
	stats["num_layers"] = len(idx.graph.layers)
	stats["m"] = idx.graph.M
	stats["ef_search"] = idx.graph.EfSearch
	stats["ml"] = idx.graph.Ml

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

	// Arrow-specific stats
	storageConfig := idx.graph.storage.config
	stats["storage_dir"] = storageConfig.Directory
	stats["batch_size"] = storageConfig.BatchSize
	stats["num_workers"] = storageConfig.NumWorkers
	stats["memory_map"] = storageConfig.MemoryMap
	stats["memory_pool_size"] = storageConfig.MemoryPoolSize

	// Calculate disk usage
	diskUsage := idx.getDiskUsage()
	stats["disk_usage_bytes"] = diskUsage
	stats["disk_usage_mb"] = float64(diskUsage) / (1024 * 1024)

	// Calculate memory usage
	memoryUsage := idx.getMemoryUsage()
	stats["memory_usage_bytes"] = memoryUsage
	stats["memory_usage_mb"] = float64(memoryUsage) / (1024 * 1024)

	// Vector store stats
	if idx.graph.vectorStore != nil {
		stats["vector_cache_size"] = len(idx.graph.vectorStore.cache)
		stats["pending_writes"] = len(idx.graph.vectorStore.pendingWrites)
		stats["pending_deletes"] = len(idx.graph.vectorStore.pendingDeletes)
		stats["flush_interval_seconds"] = idx.graph.vectorStore.flushInterval.Seconds()
	}

	// Add storage stats
	storageStats := idx.graph.storage.Stats()
	for k, v := range storageStats {
		stats["storage_"+k] = v
	}

	return stats
}

// getDiskUsage calculates the total disk usage of the index
func (idx *ArrowIndex[K]) getDiskUsage() int64 {
	var totalSize int64

	// Check vectors file
	if fileInfo, err := os.Stat(idx.graph.storage.vectorsFile); err == nil {
		totalSize += fileInfo.Size()
	}

	// Check layers file
	if fileInfo, err := os.Stat(idx.graph.storage.layersFile); err == nil {
		totalSize += fileInfo.Size()
	}

	// Check neighbors file
	if fileInfo, err := os.Stat(idx.graph.storage.neighborsFile); err == nil {
		totalSize += fileInfo.Size()
	}

	return totalSize
}

// getMemoryUsage estimates the memory usage of the index
func (idx *ArrowIndex[K]) getMemoryUsage() int64 {
	var totalSize int64

	// Estimate size of layers
	layersSize := 0
	for _, layer := range idx.graph.layers {
		// Each layer is a map of maps
		layersSize += len(layer) * 16 // Approximate size of map entry

		// Add size of neighbor maps
		for _, neighbors := range layer {
			layersSize += len(neighbors) * 16 // Approximate size of map entry
		}
	}
	totalSize += int64(layersSize)

	// Estimate size of vector cache
	if idx.graph.vectorStore != nil {
		cacheSize := 0
		for _, vector := range idx.graph.vectorStore.cache {
			cacheSize += len(vector) * 4 // 4 bytes per float32
		}
		totalSize += int64(cacheSize)

		// Add size of pending writes
		pendingWritesSize := 0
		for _, vector := range idx.graph.vectorStore.pendingWrites {
			pendingWritesSize += len(vector) * 4 // 4 bytes per float32
		}
		totalSize += int64(pendingWritesSize)
	}

	return totalSize
}

// Facet represents a facet (attribute) of a vector
type Facet struct {
	Field string      // Field name
	Value interface{} // Field value
}

// FacetFilter represents a filter for facets
type FacetFilter struct {
	Field     string      // Field name
	Operation string      // Filter operation (equals, range, contains, etc.)
	Value     interface{} // Filter value
}

// FilterOperation constants
const (
	FilterEquals      = "equals"
	FilterNotEquals   = "not_equals"
	FilterLessThan    = "less_than"
	FilterGreaterThan = "greater_than"
	FilterRange       = "range"
	FilterContains    = "contains"
)

// AddWithFacets adds a vector to the index with facets
func (idx *ArrowIndex[K]) AddWithFacets(key K, vector []float32, facets []Facet) error {
	// First add the vector to the index
	if err := idx.Add(key, vector); err != nil {
		return err
	}

	// Then store the facets in Arrow format
	// This would typically be implemented using a separate facet store
	// For now, we'll just return nil as a placeholder
	return nil
}

// SearchWithFacets searches for vectors with facet filtering
func (idx *ArrowIndex[K]) SearchWithFacets(query []float32, k int, filters []FacetFilter) ([]hnsw.Node[K], error) {
	// First perform a regular search to get candidates
	candidates, err := idx.Search(query, k*2) // Get more candidates than needed
	if err != nil {
		return nil, err
	}

	// Filter candidates based on facets
	// This would typically be implemented using a separate facet store
	// For now, we'll just return the candidates as a placeholder
	return candidates, nil
}

// Metadata represents metadata for a vector
type Metadata map[string]interface{}

// AddWithMetadata adds a vector to the index with metadata
func (idx *ArrowIndex[K]) AddWithMetadata(key K, vector []float32, metadata Metadata) error {
	// First add the vector to the index
	if err := idx.Add(key, vector); err != nil {
		return err
	}

	// Then store the metadata in Arrow format
	// This would typically be implemented using a separate metadata store
	// For now, we'll just return nil as a placeholder
	return nil
}

// GetMetadata retrieves metadata for a vector
func (idx *ArrowIndex[K]) GetMetadata(key K) (Metadata, error) {
	// Check if the vector exists
	if _, err := idx.GetVector(key); err != nil {
		return nil, fmt.Errorf("vector not found: %w", err)
	}

	// Retrieve metadata from storage
	// This would typically be implemented using a separate metadata store
	// For now, we'll just return an empty metadata as a placeholder
	return Metadata{}, nil
}

// SearchWithMetadataFilter searches for vectors with metadata filtering
func (idx *ArrowIndex[K]) SearchWithMetadataFilter(query []float32, k int, filter []byte) ([]hnsw.Node[K], error) {
	// First perform a regular search to get candidates
	candidates, err := idx.Search(query, k*2) // Get more candidates than needed
	if err != nil {
		return nil, err
	}

	// Filter candidates based on metadata
	// This would typically be implemented using a separate metadata store
	// For now, we'll just return the candidates as a placeholder
	return candidates, nil
}
