// Package hybrid provides hybrid index structures that combine HNSW with
// complementary indexing approaches to overcome limitations in specific scenarios.
package hybrid

import (
	"cmp"
	"fmt"
	"sync"

	"github.com/TFMV/hnsw"
)

// VectorIndex defines a common interface for all vector indexing structures.
// This allows different implementations to be used interchangeably.
type VectorIndex[K cmp.Ordered] interface {
	// Add adds a vector to the index
	Add(key K, vector []float32) error

	// BatchAdd adds multiple vectors to the index
	BatchAdd(keys []K, vectors [][]float32) error

	// Search finds the k nearest neighbors to the query vector
	Search(query []float32, k int) ([]hnsw.Node[K], error)

	// Delete removes a vector from the index
	Delete(key K) bool

	// BatchDelete removes multiple vectors from the index
	BatchDelete(keys []K) []bool

	// Len returns the number of vectors in the index
	Len() int

	// Close releases resources used by the index
	Close() error
}

// SearchableIndex defines the interface for any index that can be used in the hybrid index.
type SearchableIndex[K cmp.Ordered] interface {
	// Add adds a vector to the index.
	Add(key K, vector []float32) error

	// BatchAdd adds multiple vectors to the index.
	BatchAdd(keys []K, vectors [][]float32) []error

	// Search finds the k nearest neighbors to the query vector.
	Search(query []float32, k int) ([]K, []float32)

	// Delete removes a vector from the index.
	Delete(key K) bool

	// BatchDelete removes multiple vectors from the index.
	BatchDelete(keys []K) []bool

	// Len returns the number of vectors in the index.
	Len() int

	// Close releases resources used by the index.
	Close()
}

// IndexType represents the type of index to use
type IndexType int

const (
	// ExactIndexType uses brute force exact search
	ExactIndexType IndexType = iota

	// HNSWIndexType uses the HNSW graph
	HNSWIndexType

	// LSHIndexType uses Locality-Sensitive Hashing
	LSHIndexType

	// HybridIndexType uses a combination of approaches
	HybridIndexType
)

// UsePartitioning returns whether the index type uses partitioning
func (t IndexType) UsePartitioning() bool {
	return t == HybridIndexType
}

// IndexConfig contains configuration options for the hybrid index
type IndexConfig struct {
	// The type of index to use
	Type IndexType

	// The threshold for switching between exact and HNSW search
	// (number of vectors)
	ExactThreshold int

	// HNSW configuration
	M        int
	Ml       float64
	EfSearch int
	Distance hnsw.DistanceFunc

	// LSH configuration
	NumHashTables int
	NumHashBits   int

	// Partitioning configuration
	NumPartitions int
	PartitionSize int
}

// DefaultIndexConfig returns a default configuration for the hybrid index
func DefaultIndexConfig() IndexConfig {
	return IndexConfig{
		Type:           HybridIndexType,
		ExactThreshold: 1000,
		M:              16,
		Ml:             0.25,
		EfSearch:       20,
		Distance:       hnsw.CosineDistance,
		NumHashTables:  4,
		NumHashBits:    8,
		NumPartitions:  10,
		PartitionSize:  10000,
	}
}

// HybridIndex is the main implementation of the VectorIndex interface
// that combines multiple indexing approaches.
type HybridIndex[K cmp.Ordered] struct {
	config IndexConfig

	// The underlying indexes
	exactIndex *ExactIndex[K]
	hnswIndex  *hnsw.Graph[K]
	lshIndex   *LSHIndex[K]

	// Partitioning
	partitioner *Partitioner[K]

	// Statistics
	stats IndexStats

	// Mutex for thread safety
	mu sync.RWMutex

	// Vector storage
	vectors map[K][]float32
}

// IndexStats contains statistics about the index
type IndexStats struct {
	TotalVectors     int
	ExactVectors     int
	HNSWVectors      int
	NumPartitions    int
	AvgPartitionSize float64
}

// NewHybridIndex creates a new hybrid index with the given configuration
func NewHybridIndex[K cmp.Ordered](config IndexConfig) (*HybridIndex[K], error) {
	if config.Type == IndexType(0) {
		config = DefaultIndexConfig()
	}

	// Validate configuration
	if config.M <= 0 {
		return nil, fmt.Errorf("M must be greater than 0")
	}
	if config.Ml <= 0 || config.Ml >= 1 {
		return nil, fmt.Errorf("Ml must be between 0 and 1")
	}
	if config.EfSearch <= 0 {
		return nil, fmt.Errorf("EfSearch must be greater than 0")
	}

	// Create the hybrid index
	idx := &HybridIndex[K]{
		config:  config,
		stats:   IndexStats{},
		vectors: make(map[K][]float32),
	}

	// Initialize the appropriate indexes based on configuration
	switch config.Type {
	case ExactIndexType:
		idx.exactIndex = NewExactIndex[K](config.Distance)
	case HNSWIndexType:
		hnswGraph, err := hnsw.NewGraphWithConfig[K](
			config.M,
			config.Ml,
			config.EfSearch,
			config.Distance,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create HNSW graph: %w", err)
		}
		idx.hnswIndex = hnswGraph
	case LSHIndexType:
		idx.lshIndex = NewLSHIndex[K](
			config.NumHashTables,
			config.NumHashBits,
			config.Distance,
		)
	case HybridIndexType:
		// For hybrid, initialize all indexes
		idx.exactIndex = NewExactIndex[K](config.Distance)

		hnswGraph, err := hnsw.NewGraphWithConfig[K](
			config.M,
			config.Ml,
			config.EfSearch,
			config.Distance,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create HNSW graph: %w", err)
		}
		idx.hnswIndex = hnswGraph

		idx.lshIndex = NewLSHIndex[K](
			config.NumHashTables,
			config.NumHashBits,
			config.Distance,
		)

		// Initialize partitioner
		idx.partitioner = NewPartitioner[K](
			config.NumPartitions,
			config.Distance,
		)
	}

	return idx, nil
}

// Add adds a vector to the index
func (idx *HybridIndex[K]) Add(key K, vector []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Store vector
	idx.vectors[key] = vector

	// Update statistics
	idx.stats.TotalVectors++

	// Choose the appropriate index based on configuration and current state
	switch idx.config.Type {
	case ExactIndexType:
		return idx.exactIndex.Add(key, vector)

	case HNSWIndexType:
		node := hnsw.Node[K]{
			Key:   key,
			Value: vector,
		}
		return idx.hnswIndex.Add(node)

	case LSHIndexType:
		return idx.lshIndex.Add(key, vector)

	case HybridIndexType:
		// For hybrid approach, use tiered strategy
		if idx.stats.TotalVectors <= idx.config.ExactThreshold {
			// Use exact index for small datasets
			idx.stats.ExactVectors++
			return idx.exactIndex.Add(key, vector)
		} else {
			// For larger datasets, use partitioning strategy
			partitionIdx := idx.partitioner.AssignPartition(vector)
			_ = partitionIdx // Use the partition index in the future

			// Add to HNSW index
			idx.stats.HNSWVectors++
			node := hnsw.Node[K]{
				Key:   key,
				Value: vector,
			}
			return idx.hnswIndex.Add(node)
		}
	}

	return fmt.Errorf("unknown index type: %v", idx.config.Type)
}

// BatchAdd adds multiple vectors to the index
func (idx *HybridIndex[K]) BatchAdd(keys []K, vectors [][]float32) error {
	if len(keys) != len(vectors) {
		return fmt.Errorf("number of keys (%d) does not match number of vectors (%d)", len(keys), len(vectors))
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Store vectors
	for i, key := range keys {
		idx.vectors[key] = vectors[i]
	}

	// Update statistics
	idx.stats.TotalVectors += len(keys)

	// Choose the appropriate index based on configuration and current state
	switch idx.config.Type {
	case ExactIndexType:
		return idx.exactIndex.BatchAdd(keys, vectors)

	case HNSWIndexType:
		// Add each vector individually since BatchAdd expects Node objects
		for i, key := range keys {
			node := hnsw.Node[K]{
				Key:   key,
				Value: vectors[i],
			}
			if err := idx.hnswIndex.Add(node); err != nil {
				return err
			}
		}
		return nil

	case LSHIndexType:
		// LSH BatchAdd returns []error, but we need to return a single error
		errors := idx.lshIndex.BatchAdd(keys, vectors)
		for _, err := range errors {
			if err != nil {
				return err
			}
		}
		return nil

	case HybridIndexType:
		// For hybrid approach, use tiered strategy
		if idx.stats.TotalVectors <= idx.config.ExactThreshold {
			// Use exact index for small datasets
			idx.stats.ExactVectors += len(keys)
			return idx.exactIndex.BatchAdd(keys, vectors)
		} else {
			// For larger datasets, use partitioning strategy with HNSW
			for i, key := range keys {
				// Assign to a partition
				idx.partitioner.AssignPartition(vectors[i])

				// Add to HNSW index
				node := hnsw.Node[K]{
					Key:   key,
					Value: vectors[i],
				}
				if err := idx.hnswIndex.Add(node); err != nil {
					return err
				}
			}

			idx.stats.HNSWVectors += len(keys)
			return nil
		}
	}

	return fmt.Errorf("unknown index type: %v", idx.config.Type)
}

// Search finds the k nearest neighbors to the query vector
func (idx *HybridIndex[K]) Search(query []float32, k int) ([]hnsw.Node[K], error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Choose the appropriate search strategy based on configuration and current state
	switch idx.config.Type {
	case ExactIndexType:
		return idx.exactIndex.Search(query, k)

	case HNSWIndexType:
		return idx.hnswIndex.Search(query, k)

	case LSHIndexType:
		// Convert LSH results to HNSW nodes
		keys, _ := idx.lshIndex.Search(query, k)
		nodes := make([]hnsw.Node[K], len(keys))
		for i, key := range keys {
			vector := idx.vectors[key]
			nodes[i] = hnsw.Node[K]{
				Key:   key,
				Value: vector,
			}
		}
		return nodes, nil

	case HybridIndexType:
		// For hybrid approach, use tiered strategy
		if idx.stats.TotalVectors <= idx.config.ExactThreshold {
			// Use exact search for small datasets
			return idx.exactIndex.Search(query, k)
		} else if idx.stats.TotalVectors >= idx.config.PartitionSize*idx.config.NumPartitions {
			// For very large datasets, use LSH + HNSW
			// First, find candidate partitions using LSH
			candidates := idx.lshIndex.GetCandidates(query)
			_ = candidates // Use candidates in the future

			// Then search within those partitions using HNSW
			return idx.hnswIndex.Search(query, k)
		} else {
			// For medium datasets, just use HNSW
			return idx.hnswIndex.Search(query, k)
		}
	}

	return nil, fmt.Errorf("unknown index type: %v", idx.config.Type)
}

// Delete removes a vector from the index
func (idx *HybridIndex[K]) Delete(key K) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Update statistics if deletion is successful
	deleted := false

	// Delete from all applicable indexes
	switch idx.config.Type {
	case ExactIndexType:
		deleted = idx.exactIndex.Delete(key)

	case HNSWIndexType:
		deleted = idx.hnswIndex.Delete(key)

	case LSHIndexType:
		deleted = idx.lshIndex.Delete(key)

	case HybridIndexType:
		// For hybrid, try to delete from all indexes
		exactDeleted := idx.exactIndex.Delete(key)
		hnswDeleted := idx.hnswIndex.Delete(key)
		lshDeleted := idx.lshIndex.Delete(key)

		deleted = exactDeleted || hnswDeleted || lshDeleted
	}

	if deleted {
		idx.stats.TotalVectors--
		delete(idx.vectors, key)
	}

	return deleted
}

// BatchDelete removes multiple vectors from the index
func (idx *HybridIndex[K]) BatchDelete(keys []K) []bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	results := make([]bool, len(keys))

	// Delete from all applicable indexes
	switch idx.config.Type {
	case ExactIndexType:
		results = idx.exactIndex.BatchDelete(keys)

	case HNSWIndexType:
		results = idx.hnswIndex.BatchDelete(keys)

	case LSHIndexType:
		results = idx.lshIndex.BatchDelete(keys)

	case HybridIndexType:
		// For hybrid, delete from all indexes
		exactResults := idx.exactIndex.BatchDelete(keys)
		hnswResults := idx.hnswIndex.BatchDelete(keys)
		lshResults := idx.lshIndex.BatchDelete(keys)

		// Combine results
		for i := range results {
			results[i] = exactResults[i] || hnswResults[i] || lshResults[i]
		}
	}

	// Update statistics and remove vectors
	for i, deleted := range results {
		if deleted {
			idx.stats.TotalVectors--
			delete(idx.vectors, keys[i])
		}
	}

	return results
}

// Len returns the number of vectors in the index
func (idx *HybridIndex[K]) Len() int {
	return idx.stats.TotalVectors
}

// Close releases resources used by the index
func (idx *HybridIndex[K]) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Close all applicable indexes
	var err error

	if idx.exactIndex != nil {
		if closeErr := idx.exactIndex.Close(); closeErr != nil {
			err = closeErr
		}
	}

	if idx.lshIndex != nil {
		idx.lshIndex.Close()
	}

	// Clear vectors
	idx.vectors = nil

	return err
}

// GetStats returns statistics about the index
func (idx *HybridIndex[K]) GetStats() IndexStats {
	return idx.stats
}

// GetPartitionStats returns statistics about the partitions.
func (idx *HybridIndex[K]) GetPartitionStats() []int {
	if !idx.config.Type.UsePartitioning() {
		return nil
	}

	return idx.partitioner.GetPartitionStats()
}

// ForceRebalance forces a rebalance of the partitions.
func (idx *HybridIndex[K]) ForceRebalance() {
	if !idx.config.Type.UsePartitioning() {
		return
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Update centroids based on current vectors
	idx.partitioner.UpdateCentroids(idx.vectors)

	// Rebalance partitions
	idx.partitioner.Rebalance(idx.vectors)
}
