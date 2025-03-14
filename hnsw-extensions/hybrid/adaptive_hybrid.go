package hybrid

import (
	"cmp"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/TFMV/hnsw"
)

// AdaptiveHybridIndex implements a hybrid index that dynamically selects
// the best strategy for each query based on runtime performance metrics
type AdaptiveHybridIndex[K cmp.Ordered] struct {
	// Underlying indexes
	exactIndex  *ExactIndex[K]
	hnswAdapter *HNSWAdapter[K]
	lshIndex    *LSHIndex[K]

	// Adaptive strategy selector
	selector *AdaptiveSelector[K]

	// Distance function
	distFunc hnsw.DistanceFunc

	// Mutex for thread safety
	mu sync.RWMutex

	// Stats
	vectorCount int
}

// NewAdaptiveHybridIndex creates a new adaptive hybrid index
func NewAdaptiveHybridIndex[K cmp.Ordered](
	exactIndex *ExactIndex[K],
	hnswGraph *hnsw.Graph[K],
	lshIndex *LSHIndex[K],
	distFunc hnsw.DistanceFunc,
	config AdaptiveConfig,
) *AdaptiveHybridIndex[K] {
	// Create adapters
	hnswAdapter := NewHNSWAdapter(hnswGraph, distFunc)

	// Create the adaptive selector
	selector := NewAdaptiveSelector(
		exactIndex,
		hnswGraph,
		lshIndex,
		nil, // No partitioner for now
		config,
	)

	return &AdaptiveHybridIndex[K]{
		exactIndex:  exactIndex,
		hnswAdapter: hnswAdapter,
		lshIndex:    lshIndex,
		selector:    selector,
		distFunc:    distFunc,
	}
}

// Add adds a vector to the index
func (idx *AdaptiveHybridIndex[K]) Add(key K, vector []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Add to all underlying indexes
	if idx.exactIndex != nil {
		if err := idx.exactIndex.Add(key, vector); err != nil {
			return fmt.Errorf("failed to add to exact index: %s", err.Error())
		}
	}

	if idx.hnswAdapter != nil {
		if err := idx.hnswAdapter.Add(key, vector); err != nil {
			return fmt.Errorf("failed to add to HNSW index: %s", err.Error())
		}
	}

	if idx.lshIndex != nil {
		if err := idx.lshIndex.Add(key, vector); err != nil {
			return fmt.Errorf("failed to add to LSH index: %s", err.Error())
		}
	}

	idx.vectorCount++
	idx.selector.UpdateDatasetSize(idx.vectorCount)
	return nil
}

// BatchAdd adds multiple vectors to the index
func (idx *AdaptiveHybridIndex[K]) BatchAdd(keys []K, vectors [][]float32) error {
	if len(keys) != len(vectors) {
		return fmt.Errorf("number of keys (%d) does not match number of vectors (%d)", len(keys), len(vectors))
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Add to all underlying indexes
	if idx.exactIndex != nil {
		if err := idx.exactIndex.BatchAdd(keys, vectors); err != nil {
			return fmt.Errorf("failed to batch add to exact index: %s", err.Error())
		}
	}

	if idx.hnswAdapter != nil {
		errs := idx.hnswAdapter.BatchAdd(keys, vectors)
		for _, err := range errs {
			if err != nil {
				return fmt.Errorf("failed to batch add to HNSW index: %s", err.Error())
			}
		}
	}

	if idx.lshIndex != nil {
		errs := idx.lshIndex.BatchAdd(keys, vectors)
		for _, err := range errs {
			if err != nil {
				return fmt.Errorf("failed to batch add to LSH index: %s", err.Error())
			}
		}
	}

	idx.vectorCount += len(keys)
	idx.selector.UpdateDatasetSize(idx.vectorCount)
	return nil
}

// Search performs a k-nearest neighbor search
func (idx *AdaptiveHybridIndex[K]) Search(query []float32, k int) ([]K, []float32, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Select the best strategy for this query
	startTime := time.Now()
	strategy := idx.selector.SelectStrategy(query, k)

	var keys []K
	var distances []float32
	var resultCount int

	// Execute search using the selected strategy
	switch strategy {
	case ExactIndexType:
		if idx.exactIndex != nil {
			// ExactIndex.Search returns []hnsw.Node[K], error
			nodes, err := idx.exactIndex.Search(query, k)
			if err == nil && len(nodes) > 0 {
				keys = make([]K, len(nodes))
				distances = make([]float32, len(nodes))
				for i, node := range nodes {
					keys[i] = node.Key
					distances[i] = idx.distFunc(query, node.Value)
				}
				resultCount = len(keys)
			}
		} else {
			// Fall back to HNSW if exact index is not available
			strategy = HNSWIndexType
		}
	case HybridIndexType:
		// For hybrid strategy, we'll use HNSW as the primary index
		// and fall back to other indexes if needed
		if idx.hnswAdapter != nil {
			keys, distances = idx.hnswAdapter.Search(query, k)
			resultCount = len(keys)

			// If HNSW didn't return enough results, try LSH
			if len(keys) < k && idx.lshIndex != nil {
				lshKeys, lshDistances := idx.lshIndex.Search(query, k)

				// Merge results (simple approach for now)
				for i, key := range lshKeys {
					// Check if this key is already in the results
					found := false
					for _, existingKey := range keys {
						if existingKey == key {
							found = true
							break
						}
					}

					if !found {
						keys = append(keys, key)
						distances = append(distances, lshDistances[i])
						resultCount++

						// Stop if we have enough results
						if resultCount >= k {
							break
						}
					}
				}
			}
		} else if idx.lshIndex != nil {
			// Fall back to LSH if HNSW is not available
			keys, distances = idx.lshIndex.Search(query, k)
			resultCount = len(keys)
		} else if idx.exactIndex != nil {
			// Fall back to Exact if neither HNSW nor LSH is available
			nodes, err := idx.exactIndex.Search(query, k)
			if err == nil && len(nodes) > 0 {
				keys = make([]K, len(nodes))
				distances = make([]float32, len(nodes))
				for i, node := range nodes {
					keys[i] = node.Key
					distances[i] = idx.distFunc(query, node.Value)
				}
				resultCount = len(keys)
			}
		}
	}

	if strategy == HNSWIndexType || (strategy == ExactIndexType && keys == nil) {
		if idx.hnswAdapter != nil {
			keys, distances = idx.hnswAdapter.Search(query, k)
			resultCount = len(keys)
		} else {
			// Fall back to LSH if HNSW is not available
			strategy = LSHIndexType
		}
	}

	if strategy == LSHIndexType || ((strategy == ExactIndexType || strategy == HNSWIndexType) && keys == nil) {
		if idx.lshIndex != nil {
			keys, distances = idx.lshIndex.Search(query, k)
			resultCount = len(keys)
		} else {
			return nil, nil, fmt.Errorf("no suitable index available for search")
		}
	}

	// Record metrics for this query
	duration := time.Since(startTime)
	metrics := QueryMetrics{
		Strategy:    strategy,
		QueryVector: query,
		Dimension:   len(query),
		K:           k,
		Duration:    duration,
		ResultCount: resultCount,
	}

	// Calculate distance statistics if we have results
	if len(distances) > 0 {
		var sum, min, max, sumSquares float32
		min = distances[0]
		max = distances[0]

		for _, d := range distances {
			sum += d
			sumSquares += d * d
			if d < min {
				min = d
			}
			if d > max {
				max = d
			}
		}

		mean := sum / float32(len(distances))
		variance := (sumSquares / float32(len(distances))) - (mean * mean)

		metrics.DistanceStats = DistanceStats{
			Min:      min,
			Max:      max,
			Mean:     mean,
			Variance: variance,
		}
	}

	// Record the metrics asynchronously to avoid blocking the search
	go idx.selector.RecordQueryMetrics(metrics)

	if keys == nil {
		return nil, nil, fmt.Errorf("search failed with all available strategies")
	}

	return keys, distances, nil
}

// Delete removes a vector from the index
func (idx *AdaptiveHybridIndex[K]) Delete(key K) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Delete from all underlying indexes
	var errs []error

	if idx.exactIndex != nil {
		success := idx.exactIndex.Delete(key)
		if !success {
			errs = append(errs, fmt.Errorf("failed to delete from exact index"))
		}
	}

	if idx.hnswAdapter != nil {
		success := idx.hnswAdapter.Delete(key)
		if !success {
			errs = append(errs, fmt.Errorf("failed to delete from HNSW index"))
		}
	}

	if idx.lshIndex != nil {
		success := idx.lshIndex.Delete(key)
		if !success {
			errs = append(errs, fmt.Errorf("failed to delete from LSH index"))
		}
	}

	// If any errors occurred, return a combined error
	if len(errs) > 0 {
		errMsg := "errors occurred during deletion:"
		for _, err := range errs {
			errMsg += " " + err.Error()
		}
		return errors.New(errMsg)
	}

	idx.vectorCount--
	return nil
}

// Count returns the number of vectors in the index
func (idx *AdaptiveHybridIndex[K]) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.vectorCount
}

// GetStats returns statistics about the adaptive index
func (idx *AdaptiveHybridIndex[K]) GetStats() map[string]interface{} {
	stats := idx.selector.GetStats()

	// Add index-specific stats
	stats["vector_count"] = idx.Count()

	// Add underlying index stats if available
	if idx.exactIndex != nil {
		stats["exact_index_count"] = idx.exactIndex.Len()
	}

	if idx.hnswAdapter != nil {
		stats["hnsw_index_count"] = idx.hnswAdapter.Len()
	}

	if idx.lshIndex != nil {
		stats["lsh_index_count"] = idx.lshIndex.Len()
	}

	return stats
}

// ResetStats resets all performance statistics
func (idx *AdaptiveHybridIndex[K]) ResetStats() {
	idx.selector.ResetStats()
}
