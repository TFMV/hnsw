package hybrid

import (
	"cmp"
	"fmt"

	"github.com/TFMV/hnsw"
)

// HNSWAdapter adapts the HNSW Graph to the SearchableIndex interface.
// This allows the HNSW graph to be used in the hybrid index.
type HNSWAdapter[K cmp.Ordered] struct {
	graph    *hnsw.Graph[K]
	distance hnsw.DistanceFunc // Store the distance function separately
}

// NewHNSWAdapter creates a new adapter for an HNSW graph.
func NewHNSWAdapter[K cmp.Ordered](graph *hnsw.Graph[K], distance hnsw.DistanceFunc) *HNSWAdapter[K] {
	return &HNSWAdapter[K]{
		graph:    graph,
		distance: distance,
	}
}

// Add adds a vector to the index.
func (a *HNSWAdapter[K]) Add(key K, vector []float32) error {
	node := hnsw.Node[K]{
		Key:   key,
		Value: vector,
	}
	return a.graph.Add(node)
}

// BatchAdd adds multiple vectors to the index.
func (a *HNSWAdapter[K]) BatchAdd(keys []K, vectors [][]float32) []error {
	if len(keys) != len(vectors) {
		return []error{fmt.Errorf("number of keys (%d) does not match number of vectors (%d)", len(keys), len(vectors))}
	}

	errors := make([]error, len(keys))
	for i := range keys {
		node := hnsw.Node[K]{
			Key:   keys[i],
			Value: vectors[i],
		}
		errors[i] = a.graph.Add(node)
	}
	return errors
}

// Search finds the k nearest neighbors to the query vector.
func (a *HNSWAdapter[K]) Search(query []float32, k int) ([]K, []float32) {
	nodes, err := a.graph.Search(query, k)
	if err != nil {
		return nil, nil
	}

	keys := make([]K, len(nodes))
	distances := make([]float32, len(nodes))
	for i, node := range nodes {
		keys[i] = node.Key
		// Calculate distance using the stored distance function
		if a.distance != nil && len(node.Value) > 0 {
			distances[i] = a.distance(query, node.Value)
		}
	}
	return keys, distances
}

// Delete removes a vector from the index.
func (a *HNSWAdapter[K]) Delete(key K) bool {
	return a.graph.Delete(key)
}

// BatchDelete removes multiple vectors from the index.
func (a *HNSWAdapter[K]) BatchDelete(keys []K) []bool {
	return a.graph.BatchDelete(keys)
}

// Len returns the number of vectors in the index.
func (a *HNSWAdapter[K]) Len() int {
	return a.graph.Len()
}

// Close releases resources used by the index.
func (a *HNSWAdapter[K]) Close() {
	// HNSW graph doesn't have a Close method, but we need to implement the interface
}

// ExactAdapter adapts the ExactIndex to the SearchableIndex interface.
// This allows the ExactIndex to be used in the hybrid index.
type ExactAdapter[K cmp.Ordered] struct {
	index *ExactIndex[K]
}

// NewExactAdapter creates a new adapter for an ExactIndex.
func NewExactAdapter[K cmp.Ordered](index *ExactIndex[K]) *ExactAdapter[K] {
	return &ExactAdapter[K]{
		index: index,
	}
}

// Add adds a vector to the index.
func (a *ExactAdapter[K]) Add(key K, vector []float32) error {
	return a.index.Add(key, vector)
}

// BatchAdd adds multiple vectors to the index.
func (a *ExactAdapter[K]) BatchAdd(keys []K, vectors [][]float32) []error {
	if len(keys) != len(vectors) {
		return []error{fmt.Errorf("number of keys (%d) does not match number of vectors (%d)", len(keys), len(vectors))}
	}

	err := a.index.BatchAdd(keys, vectors)
	errors := make([]error, len(keys))
	if err != nil {
		for i := range errors {
			errors[i] = err
		}
	}
	return errors
}

// Search finds the k nearest neighbors to the query vector.
func (a *ExactAdapter[K]) Search(query []float32, k int) ([]K, []float32) {
	nodes, err := a.index.Search(query, k)
	if err != nil {
		return nil, nil
	}

	keys := make([]K, len(nodes))
	distances := make([]float32, len(nodes))
	for i, node := range nodes {
		keys[i] = node.Key
		// Calculate distance using the index's distance function
		if a.index.distance != nil && len(node.Value) > 0 {
			distances[i] = a.index.distance(query, node.Value)
		}
	}
	return keys, distances
}

// Delete removes a vector from the index.
func (a *ExactAdapter[K]) Delete(key K) bool {
	return a.index.Delete(key)
}

// BatchDelete removes multiple vectors from the index.
func (a *ExactAdapter[K]) BatchDelete(keys []K) []bool {
	return a.index.BatchDelete(keys)
}

// Len returns the number of vectors in the index.
func (a *ExactAdapter[K]) Len() int {
	return a.index.Len()
}

// Close releases resources used by the index.
func (a *ExactAdapter[K]) Close() {
	a.index.Close()
}

// LSHAdapter adapts the LSHIndex to the SearchableIndex interface.
// This allows the LSHIndex to be used in the hybrid index.
type LSHAdapter[K cmp.Ordered] struct {
	index *LSHIndex[K]
}

// NewLSHAdapter creates a new adapter for an LSHIndex.
func NewLSHAdapter[K cmp.Ordered](index *LSHIndex[K]) *LSHAdapter[K] {
	return &LSHAdapter[K]{
		index: index,
	}
}

// Add adds a vector to the index.
func (a *LSHAdapter[K]) Add(key K, vector []float32) error {
	return a.index.Add(key, vector)
}

// BatchAdd adds multiple vectors to the index.
func (a *LSHAdapter[K]) BatchAdd(keys []K, vectors [][]float32) []error {
	return a.index.BatchAdd(keys, vectors)
}

// Search finds the k nearest neighbors to the query vector.
func (a *LSHAdapter[K]) Search(query []float32, k int) ([]K, []float32) {
	return a.index.Search(query, k)
}

// Delete removes a vector from the index.
func (a *LSHAdapter[K]) Delete(key K) bool {
	return a.index.Delete(key)
}

// BatchDelete removes multiple vectors from the index.
func (a *LSHAdapter[K]) BatchDelete(keys []K) []bool {
	return a.index.BatchDelete(keys)
}

// Len returns the number of vectors in the index.
func (a *LSHAdapter[K]) Len() int {
	return a.index.Len()
}

// Close releases resources used by the index.
func (a *LSHAdapter[K]) Close() {
	a.index.Close()
}

// MultiIndexAdapter combines multiple indexes into a single SearchableIndex.
// This allows searching across multiple indexes and combining the results.
type MultiIndexAdapter[K cmp.Ordered] struct {
	indexes []SearchableIndex[K]
}

// NewMultiIndexAdapter creates a new adapter for multiple indexes.
func NewMultiIndexAdapter[K cmp.Ordered](indexes ...SearchableIndex[K]) *MultiIndexAdapter[K] {
	return &MultiIndexAdapter[K]{
		indexes: indexes,
	}
}

// Add adds a vector to all indexes.
func (a *MultiIndexAdapter[K]) Add(key K, vector []float32) error {
	for _, idx := range a.indexes {
		if err := idx.Add(key, vector); err != nil {
			return err
		}
	}
	return nil
}

// BatchAdd adds multiple vectors to all indexes.
func (a *MultiIndexAdapter[K]) BatchAdd(keys []K, vectors [][]float32) []error {
	if len(keys) != len(vectors) {
		return []error{fmt.Errorf("number of keys (%d) does not match number of vectors (%d)", len(keys), len(vectors))}
	}

	// Initialize errors
	errors := make([]error, len(keys))

	// Add to all indexes
	for _, idx := range a.indexes {
		idxErrors := idx.BatchAdd(keys, vectors)
		for i, err := range idxErrors {
			if err != nil && errors[i] == nil {
				errors[i] = err
			}
		}
	}

	return errors
}

// Search finds the k nearest neighbors across all indexes.
func (a *MultiIndexAdapter[K]) Search(query []float32, k int) ([]K, []float32) {
	type result struct {
		key      K
		distance float32
	}

	// Collect results from all indexes
	var allResults []result
	for _, idx := range a.indexes {
		keys, distances := idx.Search(query, k)
		for i := range keys {
			allResults = append(allResults, result{
				key:      keys[i],
				distance: distances[i],
			})
		}
	}

	// Sort results by distance
	for i := 1; i < len(allResults); i++ {
		j := i
		for j > 0 && allResults[j-1].distance > allResults[j].distance {
			allResults[j-1], allResults[j] = allResults[j], allResults[j-1]
			j--
		}
	}

	// Deduplicate results (keep the closest instance of each key)
	seen := make(map[K]struct{})
	uniqueResults := make([]result, 0, len(allResults))
	for _, r := range allResults {
		if _, exists := seen[r.key]; !exists {
			seen[r.key] = struct{}{}
			uniqueResults = append(uniqueResults, r)
		}
	}

	// Return the k nearest neighbors
	resultCount := k
	if resultCount > len(uniqueResults) {
		resultCount = len(uniqueResults)
	}

	keys := make([]K, resultCount)
	distances := make([]float32, resultCount)
	for i := 0; i < resultCount; i++ {
		keys[i] = uniqueResults[i].key
		distances[i] = uniqueResults[i].distance
	}

	return keys, distances
}

// Delete removes a vector from all indexes.
func (a *MultiIndexAdapter[K]) Delete(key K) bool {
	success := true
	for _, idx := range a.indexes {
		if !idx.Delete(key) {
			success = false
		}
	}
	return success
}

// BatchDelete removes multiple vectors from all indexes.
func (a *MultiIndexAdapter[K]) BatchDelete(keys []K) []bool {
	results := make([]bool, len(keys))
	for i := range results {
		results[i] = true
	}

	for _, idx := range a.indexes {
		idxResults := idx.BatchDelete(keys)
		for i, success := range idxResults {
			if !success {
				results[i] = false
			}
		}
	}

	return results
}

// Len returns the number of vectors in the first index.
// Note: This assumes all indexes have the same content.
func (a *MultiIndexAdapter[K]) Len() int {
	if len(a.indexes) == 0 {
		return 0
	}
	return a.indexes[0].Len()
}

// Close releases resources used by all indexes.
func (a *MultiIndexAdapter[K]) Close() {
	for _, idx := range a.indexes {
		idx.Close()
	}
}
