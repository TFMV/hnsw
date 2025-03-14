package hybrid

import (
	"cmp"
	"fmt"
	"sync"

	"github.com/TFMV/hnsw"
)

// ExactIndex implements a brute-force exact search index.
// This is used for small datasets where exact search is fast enough.
type ExactIndex[K cmp.Ordered] struct {
	vectors  map[K][]float32
	distance hnsw.DistanceFunc
	mu       sync.RWMutex
}

// NewExactIndex creates a new exact search index.
func NewExactIndex[K cmp.Ordered](distance hnsw.DistanceFunc) *ExactIndex[K] {
	return &ExactIndex[K]{
		vectors:  make(map[K][]float32),
		distance: distance,
	}
}

// Add adds a vector to the index.
func (idx *ExactIndex[K]) Add(key K, vector []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Store a copy of the vector to prevent external modifications
	vectorCopy := make([]float32, len(vector))
	copy(vectorCopy, vector)

	idx.vectors[key] = vectorCopy
	return nil
}

// BatchAdd adds multiple vectors to the index.
func (idx *ExactIndex[K]) BatchAdd(keys []K, vectors [][]float32) error {
	if len(keys) != len(vectors) {
		return fmt.Errorf("number of keys (%d) does not match number of vectors (%d)", len(keys), len(vectors))
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	for i, key := range keys {
		// Store a copy of the vector to prevent external modifications
		vectorCopy := make([]float32, len(vectors[i]))
		copy(vectorCopy, vectors[i])

		idx.vectors[key] = vectorCopy
	}

	return nil
}

// Search finds the k nearest neighbors to the query vector.
func (idx *ExactIndex[K]) Search(query []float32, k int) ([]hnsw.Node[K], error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.vectors) == 0 {
		return []hnsw.Node[K]{}, nil
	}

	// Calculate distances to all vectors
	type distanceItem struct {
		key      K
		vector   []float32
		distance float32
	}

	items := make([]distanceItem, 0, len(idx.vectors))
	for key, vector := range idx.vectors {
		dist := idx.distance(query, vector)
		items = append(items, distanceItem{
			key:      key,
			vector:   vector,
			distance: dist,
		})
	}

	// Sort by distance (ascending)
	// Using a simple insertion sort since k is typically small
	for i := 1; i < len(items); i++ {
		j := i
		for j > 0 && items[j-1].distance > items[j].distance {
			items[j-1], items[j] = items[j], items[j-1]
			j--
		}
	}

	// Return the k nearest neighbors
	resultCount := k
	if resultCount > len(items) {
		resultCount = len(items)
	}

	results := make([]hnsw.Node[K], resultCount)
	for i := 0; i < resultCount; i++ {
		results[i] = hnsw.Node[K]{
			Key:   items[i].key,
			Value: items[i].vector,
		}
	}

	return results, nil
}

// Delete removes a vector from the index.
func (idx *ExactIndex[K]) Delete(key K) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	_, exists := idx.vectors[key]
	if exists {
		delete(idx.vectors, key)
		return true
	}

	return false
}

// BatchDelete removes multiple vectors from the index.
func (idx *ExactIndex[K]) BatchDelete(keys []K) []bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	results := make([]bool, len(keys))

	for i, key := range keys {
		_, exists := idx.vectors[key]
		if exists {
			delete(idx.vectors, key)
			results[i] = true
		}
	}

	return results
}

// Len returns the number of vectors in the index.
func (idx *ExactIndex[K]) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return len(idx.vectors)
}

// Close releases resources used by the index.
func (idx *ExactIndex[K]) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.vectors = nil
	return nil
}
