package hybrid

import (
	"cmp"
	"fmt"
	"math/rand"
	"sync"

	"github.com/TFMV/hnsw"
)

// LSHIndex implements a Locality-Sensitive Hashing index for approximate nearest neighbor search.
// This is particularly useful for high-dimensional data where HNSW might struggle.
type LSHIndex[K cmp.Ordered] struct {
	// Number of hash tables
	numTables int

	// Number of bits per hash
	numBits int

	// Random projection vectors for hashing
	projections [][]float32

	// Hash tables (table -> hash -> keys)
	tables []map[uint64][]K

	// Vector storage
	vectors map[K][]float32

	// Distance function
	distance hnsw.DistanceFunc

	// Mutex for thread safety
	mu sync.RWMutex
}

// NewLSHIndex creates a new LSH index.
func NewLSHIndex[K cmp.Ordered](numTables, numBits int, distance hnsw.DistanceFunc) *LSHIndex[K] {
	if numTables <= 0 {
		numTables = 4 // Default value
	}
	if numBits <= 0 {
		numBits = 8 // Default value
	}

	lsh := &LSHIndex[K]{
		numTables: numTables,
		numBits:   numBits,
		tables:    make([]map[uint64][]K, numTables),
		vectors:   make(map[K][]float32),
		distance:  distance,
	}

	// Initialize hash tables
	for i := range lsh.tables {
		lsh.tables[i] = make(map[uint64][]K)
	}

	return lsh
}

// initProjections initializes random projection vectors for hashing.
// This is called lazily when the first vector is added to determine the dimensionality.
func (idx *LSHIndex[K]) initProjections(dimensions int) {
	if idx.projections != nil {
		return
	}

	// Create random projections
	rng := rand.New(rand.NewSource(42)) // Fixed seed for reproducibility
	idx.projections = make([][]float32, idx.numTables*idx.numBits)

	for i := range idx.projections {
		// Create a random unit vector
		projection := make([]float32, dimensions)
		var norm float32

		for j := range projection {
			// Random values between -1 and 1
			projection[j] = float32(rng.Float64()*2 - 1)
			norm += projection[j] * projection[j]
		}

		// Normalize to unit length
		norm = float32(1.0 / float64(norm))
		for j := range projection {
			projection[j] *= norm
		}

		idx.projections[i] = projection
	}
}

// computeHash computes the LSH hash for a vector.
func (idx *LSHIndex[K]) computeHash(vector []float32, tableIdx int) uint64 {
	var hash uint64

	// Compute dot products with projection vectors
	for i := 0; i < idx.numBits; i++ {
		projIdx := tableIdx*idx.numBits + i
		projection := idx.projections[projIdx]

		// Compute dot product
		var dotProduct float32
		for j := range vector {
			dotProduct += vector[j] * projection[j]
		}

		// Set bit if dot product is positive
		if dotProduct > 0 {
			hash |= 1 << i
		}
	}

	return hash
}

// Add adds a vector to the index.
func (idx *LSHIndex[K]) Add(key K, vector []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Initialize projections if this is the first vector
	if idx.projections == nil {
		idx.initProjections(len(vector))
	}

	// Store a copy of the vector
	vectorCopy := make([]float32, len(vector))
	copy(vectorCopy, vector)
	idx.vectors[key] = vectorCopy

	// Add to hash tables
	for i := 0; i < idx.numTables; i++ {
		hash := idx.computeHash(vector, i)
		idx.tables[i][hash] = append(idx.tables[i][hash], key)
	}

	return nil
}

// BatchAdd adds multiple vectors to the index.
func (idx *LSHIndex[K]) BatchAdd(keys []K, vectors [][]float32) []error {
	if len(keys) != len(vectors) {
		return []error{fmt.Errorf("number of keys (%d) does not match number of vectors (%d)", len(keys), len(vectors))}
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Initialize projections if this is the first batch
	if idx.projections == nil && len(vectors) > 0 {
		idx.initProjections(len(vectors[0]))
	}

	errors := make([]error, len(keys))

	for i, key := range keys {
		// Store a copy of the vector
		vectorCopy := make([]float32, len(vectors[i]))
		copy(vectorCopy, vectors[i])
		idx.vectors[key] = vectorCopy

		// Add to hash tables
		for j := 0; j < idx.numTables; j++ {
			hash := idx.computeHash(vectors[i], j)
			idx.tables[j][hash] = append(idx.tables[j][hash], key)
		}
	}

	return errors
}

// GetCandidates returns candidate keys that might be close to the query vector.
func (idx *LSHIndex[K]) GetCandidates(query []float32) []K {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.projections == nil {
		return nil
	}

	// Use a map to deduplicate candidates
	candidates := make(map[K]struct{})

	// Get candidates from all hash tables
	for i := 0; i < idx.numTables; i++ {
		hash := idx.computeHash(query, i)
		for _, key := range idx.tables[i][hash] {
			candidates[key] = struct{}{}
		}
	}

	// Convert map to slice
	result := make([]K, 0, len(candidates))
	for key := range candidates {
		result = append(result, key)
	}

	return result
}

// Search finds the k nearest neighbors to the query vector.
func (idx *LSHIndex[K]) Search(query []float32, k int) ([]K, []float32) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.projections == nil || len(idx.vectors) == 0 {
		return nil, nil
	}

	// Get candidates
	candidates := idx.GetCandidates(query)

	// If no candidates, return empty result
	if len(candidates) == 0 {
		return nil, nil
	}

	// Calculate distances to candidates
	type distanceItem struct {
		key      K
		distance float32
	}

	items := make([]distanceItem, 0, len(candidates))
	for _, key := range candidates {
		vector, exists := idx.vectors[key]
		if exists {
			dist := idx.distance(query, vector)
			items = append(items, distanceItem{
				key:      key,
				distance: dist,
			})
		}
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

	keys := make([]K, resultCount)
	distances := make([]float32, resultCount)
	for i := 0; i < resultCount; i++ {
		keys[i] = items[i].key
		distances[i] = items[i].distance
	}

	return keys, distances
}

// Delete removes a vector from the index.
func (idx *LSHIndex[K]) Delete(key K) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	vector, exists := idx.vectors[key]
	if !exists {
		return false
	}

	// Remove from hash tables
	for i := 0; i < idx.numTables; i++ {
		hash := idx.computeHash(vector, i)

		// Find and remove the key
		keys := idx.tables[i][hash]
		for j, k := range keys {
			if k == key {
				// Remove by swapping with the last element and truncating
				keys[j] = keys[len(keys)-1]
				idx.tables[i][hash] = keys[:len(keys)-1]
				break
			}
		}

		// If the bucket is empty, remove it
		if len(idx.tables[i][hash]) == 0 {
			delete(idx.tables[i], hash)
		}
	}

	// Remove from vector storage
	delete(idx.vectors, key)

	return true
}

// BatchDelete removes multiple vectors from the index.
func (idx *LSHIndex[K]) BatchDelete(keys []K) []bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	results := make([]bool, len(keys))

	for i, key := range keys {
		vector, exists := idx.vectors[key]
		if !exists {
			continue
		}

		// Remove from hash tables
		for j := 0; j < idx.numTables; j++ {
			hash := idx.computeHash(vector, j)

			// Find and remove the key
			tableKeys := idx.tables[j][hash]
			for k, tableKey := range tableKeys {
				if tableKey == key {
					// Remove by swapping with the last element and truncating
					tableKeys[k] = tableKeys[len(tableKeys)-1]
					idx.tables[j][hash] = tableKeys[:len(tableKeys)-1]
					break
				}
			}

			// If the bucket is empty, remove it
			if len(idx.tables[j][hash]) == 0 {
				delete(idx.tables[j], hash)
			}
		}

		// Remove from vector storage
		delete(idx.vectors, key)
		results[i] = true
	}

	return results
}

// Len returns the number of vectors in the index.
func (idx *LSHIndex[K]) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return len(idx.vectors)
}

// Close releases resources used by the index.
func (idx *LSHIndex[K]) Close() {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.tables = nil
	idx.vectors = nil
	idx.projections = nil
}
