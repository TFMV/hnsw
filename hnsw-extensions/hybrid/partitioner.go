package hybrid

import (
	"cmp"
	"math/rand"
	"sync"

	"github.com/TFMV/hnsw"
)

// Partitioner implements data-aware partitioning for the hybrid index.
// It assigns vectors to partitions based on their similarity.
type Partitioner[K cmp.Ordered] struct {
	// Number of partitions
	numPartitions int

	// Distance function
	distance hnsw.DistanceFunc

	// Partition centroids
	centroids [][]float32

	// Vectors in each partition
	partitions []map[K]struct{}

	// Mutex for thread safety
	mu sync.RWMutex
}

// NewPartitioner creates a new partitioner.
func NewPartitioner[K cmp.Ordered](numPartitions int, distance hnsw.DistanceFunc) *Partitioner[K] {
	if numPartitions <= 0 {
		numPartitions = 10 // Default value
	}

	partitioner := &Partitioner[K]{
		numPartitions: numPartitions,
		distance:      distance,
		partitions:    make([]map[K]struct{}, numPartitions),
	}

	// Initialize partitions
	for i := range partitioner.partitions {
		partitioner.partitions[i] = make(map[K]struct{})
	}

	return partitioner
}

// initCentroids initializes random centroids for partitioning.
// This is called lazily when the first vector is assigned to determine the dimensionality.
func (p *Partitioner[K]) initCentroids(dimensions int) {
	if p.centroids != nil {
		return
	}

	// Create random centroids
	rng := rand.New(rand.NewSource(42)) // Fixed seed for reproducibility
	p.centroids = make([][]float32, p.numPartitions)

	for i := range p.centroids {
		// Create a random unit vector
		centroid := make([]float32, dimensions)
		var sum float32

		for j := range centroid {
			// Random values between -1 and 1
			centroid[j] = float32(rng.Float64()*2 - 1)
			sum += centroid[j] * centroid[j]
		}

		// Normalize to unit length
		norm := float32(1.0 / float64(sum))
		for j := range centroid {
			centroid[j] *= norm
		}

		p.centroids[i] = centroid
	}
}

// AssignPartition assigns a vector to a partition and returns the partition index.
func (p *Partitioner[K]) AssignPartition(vector []float32) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Initialize centroids if this is the first vector
	if p.centroids == nil {
		p.initCentroids(len(vector))
	}

	// Find the closest centroid
	bestPartition := 0
	bestDistance := float32(1e10)

	for i, centroid := range p.centroids {
		dist := p.distance(vector, centroid)
		if dist < bestDistance {
			bestDistance = dist
			bestPartition = i
		}
	}

	return bestPartition
}

// AssignVectorToPartition assigns a vector with a key to a partition.
func (p *Partitioner[K]) AssignVectorToPartition(key K, vector []float32) int {
	partitionIdx := p.AssignPartition(vector)

	p.mu.Lock()
	defer p.mu.Unlock()

	// Add to partition
	p.partitions[partitionIdx][key] = struct{}{}

	return partitionIdx
}

// GetPartition returns all keys in a partition.
func (p *Partitioner[K]) GetPartition(partitionIdx int) []K {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if partitionIdx < 0 || partitionIdx >= p.numPartitions {
		return nil
	}

	// Convert map to slice
	keys := make([]K, 0, len(p.partitions[partitionIdx]))
	for key := range p.partitions[partitionIdx] {
		keys = append(keys, key)
	}

	return keys
}

// RemoveFromPartitions removes a key from all partitions.
func (p *Partitioner[K]) RemoveFromPartitions(key K) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for i := range p.partitions {
		delete(p.partitions[i], key)
	}
}

// GetPartitionStats returns statistics about the partitions.
func (p *Partitioner[K]) GetPartitionStats() []int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats := make([]int, p.numPartitions)
	for i, partition := range p.partitions {
		stats[i] = len(partition)
	}

	return stats
}

// UpdateCentroids recalculates centroids based on the current partition assignments.
// This is used to improve partitioning quality after adding many vectors.
func (p *Partitioner[K]) UpdateCentroids(vectors map[K][]float32) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(vectors) == 0 || p.centroids == nil {
		return
	}

	dimensions := len(p.centroids[0])
	newCentroids := make([][]float32, p.numPartitions)
	counts := make([]int, p.numPartitions)

	// Initialize new centroids
	for i := range newCentroids {
		newCentroids[i] = make([]float32, dimensions)
	}

	// Sum vectors in each partition
	for key, vector := range vectors {
		for partitionIdx, partition := range p.partitions {
			if _, exists := partition[key]; exists {
				for j := range vector {
					newCentroids[partitionIdx][j] += vector[j]
				}
				counts[partitionIdx]++
				break
			}
		}
	}

	// Calculate average (centroid)
	for i, count := range counts {
		if count > 0 {
			for j := range newCentroids[i] {
				newCentroids[i][j] /= float32(count)
			}
		} else {
			// Keep old centroid if partition is empty
			copy(newCentroids[i], p.centroids[i])
		}
	}

	p.centroids = newCentroids
}

// Rebalance reassigns vectors to partitions based on updated centroids.
// This is used to improve partitioning quality after updating centroids.
func (p *Partitioner[K]) Rebalance(vectors map[K][]float32) {
	// Clear current partitions
	p.mu.Lock()
	for i := range p.partitions {
		p.partitions[i] = make(map[K]struct{})
	}
	p.mu.Unlock()

	// Reassign vectors to partitions
	for key, vector := range vectors {
		p.AssignVectorToPartition(key, vector)
	}
}
