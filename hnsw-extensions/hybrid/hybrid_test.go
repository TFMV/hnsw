package hybrid

import (
	"math/rand"
	"testing"

	"github.com/TFMV/hnsw"
)

func TestHybridIndex(t *testing.T) {
	// Create a hybrid index with default configuration
	config := DefaultIndexConfig()
	config.Type = HybridIndexType
	index, err := NewHybridIndex[int](config)
	if err != nil {
		t.Fatalf("Error creating hybrid index: %v", err)
	}
	defer index.Close()

	// Generate some random vectors
	numVectors := 100
	dimension := 10
	vectors := generateRandomVectors(numVectors, dimension)

	// Add vectors to the index
	for i := 0; i < numVectors; i++ {
		if err := index.Add(i, vectors[i]); err != nil {
			t.Fatalf("Error adding vector %d: %v", i, err)
		}
	}

	// Verify the number of vectors
	if index.Len() != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, index.Len())
	}

	// Search for nearest neighbors
	queryVector := generateRandomVector(dimension)
	k := 5
	results, err := index.Search(queryVector, k)
	if err != nil {
		t.Fatalf("Error searching: %v", err)
	}

	// Verify the number of results
	if len(results) > k {
		t.Errorf("Expected at most %d results, got %d", k, len(results))
	}

	// Delete some vectors
	for i := 0; i < 10; i++ {
		keyToDelete := i
		success := index.Delete(keyToDelete)
		if !success {
			t.Errorf("Failed to delete key %d", keyToDelete)
		}
	}

	// Verify the number of vectors after deletion
	expectedCount := numVectors - 10
	if index.Len() != expectedCount {
		t.Errorf("Expected %d vectors after deletion, got %d", expectedCount, index.Len())
	}
}

func TestExactIndex(t *testing.T) {
	// Create an exact index
	index := NewExactIndex[int](hnsw.CosineDistance)

	// Generate some random vectors
	numVectors := 100
	dimension := 10
	vectors := generateRandomVectors(numVectors, dimension)

	// Add vectors to the index
	for i := 0; i < numVectors; i++ {
		if err := index.Add(i, vectors[i]); err != nil {
			t.Fatalf("Error adding vector %d: %v", i, err)
		}
	}

	// Verify the number of vectors
	if index.Len() != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, index.Len())
	}

	// Search for nearest neighbors
	queryVector := generateRandomVector(dimension)
	k := 5
	results, err := index.Search(queryVector, k)
	if err != nil {
		t.Fatalf("Error searching: %v", err)
	}

	// Verify the number of results
	if len(results) > k {
		t.Errorf("Expected at most %d results, got %d", k, len(results))
	}

	// Delete some vectors
	for i := 0; i < 10; i++ {
		keyToDelete := i
		success := index.Delete(keyToDelete)
		if !success {
			t.Errorf("Failed to delete key %d", keyToDelete)
		}
	}

	// Verify the number of vectors after deletion
	expectedCount := numVectors - 10
	if index.Len() != expectedCount {
		t.Errorf("Expected %d vectors after deletion, got %d", expectedCount, index.Len())
	}
}

func TestLSHIndex(t *testing.T) {
	// Create a new LSH index
	numHashTables := 4
	numHashBits := 8
	index := NewLSHIndex[int](numHashTables, numHashBits, hnsw.CosineDistance)
	defer index.Close()

	// Generate random vectors
	numVectors := 100
	dimension := 10
	vectors := generateRandomVectors(numVectors, dimension)

	// Add vectors to the index
	for i, vector := range vectors {
		err := index.Add(i, vector)
		if err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Verify the number of vectors
	if index.Len() != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, index.Len())
	}

	// Search for nearest neighbors
	queryVector := generateRandomVector(dimension)
	k := 5
	keys, _ := index.Search(queryVector, k)

	// Verify the number of results
	if len(keys) > k {
		t.Errorf("Expected at most %d results, got %d", k, len(keys))
	}

	// Delete some vectors
	deleted := index.Delete(0)
	if !deleted {
		t.Errorf("Failed to delete vector 0")
	}

	// Verify the number of vectors after deletion
	if index.Len() != numVectors-1 {
		t.Errorf("Expected %d vectors after deletion, got %d", numVectors-1, index.Len())
	}

	// Batch delete
	keysToDelete := []int{1, 2, 3}
	results := index.BatchDelete(keysToDelete)

	// Verify batch delete results
	for i, result := range results {
		if !result {
			t.Errorf("Failed to delete vector %d", keysToDelete[i])
		}
	}

	// Verify the number of vectors after batch deletion
	expectedCount := numVectors - 1 - len(keysToDelete)
	if index.Len() != expectedCount {
		t.Errorf("Expected %d vectors after batch deletion, got %d", expectedCount, index.Len())
	}
}

// generateRandomVectors generates a slice of random vectors.
func generateRandomVectors(count, dimension int) [][]float32 {
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = generateRandomVector(dimension)
	}
	return vectors
}

// generateRandomVector generates a random unit vector.
func generateRandomVector(dimension int) []float32 {
	vector := make([]float32, dimension)
	var sum float32

	// Generate random components
	for i := range vector {
		vector[i] = rand.Float32()*2 - 1
		sum += vector[i] * vector[i]
	}

	// Normalize to unit length
	norm := float32(1.0 / float64(sum))
	for i := range vector {
		vector[i] *= norm
	}

	return vector
}
