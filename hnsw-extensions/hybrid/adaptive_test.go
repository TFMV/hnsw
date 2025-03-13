package hybrid

import (
	"math/rand"
	"testing"
	"time"

	"github.com/TFMV/hnsw"
)

func TestAdaptiveHybridIndex(t *testing.T) {
	// Set up test parameters
	numVectors := 1000
	dimension := 64
	k := 10
	distFunc := hnsw.CosineDistance

	// Create underlying indexes
	exactIndex := NewExactIndex[int](distFunc)

	// Create HNSW graph
	hnswGraph, err := hnsw.NewGraphWithConfig[int](16, 0.25, 20, distFunc)
	if err != nil {
		t.Fatalf("Failed to create HNSW graph: %v", err)
	}

	// Create LSH index
	lshIndex := NewLSHIndex[int](4, 8, distFunc)

	// Create adaptive config
	adaptiveConfig := DefaultAdaptiveConfig()
	adaptiveConfig.InitialExactThreshold = 500
	adaptiveConfig.InitialDimThreshold = 100
	adaptiveConfig.ExplorationFactor = 0.1

	// Create adaptive hybrid index
	adaptiveIndex := NewAdaptiveHybridIndex(
		exactIndex,
		hnswGraph,
		lshIndex,
		distFunc,
		adaptiveConfig,
	)

	// Generate random vectors
	rand.Seed(42) // For reproducibility
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dimension)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()*2 - 1
		}
		// Normalize
		var sum float32
		for _, v := range vectors[i] {
			sum += v * v
		}
		norm := float32(1.0 / float64(sum))
		for j := range vectors[i] {
			vectors[i][j] *= norm
		}
	}

	// Add vectors to the index
	for i, vector := range vectors {
		if err := adaptiveIndex.Add(i, vector); err != nil {
			t.Fatalf("Error adding vector %d: %v", i, err)
		}
	}

	// Verify count
	if count := adaptiveIndex.Count(); count != numVectors {
		t.Errorf("Expected count %d, got %d", numVectors, count)
	}

	// Test search
	queryVector := vectors[rand.Intn(numVectors)]
	keys, distances, err := adaptiveIndex.Search(queryVector, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Verify search results
	if len(keys) != k {
		t.Errorf("Expected %d results, got %d", k, len(keys))
	}
	if len(distances) != k {
		t.Errorf("Expected %d distances, got %d", k, len(distances))
	}

	// Test deletion
	keyToDelete := keys[0]
	if err := adaptiveIndex.Delete(keyToDelete); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// Verify count after deletion
	if count := adaptiveIndex.Count(); count != numVectors-1 {
		t.Errorf("Expected count %d after deletion, got %d", numVectors-1, count)
	}
}

func TestAdaptiveStrategySelection(t *testing.T) {
	// Set up test parameters
	numVectors := 1000
	dimension := 64
	k := 10
	distFunc := hnsw.CosineDistance

	// Create underlying indexes
	exactIndex := NewExactIndex[int](distFunc)

	// Create HNSW graph
	hnswGraph, err := hnsw.NewGraphWithConfig[int](16, 0.25, 20, distFunc)
	if err != nil {
		t.Fatalf("Failed to create HNSW graph: %v", err)
	}

	// Create LSH index
	lshIndex := NewLSHIndex[int](4, 8, distFunc)

	// Create adaptive config with specific thresholds for testing
	adaptiveConfig := DefaultAdaptiveConfig()
	adaptiveConfig.InitialExactThreshold = 500 // Use exact for small datasets
	adaptiveConfig.InitialDimThreshold = 100   // Use LSH for high dimensions
	adaptiveConfig.ExplorationFactor = 0.0     // Disable exploration for deterministic testing

	// Create adaptive hybrid index
	adaptiveIndex := NewAdaptiveHybridIndex(
		exactIndex,
		hnswGraph,
		lshIndex,
		distFunc,
		adaptiveConfig,
	)

	// Generate random vectors
	rand.Seed(42) // For reproducibility
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dimension)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()*2 - 1
		}
		// Normalize
		var sum float32
		for _, v := range vectors[i] {
			sum += v * v
		}
		norm := float32(1.0 / float64(sum))
		for j := range vectors[i] {
			vectors[i][j] *= norm
		}
	}

	// Add vectors to the index
	for i, vector := range vectors {
		if err := adaptiveIndex.Add(i, vector); err != nil {
			t.Fatalf("Error adding vector %d: %v", i, err)
		}
	}

	// Test strategy selection based on dataset size
	// Since we have 1000 vectors and threshold is 500, it should use HNSW
	selector := adaptiveIndex.selector
	queryVector := vectors[0]
	strategy := selector.SelectStrategy(queryVector, k)
	if strategy != HNSWIndexType {
		t.Errorf("Expected strategy %s for normal query, got %s",
			indexTypeToString(HNSWIndexType), indexTypeToString(strategy))
	}

	// Test strategy selection for high-dimensional data
	highDimQuery := make([]float32, 200) // Higher than dimThreshold
	for i := range highDimQuery {
		highDimQuery[i] = rand.Float32()*2 - 1
	}
	// Normalize
	var sum float32
	for _, v := range highDimQuery {
		sum += v * v
	}
	norm := float32(1.0 / float64(sum))
	for i := range highDimQuery {
		highDimQuery[i] *= norm
	}

	strategy = selector.SelectStrategy(highDimQuery, k)
	if strategy != LSHIndexType {
		t.Errorf("Expected strategy %s for high-dimensional query, got %s",
			indexTypeToString(LSHIndexType), indexTypeToString(strategy))
	}

	// Test adaptive thresholds after queries
	// Run several queries to gather statistics
	for i := 0; i < 100; i++ {
		queryVector := vectors[rand.Intn(numVectors)]
		_, _, err := adaptiveIndex.Search(queryVector, k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}

	// Get stats and verify they're being collected
	stats := adaptiveIndex.GetStats()
	if strategies, ok := stats["strategies"].(map[string]interface{}); ok {
		for strategy, stratStats := range strategies {
			if ss, ok := stratStats.(map[string]interface{}); ok {
				if totalQueries, ok := ss["total_queries"].(int); ok {
					t.Logf("Strategy %s: %d queries", strategy, totalQueries)
				}
			}
		}
	}

	// Test reset stats
	adaptiveIndex.ResetStats()
	stats = adaptiveIndex.GetStats()
	if strategies, ok := stats["strategies"].(map[string]interface{}); ok {
		for strategy, stratStats := range strategies {
			if ss, ok := stratStats.(map[string]interface{}); ok {
				if totalQueries, ok := ss["total_queries"].(int); ok && totalQueries > 0 {
					t.Errorf("Expected 0 queries after reset for strategy %s, got %d",
						strategy, totalQueries)
				}
			}
		}
	}
}

func TestAdaptiveHybridWithDifferentDatasets(t *testing.T) {
	// Test with different dataset sizes
	testSizes := []int{100, 1000}

	for _, size := range testSizes {
		t.Run(t.Name()+"-Size"+string(rune(size)), func(t *testing.T) {
			dimension := 64
			k := 10
			distFunc := hnsw.CosineDistance

			// Create underlying indexes
			exactIndex := NewExactIndex[int](distFunc)

			// Create HNSW graph
			hnswGraph, err := hnsw.NewGraphWithConfig[int](16, 0.25, 20, distFunc)
			if err != nil {
				t.Fatalf("Failed to create HNSW graph: %v", err)
			}

			// Create LSH index
			lshIndex := NewLSHIndex[int](4, 8, distFunc)

			// Create adaptive config
			adaptiveConfig := DefaultAdaptiveConfig()
			adaptiveConfig.InitialExactThreshold = 500
			adaptiveConfig.InitialDimThreshold = 100
			adaptiveConfig.ExplorationFactor = 0.1

			// Create adaptive hybrid index
			adaptiveIndex := NewAdaptiveHybridIndex(
				exactIndex,
				hnswGraph,
				lshIndex,
				distFunc,
				adaptiveConfig,
			)

			// Generate random vectors
			rand.Seed(time.Now().UnixNano())
			vectors := make([][]float32, size)
			for i := range vectors {
				vectors[i] = make([]float32, dimension)
				for j := range vectors[i] {
					vectors[i][j] = rand.Float32()*2 - 1
				}
				// Normalize
				var sum float32
				for _, v := range vectors[i] {
					sum += v * v
				}
				norm := float32(1.0 / float64(sum))
				for j := range vectors[i] {
					vectors[i][j] *= norm
				}
			}

			// Add vectors to the index
			for i, vector := range vectors {
				if err := adaptiveIndex.Add(i, vector); err != nil {
					t.Fatalf("Error adding vector %d: %v", i, err)
				}
			}

			// Verify count
			if count := adaptiveIndex.Count(); count != size {
				t.Errorf("Expected count %d, got %d", size, count)
			}

			// Test search
			queryVector := vectors[rand.Intn(size)]
			keys, distances, err := adaptiveIndex.Search(queryVector, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Verify search results
			expectedK := min(k, size)
			if len(keys) != expectedK {
				t.Errorf("Expected %d results, got %d", expectedK, len(keys))
			}
			if len(distances) != expectedK {
				t.Errorf("Expected %d distances, got %d", expectedK, len(distances))
			}
		})
	}
}
