package parquet

import (
	"cmp"
	"fmt"

	"github.com/TFMV/hnsw"
)

// HNSWAdapter adapts the ParquetGraph to the HNSW interface
type HNSWAdapter[K cmp.Ordered] struct {
	graph *ParquetGraph[K]
}

// NewHNSWAdapter creates a new adapter for the ParquetGraph
func NewHNSWAdapter[K cmp.Ordered](config ParquetGraphConfig) (*HNSWAdapter[K], error) {
	graph, err := NewParquetGraph[K](config)
	if err != nil {
		return nil, fmt.Errorf("failed to create ParquetGraph: %w", err)
	}

	return &HNSWAdapter[K]{
		graph: graph,
	}, nil
}

// Add adds a node to the graph
func (a *HNSWAdapter[K]) Add(node hnsw.Node[K]) {
	err := a.graph.Add(node)
	if err != nil {
		// Log error but continue
		fmt.Printf("Error adding node: %v\n", err)
	}
}

// BatchAdd adds multiple nodes to the graph
func (a *HNSWAdapter[K]) BatchAdd(nodes []hnsw.Node[K]) error {
	return a.graph.Add(nodes...)
}

// Search performs a k-nearest neighbor search
func (a *HNSWAdapter[K]) Search(query []float32, k int) ([]hnsw.Node[K], error) {
	return a.graph.Search(query, k)
}

// Delete removes a node from the graph
func (a *HNSWAdapter[K]) Delete(key K) bool {
	return a.graph.Delete(key)
}

// BatchDelete removes multiple nodes from the graph
func (a *HNSWAdapter[K]) BatchDelete(keys []K) []bool {
	return a.graph.BatchDelete(keys)
}

// Len returns the number of nodes in the graph
func (a *HNSWAdapter[K]) Len() int {
	return a.graph.Len()
}

// Close releases resources used by the graph
func (a *HNSWAdapter[K]) Close() error {
	return a.graph.Close()
}

// GetGraph returns the underlying ParquetGraph
func (a *HNSWAdapter[K]) GetGraph() *ParquetGraph[K] {
	return a.graph
}
