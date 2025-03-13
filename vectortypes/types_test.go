package vectortypes_test

import (
	"math"
	"testing"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/vectortypes"
)

// TestBasicSurface tests the BasicSurface implementation
func TestBasicSurface(t *testing.T) {
	// Create a basic surface with cosine distance
	surface := vectortypes.CreateSurface(func(a, b vectortypes.F32) float32 {
		return hnsw.CosineDistance(a, b)
	})

	// Test vectors
	vec1 := []float32{1.0, 0.0, 0.0}
	vec2 := []float32{0.0, 1.0, 0.0}
	vec3 := []float32{1.0, 0.0, 0.0} // Same as vec1

	// Test distances
	dist12 := surface.Distance(vec1, vec2)
	dist13 := surface.Distance(vec1, vec3)

	// Cosine distance between orthogonal vectors should be 1.0
	if math.Abs(float64(dist12-1.0)) > 1e-6 {
		t.Errorf("Expected distance between orthogonal vectors to be 1.0, got %f", dist12)
	}

	// Cosine distance between identical vectors should be 0.0
	if math.Abs(float64(dist13)) > 1e-6 {
		t.Errorf("Expected distance between identical vectors to be 0.0, got %f", dist13)
	}
}

// TestDocument is a test type for document vectors
type TestDocument struct {
	ID        string
	Embedding []float32
}

// TestContraMap tests the ContraMap implementation
func TestContraMap(t *testing.T) {
	// Create a ContraMap surface for TestDocument
	surface := vectortypes.ContraMap[vectortypes.F32, TestDocument]{
		Surface: vectortypes.CreateSurface(func(a, b vectortypes.F32) float32 {
			return hnsw.CosineDistance(a, b)
		}),
		ContraMap: func(doc TestDocument) vectortypes.F32 {
			return doc.Embedding
		},
	}

	// Test documents
	doc1 := TestDocument{ID: "doc1", Embedding: []float32{1.0, 0.0, 0.0}}
	doc2 := TestDocument{ID: "doc2", Embedding: []float32{0.0, 1.0, 0.0}}
	doc3 := TestDocument{ID: "doc3", Embedding: []float32{1.0, 0.0, 0.0}} // Same embedding as doc1

	// Test distances
	dist12 := surface.Distance(doc1, doc2)
	dist13 := surface.Distance(doc1, doc3)

	// Cosine distance between orthogonal vectors should be 1.0
	if math.Abs(float64(dist12-1.0)) > 1e-6 {
		t.Errorf("Expected distance between orthogonal documents to be 1.0, got %f", dist12)
	}

	// Cosine distance between identical vectors should be 0.0
	if math.Abs(float64(dist13)) > 1e-6 {
		t.Errorf("Expected distance between identical documents to be 0.0, got %f", dist13)
	}
}

// TestCustomSurface is a custom surface implementation for testing
type TestCustomSurface struct{}

// Distance implements the Surface interface for TestCustomSurface
func (s TestCustomSurface) Distance(a, b TestDocument) float32 {
	// Custom distance function that considers both vector similarity and ID similarity
	vectorDist := hnsw.CosineDistance(a.Embedding, b.Embedding)

	// If IDs are the same, reduce the distance
	if a.ID == b.ID {
		return vectorDist * 0.5
	}

	return vectorDist
}

// TestCustomSurfaceImpl tests a custom Surface implementation
func TestCustomSurfaceImpl(t *testing.T) {
	// Create a custom surface
	surface := TestCustomSurface{}

	// Test documents
	doc1 := TestDocument{ID: "doc1", Embedding: []float32{1.0, 0.0, 0.0}}
	doc2 := TestDocument{ID: "doc2", Embedding: []float32{0.0, 1.0, 0.0}}
	doc3 := TestDocument{ID: "doc1", Embedding: []float32{0.0, 1.0, 0.0}} // Same ID as doc1, but different embedding

	// Test distances
	dist12 := surface.Distance(doc1, doc2)
	dist13 := surface.Distance(doc1, doc3)

	// Cosine distance between orthogonal vectors should be 1.0
	if math.Abs(float64(dist12-1.0)) > 1e-6 {
		t.Errorf("Expected distance between orthogonal documents to be 1.0, got %f", dist12)
	}

	// Custom distance between doc1 and doc3 should be 0.5 (1.0 * 0.5)
	if math.Abs(float64(dist13-0.5)) > 1e-6 {
		t.Errorf("Expected custom distance to be 0.5, got %f", dist13)
	}
}

// TestVectorDistance tests the VectorDistance implementation
func TestVectorDistance(t *testing.T) {
	// Create a ContraMap surface for TestDocument
	surface := vectortypes.ContraMap[vectortypes.F32, TestDocument]{
		Surface: vectortypes.CreateSurface(func(a, b vectortypes.F32) float32 {
			return hnsw.CosineDistance(a, b)
		}),
		ContraMap: func(doc TestDocument) vectortypes.F32 {
			return doc.Embedding
		},
	}

	// Create a VectorDistance
	vectorDistance := hnsw.NewVectorDistance(surface)

	// Test documents
	doc1 := TestDocument{ID: "doc1", Embedding: []float32{1.0, 0.0, 0.0}}
	doc2 := TestDocument{ID: "doc2", Embedding: []float32{0.0, 1.0, 0.0}}

	// Test distance
	dist := vectorDistance.Distance(doc1, doc2)

	// Cosine distance between orthogonal vectors should be 1.0
	if math.Abs(float64(dist-1.0)) > 1e-6 {
		t.Errorf("Expected distance between orthogonal documents to be 1.0, got %f", dist)
	}
}

// TestNodeDistance tests the NodeDistance implementation
func TestNodeDistance(t *testing.T) {
	// Create a NodeDistance
	nodeDistance := hnsw.NewNodeDistance[string](hnsw.CosineDistance)

	// Create test nodes
	node1 := hnsw.MakeNode("node1", []float32{1.0, 0.0, 0.0})
	node2 := hnsw.MakeNode("node2", []float32{0.0, 1.0, 0.0})
	node3 := hnsw.MakeNode("node3", []float32{1.0, 0.0, 0.0}) // Same vector as node1

	// Test distances
	dist12 := nodeDistance.Distance(node1, node2)
	dist13 := nodeDistance.Distance(node1, node3)

	// Cosine distance between orthogonal vectors should be 1.0
	if math.Abs(float64(dist12-1.0)) > 1e-6 {
		t.Errorf("Expected distance between orthogonal nodes to be 1.0, got %f", dist12)
	}

	// Cosine distance between identical vectors should be 0.0
	if math.Abs(float64(dist13)) > 1e-6 {
		t.Errorf("Expected distance between identical nodes to be 0.0, got %f", dist13)
	}
}
