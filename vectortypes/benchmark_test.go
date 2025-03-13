package vectortypes_test

import (
	"math/rand"
	"testing"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/vectortypes"
)

// generateRandomVector creates a random vector of the given dimension
func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

// normalizeVector normalizes a vector to unit length
func normalizeVector(vec []float32) []float32 {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	norm := float32(1.0 / float32(sum))
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = v * norm
	}
	return result
}

// generateRandomVectors creates a slice of random vectors
func generateRandomVectors(count, dim int) [][]float32 {
	vecs := make([][]float32, count)
	for i := range vecs {
		vecs[i] = normalizeVector(generateRandomVector(dim))
	}
	return vecs
}

// CustomVector is a test type that wraps a vector
type CustomVector struct {
	ID    int
	Value []float32
}

// CustomSurface implements the Surface interface for CustomVector
type CustomSurface struct{}

// Distance implements the Surface interface
func (s CustomSurface) Distance(a, b CustomVector) float32 {
	return hnsw.CosineDistance(a.Value, b.Value)
}

// WeightedSurface implements a weighted distance function
type WeightedSurface struct{}

// Distance implements the Surface interface with weighted components
func (s WeightedSurface) Distance(a, b CustomVector) float32 {
	// Vector similarity component (80% weight)
	vectorDist := hnsw.CosineDistance(a.Value, b.Value) * 0.8

	// ID similarity component (20% weight)
	// Simple example: if IDs are close, they're more similar
	idDiff := float32(abs(a.ID-b.ID)) / 1000.0
	if idDiff > 1.0 {
		idDiff = 1.0
	}
	idDist := idDiff * 0.2

	return vectorDist + idDist
}

// BenchmarkDirectDistanceFunc benchmarks direct calls to a distance function
func BenchmarkDirectDistanceFunc(b *testing.B) {
	const (
		vectorCount = 1000
		dimension   = 128
	)

	// Generate random vectors
	vectors := generateRandomVectors(vectorCount, dimension)
	query := normalizeVector(generateRandomVector(dimension))

	// Benchmark direct distance function calls
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % vectorCount
		_ = hnsw.CosineDistance(query, vectors[idx])
	}
}

// BenchmarkBasicSurface benchmarks the BasicSurface implementation
func BenchmarkBasicSurface(b *testing.B) {
	const (
		vectorCount = 1000
		dimension   = 128
	)

	// Generate random vectors
	vectors := generateRandomVectors(vectorCount, dimension)
	query := normalizeVector(generateRandomVector(dimension))

	// Create a basic surface
	surface := vectortypes.CreateSurface(func(a, b vectortypes.F32) float32 {
		return hnsw.CosineDistance(a, b)
	})

	// Benchmark surface distance calls
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % vectorCount
		_ = surface.Distance(query, vectors[idx])
	}
}

// BenchmarkContraMap benchmarks the ContraMap implementation
func BenchmarkContraMap(b *testing.B) {
	const (
		vectorCount = 1000
		dimension   = 128
	)

	// Generate random vectors and wrap them in CustomVector
	rawVectors := generateRandomVectors(vectorCount, dimension)
	vectors := make([]CustomVector, vectorCount)
	for i, vec := range rawVectors {
		vectors[i] = CustomVector{ID: i, Value: vec}
	}
	query := CustomVector{ID: -1, Value: normalizeVector(generateRandomVector(dimension))}

	// Create a ContraMap surface
	surface := vectortypes.ContraMap[vectortypes.F32, CustomVector]{
		Surface: vectortypes.CreateSurface(func(a, b vectortypes.F32) float32 {
			return hnsw.CosineDistance(a, b)
		}),
		ContraMap: func(cv CustomVector) vectortypes.F32 {
			return cv.Value
		},
	}

	// Benchmark ContraMap distance calls
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % vectorCount
		_ = surface.Distance(query, vectors[idx])
	}
}

// BenchmarkVectorDistance benchmarks the VectorDistance implementation
func BenchmarkVectorDistance(b *testing.B) {
	const (
		vectorCount = 1000
		dimension   = 128
	)

	// Generate random vectors and wrap them in CustomVector
	rawVectors := generateRandomVectors(vectorCount, dimension)
	vectors := make([]CustomVector, vectorCount)
	for i, vec := range rawVectors {
		vectors[i] = CustomVector{ID: i, Value: vec}
	}
	query := CustomVector{ID: -1, Value: normalizeVector(generateRandomVector(dimension))}

	// Create a VectorDistance
	surface := vectortypes.ContraMap[vectortypes.F32, CustomVector]{
		Surface: vectortypes.CreateSurface(func(a, b vectortypes.F32) float32 {
			return hnsw.CosineDistance(a, b)
		}),
		ContraMap: func(cv CustomVector) vectortypes.F32 {
			return cv.Value
		},
	}
	vectorDistance := hnsw.NewVectorDistance(surface)

	// Benchmark VectorDistance distance calls
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % vectorCount
		_ = vectorDistance.Distance(query, vectors[idx])
	}
}

// BenchmarkCustomSurface benchmarks a custom Surface implementation
func BenchmarkCustomSurface(b *testing.B) {
	const (
		vectorCount = 1000
		dimension   = 128
	)

	// Generate random vectors and wrap them in CustomVector
	rawVectors := generateRandomVectors(vectorCount, dimension)
	vectors := make([]CustomVector, vectorCount)
	for i, vec := range rawVectors {
		vectors[i] = CustomVector{ID: i, Value: vec}
	}
	query := CustomVector{ID: -1, Value: normalizeVector(generateRandomVector(dimension))}

	// Create a custom surface
	surface := CustomSurface{}

	// Benchmark custom surface distance calls
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % vectorCount
		_ = surface.Distance(query, vectors[idx])
	}
}

// BenchmarkWeightedDistance benchmarks a weighted distance function
func BenchmarkWeightedDistance(b *testing.B) {
	const (
		vectorCount = 1000
		dimension   = 128
	)

	// Generate random vectors and wrap them in CustomVector
	rawVectors := generateRandomVectors(vectorCount, dimension)
	vectors := make([]CustomVector, vectorCount)
	for i, vec := range rawVectors {
		vectors[i] = CustomVector{ID: i, Value: vec}
	}
	query := CustomVector{ID: -1, Value: normalizeVector(generateRandomVector(dimension))}

	// Create a weighted surface
	surface := WeightedSurface{}

	// Benchmark weighted distance calls
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % vectorCount
		_ = surface.Distance(query, vectors[idx])
	}
}

// BenchmarkHNSWNodeDistance benchmarks the NodeDistance implementation
func BenchmarkHNSWNodeDistance(b *testing.B) {
	const (
		vectorCount = 1000
		dimension   = 128
	)

	// Generate random vectors and create HNSW nodes
	rawVectors := generateRandomVectors(vectorCount, dimension)
	nodes := make([]hnsw.Node[int], vectorCount)
	for i, vec := range rawVectors {
		nodes[i] = hnsw.MakeNode(i, vec)
	}
	query := hnsw.MakeNode(-1, normalizeVector(generateRandomVector(dimension)))

	// Create a NodeDistance
	nodeDistance := hnsw.NewNodeDistance[int](hnsw.CosineDistance)

	// Benchmark NodeDistance distance calls
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % vectorCount
		_ = nodeDistance.Distance(query, nodes[idx])
	}
}

// Helper function to get absolute value of an int
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
