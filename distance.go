package hnsw

import (
	"cmp"
	"reflect"

	"github.com/TFMV/hnsw/vectortypes"
	"github.com/viterin/vek/vek32"
)

// DistanceFunc is a function that computes the distance between two vectors.
type DistanceFunc func(a, b []float32) float32

// CosineDistance computes the cosine distance between two vectors.
func CosineDistance(a, b []float32) float32 {
	return 1 - vek32.CosineSimilarity(a, b)
}

// EuclideanDistance computes the Euclidean distance between two vectors.
func EuclideanDistance(a, b []float32) float32 {
	// Use vek32's vectorized implementation
	return vek32.Distance(a, b)
}

var distanceFuncs = map[string]DistanceFunc{
	"euclidean": EuclideanDistance,
	"cosine":    CosineDistance,
}

func distanceFuncToName(fn DistanceFunc) (string, bool) {
	for name, f := range distanceFuncs {
		fnptr := reflect.ValueOf(fn).Pointer()
		fptr := reflect.ValueOf(f).Pointer()
		if fptr == fnptr {
			return name, true
		}
	}
	return "", false
}

// RegisterDistanceFunc registers a distance function with a name.
// A distance function must be registered here before a graph can be
// exported and imported.
func RegisterDistanceFunc(name string, fn DistanceFunc) {
	distanceFuncs[name] = fn
}

// ToVectorTypesDistanceFunc converts a standard DistanceFunc to a vectortypes.DistanceFunc
func ToVectorTypesDistanceFunc(fn DistanceFunc) vectortypes.DistanceFunc {
	return func(a, b vectortypes.F32) float32 {
		return fn(a, b)
	}
}

// CreateSurface creates a vectortypes.Surface from a DistanceFunc
func CreateSurface(fn DistanceFunc) vectortypes.Surface[vectortypes.F32] {
	return vectortypes.CreateSurface(ToVectorTypesDistanceFunc(fn))
}

// NodeSurface creates a vectortypes.Surface for Node types
func NodeSurface[K cmp.Ordered](fn DistanceFunc) vectortypes.Surface[Node[K]] {
	return vectortypes.ContraMap[vectortypes.F32, Node[K]]{
		Surface:   CreateSurface(fn),
		ContraMap: func(n Node[K]) vectortypes.F32 { return n.Value },
	}
}

// VectorDistance is a generic distance calculator that can work with any type
// that can be mapped to a vectortypes.F32
type VectorDistance[T any] struct {
	Surface vectortypes.Surface[T]
}

// NewVectorDistance creates a new VectorDistance with the given surface
func NewVectorDistance[T any](surface vectortypes.Surface[T]) *VectorDistance[T] {
	return &VectorDistance[T]{Surface: surface}
}

// Distance calculates the distance between two vectors
func (vd *VectorDistance[T]) Distance(a, b T) float32 {
	return vd.Surface.Distance(a, b)
}

// NewNodeDistance creates a VectorDistance for Node types
func NewNodeDistance[K cmp.Ordered](fn DistanceFunc) *VectorDistance[Node[K]] {
	return NewVectorDistance(NodeSurface[K](fn))
}
