// Package vector provides optimized vector operations for the HNSW library.
package vector

import (
	"cmp"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/vectortypes"
)

// F32 is a type alias for []float32 to make it more expressive
type F32 = vectortypes.F32

// Surface is a type alias for vectortypes.Surface
type Surface[T any] = vectortypes.Surface[T]

// ContraMap is a type alias for vectortypes.ContraMap
type ContraMap[V, T any] = vectortypes.ContraMap[V, T]

// BasicSurface is a type alias for vectortypes.BasicSurface
type BasicSurface = vectortypes.BasicSurface

// CreateSurface creates a basic surface from an HNSW distance function
func CreateSurface(distFunc hnsw.DistanceFunc) Surface[F32] {
	return BasicSurface{DistFunc: func(a, b F32) float32 {
		return distFunc(a, b)
	}}
}

// NodeVec is a vector annotated with a key of type K
type NodeVec[K comparable] struct {
	Key K
	Vec F32
}

// CreateNodeSurface creates a surface for NodeVec types
func CreateNodeSurface[K comparable](distFunc hnsw.DistanceFunc) Surface[NodeVec[K]] {
	return ContraMap[F32, NodeVec[K]]{
		Surface:   CreateSurface(distFunc),
		ContraMap: func(n NodeVec[K]) F32 { return n.Vec },
	}
}

// HNSWNodeSurface creates a surface for hnsw.Node types
func HNSWNodeSurface[K cmp.Ordered](distFunc hnsw.DistanceFunc) Surface[hnsw.Node[K]] {
	return ContraMap[F32, hnsw.Node[K]]{
		Surface:   CreateSurface(distFunc),
		ContraMap: func(n hnsw.Node[K]) F32 { return n.Value },
	}
}
