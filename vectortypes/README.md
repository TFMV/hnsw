# Vector Types Package

This package provides a flexible and type-safe approach to vector distance calculations in the HNSW library. It introduces a set of interfaces and types that enable more sophisticated distance functions while maintaining type safety and performance.

## Key Features

- **Type-safe distance calculations**: All distance calculations are type-checked at compile time
- **Composable distance functions**: Distance functions can be composed and reused
- **Support for custom vector types**: Work with any type that can be mapped to a vector
- **Domain-specific distance functions**: Incorporate domain knowledge into distance calculations
- **Performance optimizations**: Reduces indirection and allows for specialized implementations

## Core Types

### F32

`F32` is a type alias for `[]float32` that represents a vector of 32-bit floating-point values.

```go
type F32 = []float32
```

### DistanceFunc

`DistanceFunc` is a function type that computes the distance between two vectors.

```go
type DistanceFunc func(a, b F32) float32
```

### Surface

`Surface` is an interface that represents a distance function between two values of the same type.

```go
type Surface[T any] interface {
    Distance(a, b T) float32
}
```

### ContraMap

`ContraMap` is a generic adapter that allows applying a distance function to a different type by first mapping that type to the vector type the distance function expects.

```go
type ContraMap[V, T any] struct {
    Surface   Surface[V]
    ContraMap func(T) V
}
```

### BasicSurface

`BasicSurface` wraps a standard distance function to implement the `Surface` interface.

```go
type BasicSurface struct {
    DistFunc DistanceFunc
}
```

## Usage Examples

### Basic Usage

```go
import (
    "github.com/TFMV/hnsw"
    "github.com/TFMV/hnsw/vectortypes"
)

// Create a basic surface from a distance function
distFunc := hnsw.CosineDistance
surface := vectortypes.CreateSurface(func(a, b vectortypes.F32) float32 {
    return distFunc(a, b)
})

// Calculate distance between two vectors
vec1 := []float32{1.0, 0.0, 0.0}
vec2 := []float32{0.0, 1.0, 0.0}
distance := surface.Distance(vec1, vec2) // Returns 1.0 (cosine distance)
```

### Working with Custom Types

```go
// Define a custom type
type Document struct {
    ID       string
    Embedding []float32
    Content   string
}

// Create a surface for Document type
documentSurface := vectortypes.ContraMap[vectortypes.F32, Document]{
    Surface: vectortypes.CreateSurface(func(a, b vectortypes.F32) float32 {
        return hnsw.CosineDistance(a, b)
    }),
    ContraMap: func(doc Document) vectortypes.F32 {
        return doc.Embedding
    },
}

// Calculate distance between two documents
doc1 := Document{ID: "doc1", Embedding: []float32{1.0, 0.0, 0.0}}
doc2 := Document{ID: "doc2", Embedding: []float32{0.0, 1.0, 0.0}}
distance := documentSurface.Distance(doc1, doc2) // Returns 1.0
```

### Creating a Custom Distance Function

```go
// Define a custom distance function that combines vector similarity with text similarity
type TextDocument struct {
    ID        string
    Embedding []float32
    Text      string
}

// Create a custom surface
customSurface := struct{ vectortypes.Surface[TextDocument] }{
    Distance: func(a, b TextDocument) float32 {
        // Vector similarity component (70% weight)
        vectorDist := hnsw.CosineDistance(a.Embedding, b.Embedding) * 0.7
        
        // Text similarity component (30% weight)
        // Simple example: if texts have the same length, they're more similar
        textLenDiff := float32(abs(len(a.Text) - len(b.Text))) / 1000.0
        if textLenDiff > 1.0 {
            textLenDiff = 1.0
        }
        textDist := textLenDiff * 0.3
        
        return vectorDist + textDist
    },
}

// Use with VectorDistance
distanceCalculator := hnsw.NewVectorDistance(customSurface)
distance := distanceCalculator.Distance(doc1, doc2)
```

## Integration with HNSW

The `vectortypes` package is designed to work seamlessly with the HNSW library. You can use it to create custom distance functions for your specific use case while still leveraging the performance benefits of the HNSW algorithm.

```go
// Create a VectorDistance for HNSW nodes
nodeDistance := hnsw.NewNodeDistance[string](hnsw.CosineDistance)

// Use it in your code
node1 := hnsw.MakeNode("node1", []float32{1.0, 0.0, 0.0})
node2 := hnsw.MakeNode("node2", []float32{0.0, 1.0, 0.0})
distance := nodeDistance.Distance(node1, node2)
```

## Performance Considerations

- The `ContraMap` approach adds a small overhead compared to direct function calls, but this is typically negligible in practice.
- For performance-critical applications, you can implement the `Surface` interface directly with specialized implementations.
- The type safety and flexibility benefits often outweigh the small performance cost.

## Best Practices

1. **Use the right abstraction level**: Choose between `DistanceFunc`, `Surface`, or `VectorDistance` based on your needs.
2. **Compose distance functions**: Use `ContraMap` to adapt existing distance functions to new types.
3. **Benchmark your implementation**: Always measure the performance impact of custom distance functions.
4. **Consider caching**: For expensive distance calculations, consider caching results.
5. **Normalize weights**: When combining multiple distance components, ensure the weights sum to 1.0 for predictable behavior.

## Examples

For complete examples of how to use the `vectortypes` package, see the examples directory:

- `examples/optimized_distance/main.go`: Demonstrates basic usage of the optimized distance calculations
- `examples/custom_distance/main.go`: Shows how to create a custom distance function that combines vector similarity with metadata

## Contributing

Contributions to improve the `vectortypes` package are welcome. Please feel free to submit issues or pull requests.
