# HNSW - Hierarchical Navigable Small World Graphs in Go

> This library is a fork of the original implementation by [Coder](https://github.com/coder/hnsw), enhanced with additional features, optimizations, and extensions. We acknowledge and thank the original authors for their excellent work.

## Overview

Package `hnsw` implements Hierarchical Navigable Small World graphs in Go, providing an efficient solution for approximate nearest neighbor search in high-dimensional vector spaces. HNSW graphs enable fast similarity search operations with logarithmic complexity, making them ideal for applications like semantic search, recommendation systems, and image similarity.

| Operation | Complexity            | Description                                  |
| --------- | --------------------- | -------------------------------------------- |
| Insert    | $O(log(n))$           | Insert a vector into the graph               |
| Delete    | $O(M^2 \cdot log(n))$ | Delete a vector from the graph               |
| Search    | $O(log(n))$           | Search for the nearest neighbors of a vector |
| Lookup    | $O(1)$                | Retrieve a vector by ID                      |

The library also includes extensions for metadata storage, faceted search, and other advanced features.

## Hybrid Strategy

We introduce a hybrid strategy in this library that combines multiple search techniques, including Hierarchical Navigable Small World (HNSW) graphs, exact search, and Locality-Sensitive Hashing (LSH), to optimize performance across diverse scenarios. This approach dynamically selects the most suitable method based on dataset characteristics, ensuring efficient and accurate approximate nearest neighbor search even in challenging conditions such as high-dimensional data and varying dataset sizes. By leveraging the strengths of each technique, the hybrid strategy provides a solution for applications requiring high performance and adaptability.

## Full Implementation

To see the library fully exercised, see [Quiver](https://github.com/TFMV/quiver)

## Installation

```text
go get github.com/TFMV/hnsw@main
```

## Basic Usage

```go
// Create a new graph with default parameters
g := hnsw.NewGraph[int]()

// Add some vectors
g.Add(
    hnsw.MakeNode(1, []float32{1, 1, 1}),
    hnsw.MakeNode(2, []float32{1, -1, 0.999}),
    hnsw.MakeNode(3, []float32{1, 0, -0.5}),
)

// Search for the nearest neighbor
neighbors, err := g.Search(
    []float32{0.5, 0.5, 0.5},
    1,
)
if err != nil {
    log.Fatalf("failed to search graph: %v", err)
}
fmt.Printf("best match: %v\n", neighbors[0].Value)
```

## Advanced Distance Calculations

The library provides a flexible and type-safe approach to distance calculations through the `vectortypes` package. This allows you to create custom distance functions that work with any type of data.

### Using VectorDistance with Custom Types

```go
// Define a custom type
type Document struct {
    ID        string
    Embedding []float32
    Category  string
}

// Create a custom distance function
documentDistance := hnsw.NewVectorDistance(
    vectortypes.ContraMap[vectortypes.F32, Document]{
        Surface: vectortypes.CreateSurface(hnsw.ToVectorTypesDistanceFunc(hnsw.CosineDistance)),
        ContraMap: func(doc Document) vectortypes.F32 {
            return doc.Embedding
        },
    },
)

// Calculate distance between documents
doc1 := Document{ID: "doc1", Embedding: []float32{1.0, 0.0, 0.0}, Category: "A"}
doc2 := Document{ID: "doc2", Embedding: []float32{0.0, 1.0, 0.0}, Category: "B"}
distance := documentDistance.Distance(doc1, doc2)
```

### Creating a Weighted Distance Function

```go
// Define a custom type
type Product struct {
    ID        string
    Embedding []float32
    Category  string
    Price     float32
}

// Define a custom surface
type ProductSurface struct{}

// Implement the Distance method
func (s ProductSurface) Distance(a, b Product) float32 {
    // Vector similarity component (70% weight)
    vectorDist := hnsw.CosineDistance(a.Embedding, b.Embedding) * 0.7
    
    // Category matching component (20% weight)
    var categoryDist float32
    if a.Category == b.Category {
        categoryDist = 0.0
    } else {
        categoryDist = 1.0
    }
    categoryDist *= 0.2
    
    // Price similarity component (10% weight)
    priceDiff := a.Price - b.Price
    if priceDiff < 0 {
        priceDiff = -priceDiff
    }
    priceDist := (priceDiff / 1000.0) * 0.1
    if priceDist > 0.1 {
        priceDist = 0.1
    }
    
    return vectorDist + categoryDist + priceDist
}

// Create a custom distance calculator
productDistance := hnsw.NewVectorDistance[Product](ProductSurface{})
```

## Thread-Safe Operations

The library supports concurrent operations with a thread-safe implementation:

```go
// Create a thread-safe graph with custom parameters
g, err := hnsw.NewGraphWithConfig[int](16, 0.25, 20, hnsw.EuclideanDistance)
if err != nil {
    log.Fatalf("failed to create graph: %v", err)
}

// Perform concurrent operations
var wg sync.WaitGroup
numOperations := 10

// Concurrent searches
wg.Add(numOperations)
for i := 0; i < numOperations; i++ {
    go func(i int) {
        defer wg.Done()
        query := []float32{float32(i) * 0.1, float32(i) * 0.1, float32(i) * 0.1}
        results, err := g.Search(query, 1)
        if err != nil {
            log.Printf("Search error: %v", err)
            return
        }
        fmt.Printf("Search %d found: %v\n", i, results[0].Key)
    }(i)
}

wg.Wait()
```

## Persistence

The library provides facilities for saving and loading graphs from persistent storage:

```go
path := "my_graph.hnsw"
g1, err := LoadSavedGraph[int](path)
if err != nil {
    panic(err)
}

// Insert some vectors
for i := 0; i < 128; i++ {
    g1.Add(hnsw.MakeNode(i, []float32{float32(i)}))
}

// Save to disk
err = g1.Save()
if err != nil {
    panic(err)
}

// Later...
// g2 is a copy of g1
g2, err := LoadSavedGraph[int](path)
if err != nil {
    panic(err)
}
```

## Advanced Features

### Batch Operations

For high-throughput scenarios, batch operations reduce lock contention:

```go
// Add a batch of nodes in a single operation
batch := make([]hnsw.Node[int], 5)
for i := range batch {
    nodeID := 100 + i
    vector := []float32{float32(i) * 0.5, float32(i) * 0.5, float32(i) * 0.5}
    batch[i] = hnsw.MakeNode(nodeID, vector)
}
g.BatchAdd(batch)

// Perform multiple searches in a single operation
queries := [][]float32{
    {0.1, 0.1, 0.1},
    {0.2, 0.2, 0.2},
    {0.3, 0.3, 0.3},
}
batchResults, _ := g.BatchSearch(queries, 2)

// Delete multiple nodes in a single operation
keysToDelete := []int{100, 101, 102}
deleteResults := g.BatchDelete(keysToDelete)
```

### Negative Examples

Find vectors similar to your query but dissimilar to specified negative examples:

```go
// Search with a single negative example
dogQuery := []float32{1.0, 0.2, 0.1, 0.0}      // dog query
puppyNegative := []float32{0.9, 0.3, 0.2, 0.1} // puppy (negative example)

// Find dog-related concepts but not puppies (negativeWeight = 0.5)
results, err := g.SearchWithNegative(dogQuery, puppyNegative, 3, 0.5)

// Search with multiple negative examples
petQuery := []float32{0.3, 0.3, 0.3, 0.3}      // general pet query
negatives := []hnsw.Vector{dogNegative, catNegative}

// Find pet-related concepts but not dogs or cats (negativeWeight = 0.7)
results, err = g.SearchWithNegatives(petQuery, negatives, 3, 0.7)
```

### Quality Metrics

Evaluate graph structure and performance:

```go
analyzer := hnsw.Analyzer[int]{Graph: graph}
metrics := analyzer.QualityMetrics()

fmt.Printf("Node count: %d\n", metrics.NodeCount)
fmt.Printf("Average connectivity: %.2f\n", metrics.AvgConnectivity)
fmt.Printf("Layer balance: %.2f\n", metrics.LayerBalance)
fmt.Printf("Distortion ratio: %.2f\n", metrics.DistortionRatio)
```

## Extensions

The library includes several extensions for enhanced functionality:

### Metadata Extension

Store and retrieve JSON metadata alongside vectors:

```go
// Create a graph and metadata store
graph := hnsw.NewGraph[int]()
store := meta.NewMemoryMetadataStore[int]()
metadataGraph := meta.NewMetadataGraph(graph, store)

// Create a node with metadata
node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})
metadata := map[string]interface{}{
    "name":     "Product 1",
    "category": "Electronics",
    "price":    999.99,
    "tags":     []string{"smartphone", "5G", "camera"},
}

// Add the node with metadata
metadataNode, err := meta.NewMetadataNode(node, metadata)
if err != nil {
    log.Fatalf("Failed to create metadata node: %v", err)
}

err = metadataGraph.Add(metadataNode)
if err != nil {
    log.Fatalf("Failed to add node: %v", err)
}

// Search with metadata
query := []float32{0.1, 0.2, 0.3}
results, err := metadataGraph.Search(query, 10)
```

### Faceted Search Extension

Filter and aggregate search results based on facets:

```go
// Create a faceted graph
graph := hnsw.NewGraph[int]()
store := facets.NewMemoryFacetStore[int]()
facetedGraph := facets.NewFacetedGraph(graph, store)

// Add a node with facets
node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})
productFacets := []facets.Facet{
    facets.NewBasicFacet("category", "Electronics"),
    facets.NewBasicFacet("price", 999.99),
    facets.NewBasicFacet("brand", "TechCo"),
}

facetedNode := facets.NewFacetedNode(node, productFacets)
facetedGraph.Add(facetedNode)

// Search with facet filters
query := []float32{0.1, 0.2, 0.3}
priceFilter := facets.NewRangeFilter("price", 0, 1000)
categoryFilter := facets.NewEqualityFilter("category", "Electronics")

results, err := facetedGraph.Search(
    query,
    []facets.FacetFilter{priceFilter, categoryFilter},
    10,
)
```

## Performance Considerations

- **Distance Function Selection**: Choose the appropriate distance function for your data. Cosine distance is often used for text embeddings, while Euclidean distance is common for image features.
- **Parameter Tuning**: Adjust `M`, `EfConstruction`, and `EfSearch` parameters based on your dataset size and performance requirements.
- **Memory Usage**: The memory usage grows linearly with the number of vectors and their dimensionality.
- **Batch Operations**: Use batch operations for better performance when adding or removing multiple vectors.
- **Custom Distance Functions**: While custom distance functions provide flexibility, they may introduce overhead. Benchmark your implementation to ensure it meets your performance requirements.

## Benchmarks

The following benchmarks compare different distance calculation approaches:

```
BenchmarkDirectDistanceFunc-10          10068204               106.6 ns/op
BenchmarkBasicSurface-10                11110857               107.5 ns/op
BenchmarkContraMap-10                   10687495               113.1 ns/op
BenchmarkVectorDistance-10              10065565               139.6 ns/op
BenchmarkCustomSurface-10               11162084               126.6 ns/op
BenchmarkWeightedDistance-10            10984856               109.2 ns/op
BenchmarkHNSWNodeDistance-10             9822542               123.1 ns/op
```

## Examples

For complete examples, see the examples directory:

- `examples/basic/main.go`: Basic usage of the HNSW graph
- `examples/optimized_distance/main.go`: Demonstrates the optimized distance calculation approach
- `examples/custom_distance/main.go`: Shows how to create a custom distance function that combines vector similarity with metadata

## Contributing

Contributions to improve HNSW-Go are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
