# Arrow Extension for HNSW

This extension provides a high-performance implementation of the Hierarchical Navigable Small World (HNSW) graph algorithm using Apache Arrow for efficient vector storage and retrieval.

## Features

- **Apache Arrow Storage**: Uses Apache Arrow's columnar format for efficient storage and retrieval of vectors and graph structure
- **Memory-Efficient**: Minimizes memory usage by storing vectors on disk and loading them on-demand
- **Parallel Operations**: Supports parallel search and batch operations for high throughput
- **Persistent Storage**: Automatically persists the index to disk, allowing for quick recovery and reuse
- **Configurable**: Flexible configuration options for tuning performance and storage requirements
- **Generic Keys**: Supports any ordered key type (string, int, etc.) for vector identification
- **Incremental Updates**: Efficiently handles incremental updates to the index
- **Zero-Copy Appender**: Stream Arrow record batches directly into the index with minimal copying

## Architecture

The Arrow extension consists of several key components:

1. **ArrowIndex**: High-level interface that implements the HNSW vector index
2. **ArrowGraph**: Core implementation of the HNSW graph algorithm with Arrow storage
3. **ArrowStorage**: Manages the storage of vectors and graph structure using Arrow files
4. **VectorStore**: Handles efficient storage and retrieval of vectors using Arrow columnar format
5. **ArrowAppender**: Streams Arrow record batches directly into the index with zero-copy operations

## Performance

Benchmark results on Apple M2 Pro:

```
BenchmarkArrowAdd-10                10000    428835 ns/op    187214 B/op    1484 allocs/op
BenchmarkArrowBatchAdd-10             100  39931697 ns/op  17541071 B/op  136566 allocs/op
BenchmarkArrowSearch-10              1606    636877 ns/op    210998 B/op    2019 allocs/op
BenchmarkArrowBatchSearch-10           57  19326813 ns/op  19981149 B/op  199574 allocs/op
BenchmarkArrowSave-10                 100  11427862 ns/op  15069133 B/op  427146 allocs/op
BenchmarkArrowLoad-10                 595   2040059 ns/op   3553920 B/op   39922 allocs/op
BenchmarkArrowAppenderSingle-10     10000    410228 ns/op    180517 B/op    1420 allocs/op
BenchmarkArrowAppenderBatch-10        100  42010736 ns/op  18061169 B/op  140220 allocs/op
BenchmarkArrowAppenderStream-10         6  389837236 ns/op 161608281 B/op 1310294 allocs/op
```

### Performance Analysis

- **Single Vector Add**: ~429μs per vector with moderate memory usage
- **Batch Vector Add**: More efficient for adding multiple vectors at once (~399μs per vector in batches of 100)
- **Search Performance**: ~637μs per search query, making it suitable for real-time applications
- **Batch Search**: Significantly more efficient for multiple queries (~339μs per query when searching in batches)
- **Save/Load**: Fast persistence with ~11ms to save and ~2ms to load a 10,000 vector index
- **Arrow Appender Single**: ~410μs per vector, slightly faster than direct Add with less memory usage
- **Arrow Appender Batch**: ~420μs per vector in batches of 100, comparable to direct BatchAdd
- **Arrow Appender Stream**: Processes large batches of vectors efficiently, but with higher memory usage due to Arrow record batch creation

These benchmarks were performed with 128-dimensional vectors and a 10,000 vector index. Performance may vary based on vector dimensions, index size, and hardware.

## Usage

### Creating an Index

```go
import (
    "github.com/TFMV/hnsw/hnsw-extensions/arrow"
)

// Create a new index with default configuration
config := arrow.DefaultArrowGraphConfig()
index, err := arrow.NewArrowIndex[string](config)
if err != nil {
    // Handle error
}
```

### Customizing Configuration

```go
// Customize configuration
config := arrow.ArrowGraphConfig{
    M:        20,             // Maximum number of connections per node
    Ml:       0.3,            // Level generation factor
    EfSearch: 50,             // Size of dynamic candidate list during search
    Distance: hnsw.L2Distance, // Distance function
    Storage: arrow.ArrowStorageConfig{
        StorageDir:    "/path/to/storage",
        NumWorkers:    4,
        BatchSize:     1000,
        FlushInterval: 5 * time.Minute,
    },
}

index, err := arrow.NewArrowIndex[int](config)
if err != nil {
    // Handle error
}
```

### Using the Arrow Appender

The Arrow Appender allows you to stream Arrow record batches directly into the index with minimal copying:

```go
// Create an Arrow appender
appenderConfig := arrow.DefaultAppenderConfig()
appender := arrow.NewArrowAppender[string](index, appenderConfig)

// Append a record batch
if err := appender.AppendBatch(recordBatch); err != nil {
    // Handle error
}

// Append a table
if err := appender.AppendTable(table); err != nil {
    // Handle error
}

// Stream records asynchronously
recordChan := make(chan arrow.Record, 10)
errChan := appender.StreamRecordsAsync(recordChan)

// Send records to the channel
for record := range sourceRecords {
    recordChan <- record
}
close(recordChan)

// Check for errors
if err := <-errChan; err != nil {
    // Handle error
}
```

The appender expects Arrow record batches with a specific schema:

```go
schema := arrow.NewSchema(
    []arrow.Field{
        {Name: "key", Type: arrow.BinaryTypes.String}, // Key field (can be any supported type)
        {Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)}, // Vector field (must be list of float32)
    },
    nil,
)
```

You can customize the field names using the appender configuration:

```go
appenderConfig := arrow.AppenderConfig{
    KeyField:    "id",        // Custom key field name
    VectorField: "embedding", // Custom vector field name
    BatchSize:   1000,        // Batch size for processing
}
```

### Adding Vectors

```go
// Add a single vector
err := index.Add("vector1", []float32{0.1, 0.2, 0.3, 0.4})
if err != nil {
    // Handle error
}

// Add multiple vectors in batch
keys := []string{"vector2", "vector3", "vector4"}
vectors := [][]float32{
    {0.2, 0.3, 0.4, 0.5},
    {0.3, 0.4, 0.5, 0.6},
    {0.4, 0.5, 0.6, 0.7},
}
errors := index.BatchAdd(keys, vectors)
for i, err := range errors {
    if err != nil {
        fmt.Printf("Error adding vector %s: %v\n", keys[i], err)
    }
}
```

### Searching

```go
// Search for nearest neighbors
query := []float32{0.1, 0.2, 0.3, 0.4}
results, err := index.Search(query, 10)
if err != nil {
    // Handle error
}

for _, result := range results {
    fmt.Printf("Key: %v, Distance: %f\n", result.Key, result.Distance)
}

// Batch search
queries := [][]float32{
    {0.1, 0.2, 0.3, 0.4},
    {0.2, 0.3, 0.4, 0.5},
}
batchResults, err := index.BatchSearch(queries, 5)
if err != nil {
    // Handle error
}
```

### Deleting Vectors

```go
// Delete a single vector
deleted := index.Delete("vector1")
if !deleted {
    fmt.Println("Vector not found")
}

// Delete multiple vectors
keys := []string{"vector2", "vector3"}
results := index.BatchDelete(keys)
for i, deleted := range results {
    if !deleted {
        fmt.Printf("Vector %s not found\n", keys[i])
    }
}
```

### Saving and Loading

```go
// Save the index to disk
err := index.Save()
if err != nil {
    // Handle error
}

// Load the index from disk
err = index.Load()
if err != nil {
    // Handle error
}
```

### Optimizing

```go
// Optimize the index for search performance
err := index.Optimize()
if err != nil {
    // Handle error
}
```

### Getting Statistics

```go
// Get statistics about the index
stats := index.Stats()
fmt.Printf("Number of vectors: %d\n", stats["num_vectors"])
fmt.Printf("Dimensions: %d\n", stats["dimensions"])
fmt.Printf("Average connections per node: %f\n", stats["avg_connections_per_node"])
```

### Cleanup

```go
// Close the index and release resources
err := index.Close()
if err != nil {
    // Handle error
}
```

## Performance Considerations

- **Storage Directory**: Choose a fast storage medium (SSD) for optimal performance
- **Batch Size**: Larger batch sizes improve throughput but increase memory usage
- **Number of Workers**: Set based on available CPU cores for parallel operations
- **Flush Interval**: Balance between write performance and durability
- **EfSearch Parameter**: Higher values improve search quality but reduce speed

## Future Improvements

- Distributed search across multiple nodes
- Compression of vector data for reduced storage requirements
- Support for filtering during search
- Integration with other Arrow-based tools and libraries
- Memory-mapped files for improved performance

## License

This extension is licensed under the same license as the main HNSW library.
