Goals:

Optimize file I/O operations with memory mapping and record batching
Improve the position cache implementation for faster vector retrieval
Implement incremental writes instead of rewriting the entire file
Enhance concurrency handling with better lock management
Add better error handling and recovery mechanisms
Optimize memory usage with Arrow's zero-copy features

### 1. Implement Negative Example Support

// In the arrow.ArrowIndex struct
func (a *ArrowIndex[K]) SearchWithNegative(query, negative []float32, k int, negWeight float32) ([]SearchResult[K], error) {
    results, err := a.Search(query, k*2) // Get more candidates than needed
    if err != nil {
        return nil, err
    }

    // Rerank based on combined distance
    for i := range results {
        negDist := a.Distance(results[i].Vector, negative)
        results[i].Distance = results[i].Distance - (negWeight * negDist)
    }

    // Sort by adjusted distance and return top k
    sort.Slice(results, func(i, j int) bool {
        return results[i].Distance < results[j].Distance
    })
    
    if len(results) > k {
        results = results[:k]
    }
    
    return results, nil
}

// Also implement SearchWithNegatives for multiple negative examples
func (a *ArrowIndex[K]) SearchWithNegatives(query []float32, negatives [][]float32, k int, negWeight float32) ([]SearchResult[K], error) {
    // Similar to above but handle multiple negative examples
}

```

### 2. Add Storage Configuration Support

The Arrow index should support proper storage configuration to align with Quiver's architecture:

```go
// Add these fields to ArrowGraphConfig
type ArrowGraphConfig struct {
    // Existing fields
    M        int
    Ml       float64
    EfSearch int
    Distance hnsw.DistanceFunc
    
    // New storage-related fields
    StorageDir    string        // Directory for Arrow files
    NumWorkers    int           // Number of worker threads for batch operations
    BatchSize     int           // Batch size for processing
    FlushInterval time.Duration // How often to flush to disk
}

// Update DefaultArrowGraphConfig to include these fields
func DefaultArrowGraphConfig() ArrowGraphConfig {
    return ArrowGraphConfig{
        M:             16,
        Ml:            0.4,
        EfSearch:      100,
        Distance:      hnsw.CosineDistance,
        StorageDir:    "arrow_data",
        NumWorkers:    4,
        BatchSize:     1000,
        FlushInterval: 5 * time.Minute,
    }
}
```

### 3. Implement Serialization Support

To fix the serialization issues with distance functions and enable proper backup/restore:

```go
// Add these methods to ArrowGraphConfig
func (c ArrowGraphConfig) MarshalJSON() ([]byte, error) {
    // Create a serializable version without the function
    type SerializableConfig struct {
        M             int    `json:"m"`
        Ml            float64 `json:"ml"`
        EfSearch      int    `json:"ef_search"`
        DistanceName  string `json:"distance_name"`
        StorageDir    string `json:"storage_dir"`
        NumWorkers    int    `json:"num_workers"`
        BatchSize     int    `json:"batch_size"`
        FlushInterval int64  `json:"flush_interval_ns"`
    }
    
    // Determine distance function name
    distName := "cosine"
    switch c.Distance {
    case hnsw.EuclideanDistance:
        distName = "euclidean"
    case hnsw.CosineDistance:
        distName = "cosine"
    // Add other distance functions as needed
    }
    
    sc := SerializableConfig{
        M:             c.M,
        Ml:            c.Ml,
        EfSearch:      c.EfSearch,
        DistanceName:  distName,
        StorageDir:    c.StorageDir,
        NumWorkers:    c.NumWorkers,
        BatchSize:     c.BatchSize,
        FlushInterval: c.FlushInterval.Nanoseconds(),
    }
    
    return json.Marshal(sc)
}

func (c *ArrowGraphConfig) UnmarshalJSON(data []byte) error {
    // Parse the serializable version
    var sc struct {
        M             int    `json:"m"`
        Ml            float64 `json:"ml"`
        EfSearch      int    `json:"ef_search"`
        DistanceName  string `json:"distance_name"`
        StorageDir    string `json:"storage_dir"`
        NumWorkers    int    `json:"num_workers"`
        BatchSize     int    `json:"batch_size"`
        FlushInterval int64  `json:"flush_interval_ns"`
    }
    
    if err := json.Unmarshal(data, &sc); err != nil {
        return err
    }
    
    // Set the fields
    c.M = sc.M
    c.Ml = sc.Ml
    c.EfSearch = sc.EfSearch
    c.StorageDir = sc.StorageDir
    c.NumWorkers = sc.NumWorkers
    c.BatchSize = sc.BatchSize
    c.FlushInterval = time.Duration(sc.FlushInterval)
    
    // Set the distance function based on name
    switch sc.DistanceName {
    case "euclidean":
        c.Distance = hnsw.EuclideanDistance
    case "cosine":
        c.Distance = hnsw.CosineDistance
    // Add other distance functions as needed
    default:
        c.Distance = hnsw.CosineDistance // Default
    }
    
    return nil
}
```

### 4. Implement Proper Stats and Metrics

Add comprehensive statistics and metrics to the Arrow index:

```go
// Add a Stats method to ArrowIndex
func (a *ArrowIndex[K]) Stats() map[string]interface{} {
    stats := make(map[string]interface{})
    
    // Basic stats
    stats["num_vectors"] = a.Len()
    stats["m"] = a.config.M
    stats["ef_search"] = a.config.EfSearch
    
    // Arrow-specific stats
    stats["storage_dir"] = a.config.StorageDir
    stats["batch_size"] = a.config.BatchSize
    stats["num_workers"] = a.config.NumWorkers
    
    // Performance metrics
    stats["avg_search_time_ns"] = a.metrics.avgSearchTime.Nanoseconds()
    stats["avg_add_time_ns"] = a.metrics.avgAddTime.Nanoseconds()
    stats["total_searches"] = a.metrics.totalSearches
    stats["total_adds"] = a.metrics.totalAdds
    
    // Storage metrics
    stats["disk_usage_bytes"] = a.getDiskUsage()
    stats["memory_usage_bytes"] = a.getMemoryUsage()
    
    return stats
}
```

### 5. Implement Facet and Metadata Integration

Ensure the Arrow index properly integrates with Quiver's facet and metadata systems:

```go
// Add methods to ArrowIndex to handle facets and metadata directly
func (a *ArrowIndex[K]) AddWithFacets(key K, vector []float32, facets []facets.Facet) error {
    // Store facets in Arrow format along with vectors
}

func (a *ArrowIndex[K]) SearchWithFacetFilter(query []float32, k int, filter facets.FacetFilter) ([]SearchResult[K], error) {
    // Implement facet filtering directly in the Arrow index
}

func (a *ArrowIndex[K]) SearchWithMetadataFilter(query []float32, k int, filter []byte) ([]SearchResult[K], error) {
    // Implement metadata filtering directly in the Arrow index
}
```
