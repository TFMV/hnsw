package arrow

import (
	"cmp"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
)

// ArrowGraph is a Hierarchical Navigable Small World graph that uses Arrow files for storage
type ArrowGraph[K cmp.Ordered] struct {
	// Configuration parameters
	M        int               // Maximum number of connections per node
	Ml       float64           // Level generation factor
	EfSearch int               // Size of dynamic candidate list during search
	Distance hnsw.DistanceFunc // Distance function
	Rng      *rand.Rand        // Random number generator

	// Storage
	storage     *ArrowStorage[K]
	vectorStore *VectorStore[K]

	// In-memory representation of the graph structure
	// This is a cache that can be rebuilt from Arrow files
	layers     []map[K]map[K]struct{} // layers[layer_id][node_key][neighbor_key]
	dimensions int
	nodeCount  int

	// Mutex for thread safety
	mu sync.RWMutex

	// Worker pool for parallel operations
	workerPool chan struct{}
}

// ArrowGraphConfig defines configuration options for the Arrow-based HNSW graph
type ArrowGraphConfig struct {
	M        int               // Maximum number of connections per node
	Ml       float64           // Level generation factor
	EfSearch int               // Size of dynamic candidate list during search
	Distance hnsw.DistanceFunc // Distance function

	// Storage-related fields
	StorageDir    string        // Directory for Arrow files
	NumWorkers    int           // Number of worker threads for batch operations
	BatchSize     int           // Batch size for processing
	FlushInterval time.Duration // How often to flush to disk

	// Memory management
	MemoryPoolSize int64 // Size of memory pool for Arrow allocator
	MemoryMap      bool  // Whether to use memory mapping for files

	// Advanced options
	EnableCompression bool   // Whether to enable compression for Arrow files
	CompressionType   string // Type of compression to use (e.g., "zstd", "lz4")
	CompressionLevel  int    // Compression level (1-9)
}

// MarshalJSON implements the json.Marshaler interface for ArrowGraphConfig
func (c ArrowGraphConfig) MarshalJSON() ([]byte, error) {
	// Create a serializable version without the function
	type SerializableConfig struct {
		M                 int     `json:"m"`
		Ml                float64 `json:"ml"`
		EfSearch          int     `json:"ef_search"`
		DistanceName      string  `json:"distance_name"`
		StorageDir        string  `json:"storage_dir"`
		NumWorkers        int     `json:"num_workers"`
		BatchSize         int     `json:"batch_size"`
		FlushInterval     int64   `json:"flush_interval_ns"`
		MemoryPoolSize    int64   `json:"memory_pool_size"`
		MemoryMap         bool    `json:"memory_map"`
		EnableCompression bool    `json:"enable_compression"`
		CompressionType   string  `json:"compression_type"`
		CompressionLevel  int     `json:"compression_level"`
	}

	// Determine distance function name by comparing function pointers
	distName := "cosine" // Default

	// Get the pointer to the distance function
	distPtr := fmt.Sprintf("%p", c.Distance)

	// Compare with known distance functions
	euclideanPtr := fmt.Sprintf("%p", hnsw.EuclideanDistance)
	cosinePtr := fmt.Sprintf("%p", hnsw.CosineDistance)

	switch distPtr {
	case euclideanPtr:
		distName = "euclidean"
	case cosinePtr:
		distName = "cosine"
		// Add other distance functions as needed
	}

	sc := SerializableConfig{
		M:                 c.M,
		Ml:                c.Ml,
		EfSearch:          c.EfSearch,
		DistanceName:      distName,
		StorageDir:        c.StorageDir,
		NumWorkers:        c.NumWorkers,
		BatchSize:         c.BatchSize,
		FlushInterval:     c.FlushInterval.Nanoseconds(),
		MemoryPoolSize:    c.MemoryPoolSize,
		MemoryMap:         c.MemoryMap,
		EnableCompression: c.EnableCompression,
		CompressionType:   c.CompressionType,
		CompressionLevel:  c.CompressionLevel,
	}

	return json.Marshal(sc)
}

// UnmarshalJSON implements the json.Unmarshaler interface for ArrowGraphConfig
func (c *ArrowGraphConfig) UnmarshalJSON(data []byte) error {
	// Parse the serializable version
	var sc struct {
		M                 int     `json:"m"`
		Ml                float64 `json:"ml"`
		EfSearch          int     `json:"ef_search"`
		DistanceName      string  `json:"distance_name"`
		StorageDir        string  `json:"storage_dir"`
		NumWorkers        int     `json:"num_workers"`
		BatchSize         int     `json:"batch_size"`
		FlushInterval     int64   `json:"flush_interval_ns"`
		MemoryPoolSize    int64   `json:"memory_pool_size"`
		MemoryMap         bool    `json:"memory_map"`
		EnableCompression bool    `json:"enable_compression"`
		CompressionType   string  `json:"compression_type"`
		CompressionLevel  int     `json:"compression_level"`
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
	c.MemoryPoolSize = sc.MemoryPoolSize
	c.MemoryMap = sc.MemoryMap
	c.EnableCompression = sc.EnableCompression
	c.CompressionType = sc.CompressionType
	c.CompressionLevel = sc.CompressionLevel

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

// DefaultArrowGraphConfig returns the default configuration for the Arrow-based HNSW graph
func DefaultArrowGraphConfig() ArrowGraphConfig {
	return ArrowGraphConfig{
		M:                 16,
		Ml:                0.4,
		EfSearch:          100,
		Distance:          hnsw.CosineDistance,
		StorageDir:        "arrow_data",
		NumWorkers:        4,
		BatchSize:         1000,
		FlushInterval:     5 * time.Minute,
		MemoryPoolSize:    1 << 30, // 1GB
		MemoryMap:         true,
		EnableCompression: false,
		CompressionType:   "zstd",
		CompressionLevel:  3,
	}
}

// searchCandidate represents a candidate node during search
type searchCandidate[K cmp.Ordered] struct {
	key      K
	vector   []float32
	distance float32
}

// NewArrowGraph creates a new HNSW graph with Arrow storage
func NewArrowGraph[K cmp.Ordered](config ArrowGraphConfig) (*ArrowGraph[K], error) {
	// Create storage configuration
	storageConfig := ArrowStorageConfig{
		Directory:      config.StorageDir,
		MemoryPoolSize: config.MemoryPoolSize,
		MemoryMap:      config.MemoryMap,
		BatchSize:      int64(config.BatchSize),
		NumWorkers:     config.NumWorkers,
	}

	// Create storage
	storage, err := NewArrowStorage[K](storageConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create storage: %w", err)
	}

	// Create random number generator with a fixed seed for reproducibility
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	graph := &ArrowGraph[K]{
		M:          config.M,
		Ml:         config.Ml,
		EfSearch:   config.EfSearch,
		Distance:   config.Distance,
		Rng:        rng,
		storage:    storage,
		layers:     make([]map[K]map[K]struct{}, 0),
		dimensions: 0,
		nodeCount:  0,
		workerPool: make(chan struct{}, config.NumWorkers),
	}

	// Initialize worker pool
	for i := 0; i < config.NumWorkers; i++ {
		graph.workerPool <- struct{}{}
	}

	return graph, nil
}

// Add adds a vector to the graph
func (g *ArrowGraph[K]) Add(key K, vector []float32) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check if vector already exists
	if g.vectorExists(key) {
		return fmt.Errorf("vector with key %v already exists", key)
	}

	// Initialize dimensions if this is the first vector
	if g.dimensions == 0 {
		g.dimensions = len(vector)
		g.vectorStore = NewVectorStore[K](g.storage, g.dimensions)
	} else if len(vector) != g.dimensions {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", g.dimensions, len(vector))
	}

	// Store vector
	if err := g.vectorStore.StoreVector(key, vector); err != nil {
		return fmt.Errorf("failed to store vector: %w", err)
	}

	// Generate random level for the new node
	level := g.randomLevel()

	// Ensure we have enough layers
	g.ensureLayers(level + 1)

	// Add node to all layers up to its level
	for l := 0; l <= level; l++ {
		g.layers[l][key] = make(map[K]struct{})
	}

	// If this is the first node, we're done
	if g.nodeCount == 0 {
		g.nodeCount++
		return nil
	}

	// Find entry point
	entryKey, entryVector := g.findEntryPoint()

	// Connect the new node to the graph
	currKey := entryKey
	currVector := entryVector

	// For each layer from top to bottom
	for l := len(g.layers) - 1; l >= 0; l-- {
		// If we're above the node's level, just find the best node to continue
		if l > level {
			candidates := g.searchLayer(currKey, currVector, vector, 1, g.EfSearch, l)
			if len(candidates) > 0 {
				currKey = candidates[0].key
				currVector = candidates[0].vector
			}
			continue
		}

		// Search for neighbors
		neighbors := g.searchLayer(currKey, currVector, vector, g.M, g.EfSearch, l)

		// Connect to neighbors
		for _, neighbor := range neighbors {
			g.connect(key, neighbor.key, l)
		}

		// Update current node for next layer
		if l > 0 {
			currKey = neighbors[0].key
			currVector = neighbors[0].vector
		}
	}

	g.nodeCount++
	return nil
}

// BatchAdd adds multiple vectors to the graph in a batch
func (g *ArrowGraph[K]) BatchAdd(keys []K, vectors [][]float32) []error {
	if len(keys) != len(vectors) {
		return []error{fmt.Errorf("number of keys (%d) does not match number of vectors (%d)", len(keys), len(vectors))}
	}

	errors := make([]error, len(keys))
	for i := range keys {
		errors[i] = g.Add(keys[i], vectors[i])
	}

	return errors
}

// Search finds the k nearest neighbors to the query vector
func (g *ArrowGraph[K]) Search(query []float32, k int) ([]hnsw.Node[K], error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.nodeCount == 0 {
		return nil, nil
	}

	if len(query) != g.dimensions {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", g.dimensions, len(query))
	}

	if k <= 0 {
		return nil, fmt.Errorf("k must be greater than 0")
	}

	// Find entry point
	entryKey, entryVector := g.findEntryPoint()

	// For each layer from top to bottom
	currKey := entryKey
	currVector := entryVector

	for l := len(g.layers) - 1; l >= 0; l-- {
		// Search layer
		candidates := g.searchLayer(currKey, currVector, query, 1, g.EfSearch, l)
		if len(candidates) > 0 {
			currKey = candidates[0].key
			currVector = candidates[0].vector
		}
	}

	// Final search at the bottom layer with higher ef
	candidates := g.searchLayer(currKey, currVector, query, k, g.EfSearch, 0)

	// Convert to Node format
	result := make([]hnsw.Node[K], 0, len(candidates))
	for _, candidate := range candidates {
		result = append(result, hnsw.Node[K]{
			Key:   candidate.key,
			Value: candidate.vector,
		})
	}

	return result, nil
}

// BatchSearch performs multiple searches in parallel
func (g *ArrowGraph[K]) BatchSearch(queries [][]float32, k int) ([][]hnsw.Node[K], error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.nodeCount == 0 {
		return make([][]hnsw.Node[K], len(queries)), nil
	}

	// Validate queries
	for i, query := range queries {
		if len(query) != g.dimensions {
			return nil, fmt.Errorf("query %d dimension mismatch: expected %d, got %d", i, g.dimensions, len(query))
		}
	}

	if k <= 0 {
		return nil, fmt.Errorf("k must be greater than 0")
	}

	// Create result slice
	results := make([][]hnsw.Node[K], len(queries))

	// Use a wait group to synchronize goroutines
	var wg sync.WaitGroup
	wg.Add(len(queries))

	// Process each query in parallel
	for i, query := range queries {
		go func(i int, query []float32) {
			defer wg.Done()

			// Acquire worker from pool
			<-g.workerPool
			defer func() { g.workerPool <- struct{}{} }()

			// Find entry point
			entryKey, entryVector := g.findEntryPoint()

			// For each layer from top to bottom
			currKey := entryKey
			currVector := entryVector

			for l := len(g.layers) - 1; l >= 0; l-- {
				// Search layer
				candidates := g.searchLayer(currKey, currVector, query, 1, g.EfSearch, l)
				if len(candidates) > 0 {
					currKey = candidates[0].key
					currVector = candidates[0].vector
				}
			}

			// Final search at the bottom layer with higher ef
			candidates := g.searchLayer(currKey, currVector, query, k, g.EfSearch, 0)

			// Convert to Node format
			result := make([]hnsw.Node[K], 0, len(candidates))
			for _, candidate := range candidates {
				result = append(result, hnsw.Node[K]{
					Key:   candidate.key,
					Value: candidate.vector,
				})
			}

			results[i] = result
		}(i, query)
	}

	// Wait for all searches to complete
	wg.Wait()

	return results, nil
}

// Delete removes a vector from the graph
func (g *ArrowGraph[K]) Delete(key K) bool {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check if vector exists
	if !g.vectorExists(key) {
		return false
	}

	// Remove from all layers
	for l := range g.layers {
		if _, ok := g.layers[l][key]; ok {
			// Remove connections to this node
			for neighborKey := range g.layers[l][key] {
				delete(g.layers[l][neighborKey], key)
			}
			// Remove the node itself
			delete(g.layers[l], key)
		}
	}

	// Remove from vector store
	g.vectorStore.DeleteVector(key)

	g.nodeCount--
	return true
}

// BatchDelete removes multiple vectors from the graph
func (g *ArrowGraph[K]) BatchDelete(keys []K) []bool {
	results := make([]bool, len(keys))
	for i, key := range keys {
		results[i] = g.Delete(key)
	}
	return results
}

// Save saves the graph to disk
func (g *ArrowGraph[K]) Save() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Flush vector store
	if g.vectorStore != nil {
		if err := g.vectorStore.Flush(); err != nil {
			return fmt.Errorf("failed to flush vector store: %w", err)
		}
	}

	// Save layers
	if err := g.saveLayers(); err != nil {
		return fmt.Errorf("failed to save layers: %w", err)
	}

	// Save neighbors
	if err := g.saveNeighbors(); err != nil {
		return fmt.Errorf("failed to save neighbors: %w", err)
	}

	return nil
}

// Load loads the graph from disk
func (g *ArrowGraph[K]) Load() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Load layers first
	if err := g.loadLayers(); err != nil {
		return fmt.Errorf("failed to load layers: %w", err)
	}

	// Initialize vector store with a default dimension (we'll update it later)
	// This ensures the vector store is available when loading neighbors
	if g.vectorStore == nil {
		// Use a default dimension of 128 initially
		defaultDim := 128
		g.vectorStore = NewVectorStore[K](g.storage, defaultDim)
	}

	// Now load neighbors
	if err := g.loadNeighbors(); err != nil {
		return fmt.Errorf("failed to load neighbors: %w", err)
	}

	// Update vector store dimension if needed
	if g.dimensions > 0 && g.dimensions != g.vectorStore.dims {
		// Create a new vector store with the correct dimensions
		g.vectorStore = NewVectorStore[K](g.storage, g.dimensions)
	}

	return nil
}

// Close releases resources used by the graph
func (g *ArrowGraph[K]) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	var errs []error

	// Close vector store first
	if g.vectorStore != nil {
		if err := g.vectorStore.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close vector store: %w", err))
			// Continue with other cleanup even if this fails
		}
	}

	// Close storage
	if g.storage != nil {
		if err := g.storage.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close storage: %w", err))
		}
	}

	// Clear in-memory data to help with garbage collection
	g.layers = nil

	// Return the first error if any occurred
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// Len returns the number of vectors in the graph
func (g *ArrowGraph[K]) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.nodeCount
}

// Dims returns the dimensionality of vectors in the graph
func (g *ArrowGraph[K]) Dims() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.dimensions
}

// Helper methods

// randomLevel generates a random level for a new node
func (g *ArrowGraph[K]) randomLevel() int {
	level := 0
	for g.Rng.Float64() < g.Ml && level < 32 {
		level++
	}
	return level
}

// ensureLayers ensures that the graph has at least n layers
func (g *ArrowGraph[K]) ensureLayers(n int) {
	for len(g.layers) < n {
		g.layers = append(g.layers, make(map[K]map[K]struct{}))
	}
}

// vectorExists checks if a vector with the given key exists
func (g *ArrowGraph[K]) vectorExists(key K) bool {
	// Check if key exists in any layer
	for _, layer := range g.layers {
		if _, ok := layer[key]; ok {
			return true
		}
	}
	return false
}

// findEntryPoint finds an entry point for search
func (g *ArrowGraph[K]) findEntryPoint() (K, []float32) {
	// Find a node in the top layer
	for key := range g.layers[len(g.layers)-1] {
		vector, err := g.vectorStore.GetVector(key)
		if err == nil {
			return key, vector
		}
	}

	// If top layer is empty, find a node in the bottom layer
	for key := range g.layers[0] {
		vector, err := g.vectorStore.GetVector(key)
		if err == nil {
			return key, vector
		}
	}

	// This should never happen if the graph is not empty
	var zero K
	return zero, nil
}

// connect connects two nodes in the given layer
func (g *ArrowGraph[K]) connect(key1, key2 K, layer int) {
	// Add bidirectional connection
	g.layers[layer][key1][key2] = struct{}{}
	g.layers[layer][key2][key1] = struct{}{}

	// Ensure max connections is not exceeded
	g.pruneConnections(key1, layer)
	g.pruneConnections(key2, layer)
}

// pruneConnections ensures a node doesn't have more than M connections
func (g *ArrowGraph[K]) pruneConnections(key K, layer int) {
	neighbors := g.layers[layer][key]
	if len(neighbors) <= g.M {
		return
	}

	// Get all neighbor vectors
	neighborVectors := make(map[K][]float32)
	for neighborKey := range neighbors {
		vector, err := g.vectorStore.GetVector(neighborKey)
		if err != nil {
			// If we can't get the vector, remove the connection
			delete(neighbors, neighborKey)
			continue
		}
		neighborVectors[neighborKey] = vector
	}

	// Get the current node's vector
	nodeVector, err := g.vectorStore.GetVector(key)
	if err != nil {
		// This should never happen
		return
	}

	// Calculate distances to all neighbors
	type neighborDist struct {
		key      K
		distance float32
	}
	distances := make([]neighborDist, 0, len(neighborVectors))
	for neighborKey, neighborVector := range neighborVectors {
		dist := g.Distance(nodeVector, neighborVector)
		distances = append(distances, neighborDist{
			key:      neighborKey,
			distance: dist,
		})
	}

	// Sort by distance (ascending)
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// Keep only the M closest neighbors
	newNeighbors := make(map[K]struct{}, g.M)
	for i := 0; i < g.M && i < len(distances); i++ {
		newNeighbors[distances[i].key] = struct{}{}
	}

	// Update connections
	g.layers[layer][key] = newNeighbors
}

// searchLayer searches for nearest neighbors in a specific layer
func (g *ArrowGraph[K]) searchLayer(entryKey K, entryVector, query []float32, k, ef int, layer int) []searchCandidate[K] {
	// Create visited set
	visited := make(map[K]bool)
	visited[entryKey] = true

	// Create candidate set (min heap by distance)
	candidates := make([]searchCandidate[K], 0, ef)
	candidates = append(candidates, searchCandidate[K]{
		key:      entryKey,
		vector:   entryVector,
		distance: g.Distance(entryVector, query),
	})

	// Create result set (max heap by distance)
	result := make([]searchCandidate[K], 0, ef)
	result = append(result, candidates[0])

	// Process candidates
	for len(candidates) > 0 {
		// Get closest candidate
		curr := candidates[0]
		candidates = candidates[1:]

		// If the farthest result is closer than the closest candidate, we're done
		if len(result) >= ef && result[len(result)-1].distance < curr.distance {
			break
		}

		// Get neighbors
		for neighborKey := range g.layers[layer][curr.key] {
			if visited[neighborKey] {
				continue
			}
			visited[neighborKey] = true

			// Get neighbor vector
			neighborVector, err := g.vectorStore.GetVector(neighborKey)
			if err != nil {
				continue
			}

			// Calculate distance
			dist := g.Distance(neighborVector, query)

			// If result set is not full or neighbor is closer than the farthest result
			if len(result) < ef || dist < result[len(result)-1].distance {
				// Add to candidates
				candidates = append(candidates, searchCandidate[K]{
					key:      neighborKey,
					vector:   neighborVector,
					distance: dist,
				})

				// Add to result
				result = append(result, searchCandidate[K]{
					key:      neighborKey,
					vector:   neighborVector,
					distance: dist,
				})

				// Sort candidates (min heap)
				sort.Slice(candidates, func(i, j int) bool {
					return candidates[i].distance < candidates[j].distance
				})

				// Sort result (max heap)
				sort.Slice(result, func(i, j int) bool {
					return result[i].distance < result[j].distance
				})

				// Trim result if needed
				if len(result) > ef {
					result = result[:ef]
				}
			}
		}
	}

	// Return k closest results
	if len(result) > k {
		result = result[:k]
	}
	return result
}

// saveLayers saves the layer structure to Arrow format
func (g *ArrowGraph[K]) saveLayers() error {
	// Create schema
	schema := g.storage.LayerSchema()

	// Create record builder
	recordBuilder := array.NewRecordBuilder(g.storage.alloc, schema)
	defer recordBuilder.Release()

	layerIdBuilder := recordBuilder.Field(0).(*array.Int32Builder)
	keyBuilder := recordBuilder.Field(1)

	// Add data to builders
	for layerId, layer := range g.layers {
		for key := range layer {
			layerIdBuilder.Append(int32(layerId))
			appendToBuilder(keyBuilder, key)
		}
	}

	// Create record
	record := recordBuilder.NewRecord()
	defer record.Release()

	// Create file
	file, err := os.Create(g.storage.layersFile)
	if err != nil {
		return fmt.Errorf("failed to create layers file: %w", err)
	}
	defer file.Close()

	// Write record
	writer, err := ipc.NewFileWriter(file, ipc.WithSchema(schema), ipc.WithAllocator(g.storage.alloc))
	if err != nil {
		return fmt.Errorf("failed to create Arrow writer: %w", err)
	}

	if err := writer.Write(record); err != nil {
		writer.Close()
		return fmt.Errorf("failed to write record: %w", err)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("failed to close Arrow writer: %w", err)
	}

	return nil
}

// saveNeighbors saves the neighbor connections to Arrow format
func (g *ArrowGraph[K]) saveNeighbors() error {
	// Create schema
	schema := g.storage.NeighborSchema()

	// Create record builder
	recordBuilder := array.NewRecordBuilder(g.storage.alloc, schema)
	defer recordBuilder.Release()

	layerIdBuilder := recordBuilder.Field(0).(*array.Int32Builder)
	keyBuilder := recordBuilder.Field(1)
	neighborKeyBuilder := recordBuilder.Field(2)

	// Add data to builders
	for layerId, layer := range g.layers {
		for key, neighbors := range layer {
			for neighborKey := range neighbors {
				layerIdBuilder.Append(int32(layerId))
				appendToBuilder(keyBuilder, key)
				appendToBuilder(neighborKeyBuilder, neighborKey)
			}
		}
	}

	// Create record
	record := recordBuilder.NewRecord()
	defer record.Release()

	// Create file
	file, err := os.Create(g.storage.neighborsFile)
	if err != nil {
		return fmt.Errorf("failed to create neighbors file: %w", err)
	}
	defer file.Close()

	// Write record
	writer, err := ipc.NewFileWriter(file, ipc.WithSchema(schema), ipc.WithAllocator(g.storage.alloc))
	if err != nil {
		return fmt.Errorf("failed to create Arrow writer: %w", err)
	}

	if err := writer.Write(record); err != nil {
		writer.Close()
		return fmt.Errorf("failed to write record: %w", err)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("failed to close Arrow writer: %w", err)
	}

	return nil
}

// loadLayers loads the layer structure from Arrow format
func (g *ArrowGraph[K]) loadLayers() error {
	// Check if file exists
	if _, err := os.Stat(g.storage.layersFile); os.IsNotExist(err) {
		// Initialize empty layers
		g.layers = make([]map[K]map[K]struct{}, 1)
		g.layers[0] = make(map[K]map[K]struct{})
		return nil
	}

	// Open Arrow file
	file, err := os.Open(g.storage.layersFile)
	if err != nil {
		return fmt.Errorf("failed to open layers file: %w", err)
	}
	defer file.Close()

	// Create Arrow reader
	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(g.storage.alloc))
	if err != nil {
		return fmt.Errorf("failed to create Arrow reader: %w", err)
	}
	defer reader.Close()

	// Initialize layers
	g.layers = make([]map[K]map[K]struct{}, 0)

	// Read all records
	for i := 0; i < reader.NumRecords(); i++ {
		record, err := reader.Record(i)
		if err != nil {
			return fmt.Errorf("failed to read record: %w", err)
		}
		defer record.Release()

		// Get columns
		layerIdCol := record.Column(0).(*array.Int32)
		keyCol := record.Column(1)

		// Process each row
		for j := 0; j < int(record.NumRows()); j++ {
			layerId := int(layerIdCol.Value(j))
			recordKey := GetArrayValue(keyCol, j)
			key, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			// Ensure we have enough layers
			g.ensureLayers(layerId + 1)

			// Add node to layer
			if g.layers[layerId][key] == nil {
				g.layers[layerId][key] = make(map[K]struct{})
			}
		}
	}

	// If no layers were loaded, initialize with one empty layer
	if len(g.layers) == 0 {
		g.layers = make([]map[K]map[K]struct{}, 1)
		g.layers[0] = make(map[K]map[K]struct{})
	}

	return nil
}

// loadNeighbors loads the neighbor connections from Arrow format
func (g *ArrowGraph[K]) loadNeighbors() error {
	// Check if file exists
	if _, err := os.Stat(g.storage.neighborsFile); os.IsNotExist(err) {
		return nil
	}

	// Open Arrow file
	file, err := os.Open(g.storage.neighborsFile)
	if err != nil {
		return fmt.Errorf("failed to open neighbors file: %w", err)
	}
	defer file.Close()

	// Create Arrow reader
	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(g.storage.alloc))
	if err != nil {
		return fmt.Errorf("failed to create Arrow reader: %w", err)
	}
	defer reader.Close()

	// Read all records
	for i := 0; i < reader.NumRecords(); i++ {
		record, err := reader.Record(i)
		if err != nil {
			return fmt.Errorf("failed to read record: %w", err)
		}
		defer record.Release()

		// Get columns
		layerCol := record.Column(0)
		keyCol := record.Column(1)
		neighborCol := record.Column(2)

		// Iterate through records
		for j := 0; j < int(record.NumRows()); j++ {
			layerID := int(layerCol.(*array.Int32).Value(j))
			keyValue := GetArrayValue(keyCol, j)
			neighborValue := GetArrayValue(neighborCol, j)

			key, err := convertArrowToKey[K](keyValue)
			if err != nil {
				continue
			}

			neighbor, err := convertArrowToKey[K](neighborValue)
			if err != nil {
				continue
			}

			// Ensure layer exists
			g.ensureLayers(layerID + 1)

			// Ensure node exists in layer
			if _, ok := g.layers[layerID][key]; !ok {
				g.layers[layerID][key] = make(map[K]struct{})
			}

			// Add neighbor
			g.layers[layerID][key][neighbor] = struct{}{}
		}
	}

	// Count unique nodes
	nodeSet := make(map[K]bool)
	for _, layer := range g.layers {
		for key := range layer {
			nodeSet[key] = true
		}
	}
	g.nodeCount = len(nodeSet)

	// Determine dimensions if not already set
	if g.dimensions == 0 && g.nodeCount > 0 && g.vectorStore != nil {
		// Try to get a vector to determine dimensions
		for key := range nodeSet {
			// Check if vector file exists first
			if _, err := os.Stat(g.storage.vectorsFile); os.IsNotExist(err) {
				// If vector file doesn't exist, use the default dimension
				g.dimensions = g.vectorStore.dims
				break
			}

			// Try to get the vector
			vector, err := g.vectorStore.GetVector(key)
			if err == nil && len(vector) > 0 {
				g.dimensions = len(vector)
				break
			}
		}

		// If we still couldn't determine dimensions, use the default
		if g.dimensions == 0 {
			g.dimensions = g.vectorStore.dims
		}
	}

	return nil
}
