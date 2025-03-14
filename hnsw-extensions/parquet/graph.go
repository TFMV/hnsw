package parquet

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"container/heap"

	"github.com/TFMV/hnsw"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

// ParquetGraph is a Hierarchical Navigable Small World graph that uses Parquet files for storage
type ParquetGraph[K cmp.Ordered] struct {
	// Configuration parameters
	M        int               // Maximum number of connections per node
	Ml       float64           // Level generation factor
	EfSearch int               // Size of dynamic candidate list during search
	Distance hnsw.DistanceFunc // Distance function
	Rng      *rand.Rand        // Random number generator

	// Storage
	storage     *ParquetStorage[K]
	vectorStore *VectorStore[K]

	// In-memory representation of the graph structure
	// This is a cache that can be rebuilt from Parquet files
	layers     []map[K]map[K]struct{} // layers[layer_id][node_key][neighbor_key]
	dimensions int
	nodeCount  int

	// Mutex for thread safety
	mu sync.RWMutex
}

// ParquetGraphConfig defines configuration options for the Parquet-based HNSW graph
type ParquetGraphConfig struct {
	M        int               // Maximum number of connections per node
	Ml       float64           // Level generation factor
	EfSearch int               // Size of dynamic candidate list during search
	Distance hnsw.DistanceFunc // Distance function
	Storage  ParquetStorageConfig

	// Incremental update configuration
	Incremental IncrementalConfig
}

// DefaultParquetGraphConfig returns the default configuration
func DefaultParquetGraphConfig() ParquetGraphConfig {
	return ParquetGraphConfig{
		M:           16,
		Ml:          0.25,
		EfSearch:    20,
		Distance:    hnsw.CosineDistance,
		Storage:     DefaultParquetStorageConfig(),
		Incremental: DefaultIncrementalConfig(),
	}
}

// searchCandidate represents a candidate node during search
type searchCandidate[K cmp.Ordered] struct {
	key  K
	dist float32
}

// candidateMaxHeap is a max heap of search candidates (for results)
type candidateMaxHeap[K cmp.Ordered] []*searchCandidate[K]

func (h candidateMaxHeap[K]) Len() int           { return len(h) }
func (h candidateMaxHeap[K]) Less(i, j int) bool { return h[i].dist > h[j].dist } // Max heap
func (h candidateMaxHeap[K]) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *candidateMaxHeap[K]) Push(x interface{}) {
	*h = append(*h, x.(*searchCandidate[K]))
}

func (h *candidateMaxHeap[K]) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// candidateMinHeap is a min heap of search candidates (for candidates)
type candidateMinHeap[K cmp.Ordered] []*searchCandidate[K]

func (h candidateMinHeap[K]) Len() int           { return len(h) }
func (h candidateMinHeap[K]) Less(i, j int) bool { return h[i].dist < h[j].dist } // Min heap
func (h candidateMinHeap[K]) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *candidateMinHeap[K]) Push(x interface{}) {
	*h = append(*h, x.(*searchCandidate[K]))
}

func (h *candidateMinHeap[K]) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// NewParquetGraph creates a new Parquet-based HNSW graph
func NewParquetGraph[K cmp.Ordered](config ParquetGraphConfig) (*ParquetGraph[K], error) {
	// Create storage
	storage, err := NewParquetStorage[K](config.Storage)
	if err != nil {
		return nil, fmt.Errorf("failed to create storage: %w", err)
	}

	// Create graph
	graph := &ParquetGraph[K]{
		M:        config.M,
		Ml:       config.Ml,
		EfSearch: config.EfSearch,
		Distance: config.Distance,
		Rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
		storage:  storage,
		layers:   make([]map[K]map[K]struct{}, 1),
	}

	// Initialize base layer
	graph.layers[0] = make(map[K]map[K]struct{})

	// Load graph from Parquet files if they exist
	if err := graph.loadFromParquet(); err != nil {
		return nil, fmt.Errorf("failed to load graph from Parquet: %w", err)
	}

	// Try to load vector dimensions from existing files
	if err := graph.loadVectorDimensions(); err != nil {
		// If files don't exist or there's an error, that's fine for a new graph
		if !os.IsNotExist(err) && !strings.Contains(err.Error(), "file does not exist") {
			fmt.Printf("Warning: failed to load vector dimensions: %v\n", err)
		}
	}

	// Create the incremental store with the dimensions we have (might be 0 for a new graph)
	incremental, err := NewIncrementalStore[K](storage, config.Incremental, graph.dimensions)
	if err != nil {
		return nil, fmt.Errorf("failed to create incremental store: %w", err)
	}

	// Compact incremental changes to improve performance when reopening
	if err := incremental.Compact(); err != nil {
		fmt.Printf("Warning: failed to compact incremental changes: %v\n", err)
	}

	// Initialize vector store with dimensions from loaded graph and the incremental store
	graph.vectorStore = NewVectorStore[K](storage, graph.dimensions, incremental)

	// Don't preload all vectors at once as it can cause performance issues
	// Instead, we'll let the cache fill naturally during operations

	return graph, nil
}

// loadFromParquet loads the graph structure from Parquet files
func (g *ParquetGraph[K]) loadFromParquet() error {
	// Load layers
	if err := g.loadLayers(); err != nil {
		// If files don't exist, that's fine - we're creating a new graph
		if os.IsNotExist(err) || strings.Contains(err.Error(), "file does not exist") {
			return nil
		}
		return fmt.Errorf("failed to load layers: %w", err)
	}

	// Load neighbors
	if err := g.loadNeighbors(); err != nil {
		// If files don't exist, that's fine - we're creating a new graph
		if os.IsNotExist(err) || strings.Contains(err.Error(), "file does not exist") {
			return nil
		}
		return fmt.Errorf("failed to load neighbors: %w", err)
	}

	// Update node count
	g.nodeCount = len(g.layers[0])

	return nil
}

// loadLayers loads the layer structure from Parquet
func (g *ParquetGraph[K]) loadLayers() error {
	// Check if file exists
	if _, err := os.Stat(g.storage.layersFile); os.IsNotExist(err) {
		return fmt.Errorf("file does not exist")
	}

	ctx := context.Background()

	// Open Parquet file
	reader, err := file.OpenParquetFile(g.storage.layersFile, g.storage.config.MemoryMap)
	if err != nil {
		return fmt.Errorf("failed to open layers file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		g.storage.createArrowReadProperties(),
		g.storage.alloc,
	)
	if err != nil {
		return fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read all records
	recordReader, err := arrowReader.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	// Initialize layers
	g.layers = make([]map[K]map[K]struct{}, 0)

	for recordReader.Next() {
		record := recordReader.Record()

		// Get layer_id and key columns
		layerIdCol := record.Column(0).(*array.Int32)
		keyCol := record.Column(1)

		// Iterate through records
		for i := 0; i < int(record.NumRows()); i++ {
			layerId := int(layerIdCol.Value(i))
			recordKey := keyCol.GetOneForMarshal(i)
			key, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			// Ensure we have enough layers
			for layerId >= len(g.layers) {
				g.layers = append(g.layers, make(map[K]map[K]struct{}))
			}

			// Add node to layer
			if g.layers[layerId] == nil {
				g.layers[layerId] = make(map[K]map[K]struct{})
			}
			if g.layers[layerId][key] == nil {
				g.layers[layerId][key] = make(map[K]struct{})
			}
		}
	}

	return nil
}

// loadNeighbors loads the neighbor connections from Parquet
func (g *ParquetGraph[K]) loadNeighbors() error {
	// Check if file exists
	if _, err := os.Stat(g.storage.neighborsFile); os.IsNotExist(err) {
		return fmt.Errorf("file does not exist")
	}

	ctx := context.Background()

	// Open Parquet file
	reader, err := file.OpenParquetFile(g.storage.neighborsFile, g.storage.config.MemoryMap)
	if err != nil {
		return fmt.Errorf("failed to open neighbors file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		g.storage.createArrowReadProperties(),
		g.storage.alloc,
	)
	if err != nil {
		return fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read all records
	recordReader, err := arrowReader.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	for recordReader.Next() {
		record := recordReader.Record()

		// Get layer_id, key, and neighbor_key columns
		layerIdCol := record.Column(0).(*array.Int32)
		keyCol := record.Column(1)
		neighborKeyCol := record.Column(2)

		// Iterate through records
		for i := 0; i < int(record.NumRows()); i++ {
			layerId := int(layerIdCol.Value(i))
			recordKey := keyCol.GetOneForMarshal(i)
			key, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			recordNeighborKey := neighborKeyCol.GetOneForMarshal(i)
			neighborKey, err := convertArrowToKey[K](recordNeighborKey)
			if err != nil {
				continue
			}

			// Ensure we have enough layers
			for layerId >= len(g.layers) {
				g.layers = append(g.layers, make(map[K]map[K]struct{}))
			}

			// Add node and neighbor to layer
			if g.layers[layerId] == nil {
				g.layers[layerId] = make(map[K]map[K]struct{})
			}
			if g.layers[layerId][key] == nil {
				g.layers[layerId][key] = make(map[K]struct{})
			}
			g.layers[layerId][key][neighborKey] = struct{}{}
		}
	}

	// Count nodes in base layer
	if len(g.layers) > 0 {
		g.nodeCount = len(g.layers[0])
	}

	return nil
}

// loadVectorDimensions loads a sample vector to determine dimensions
func (g *ParquetGraph[K]) loadVectorDimensions() error {
	// Check if file exists
	if _, err := os.Stat(g.storage.vectorsFile); os.IsNotExist(err) {
		return fmt.Errorf("file does not exist")
	}

	ctx := context.Background()

	// Open Parquet file
	reader, err := file.OpenParquetFile(g.storage.vectorsFile, g.storage.config.MemoryMap)
	if err != nil {
		return fmt.Errorf("failed to open vectors file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		g.storage.createArrowReadProperties(),
		g.storage.alloc,
	)
	if err != nil {
		return fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read first record
	recordReader, err := arrowReader.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	if recordReader.Next() {
		record := recordReader.Record()
		if record.NumRows() > 0 {
			// Get vector column
			vectorCol := record.Column(1)
			listArray := vectorCol.(*array.List)

			// Get first vector's length
			start := int(listArray.Offsets()[0])
			end := int(listArray.Offsets()[1])
			g.dimensions = end - start
		}
	}

	return nil
}

// Add adds a node to the graph
func (g *ParquetGraph[K]) Add(nodes ...hnsw.Node[K]) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Validate dimensions
	if g.dimensions > 0 {
		for _, node := range nodes {
			if len(node.Value) != g.dimensions {
				return fmt.Errorf("vector dimension mismatch: expected %d, got %d", g.dimensions, len(node.Value))
			}
		}
	} else if len(nodes) > 0 {
		// Set dimensions from first node
		g.dimensions = len(nodes[0].Value)

		// Create incremental store with correct dimensions
		incrementalConfig := DefaultIncrementalConfig()
		incremental, err := NewIncrementalStore[K](g.storage, incrementalConfig, g.dimensions)
		if err != nil {
			return fmt.Errorf("failed to create incremental store: %w", err)
		}

		// Create vector store with correct dimensions
		g.vectorStore = NewVectorStore[K](g.storage, g.dimensions, incremental)
	}

	// Store vectors
	vectors := make(map[K][]float32, len(nodes))
	for _, node := range nodes {
		vectors[node.Key] = node.Value
	}

	if err := g.vectorStore.StoreVectors(vectors); err != nil {
		return fmt.Errorf("failed to store vectors: %w", err)
	}

	// Add nodes to graph structure
	for _, node := range nodes {
		if err := g.addNode(node.Key, node.Value); err != nil {
			return fmt.Errorf("failed to add node: %w", err)
		}
	}

	// Save graph structure to Parquet
	if err := g.saveToParquet(); err != nil {
		return fmt.Errorf("failed to save graph to Parquet: %w", err)
	}

	return nil
}

// addNode adds a single node to the graph structure
func (g *ParquetGraph[K]) addNode(key K, vector []float32) error {
	// Generate random level
	level := g.randomLevel()

	// Ensure we have enough layers
	for level >= len(g.layers) {
		g.layers = append(g.layers, make(map[K]map[K]struct{}))
	}

	// Add node to each layer
	for l := 0; l <= level; l++ {
		if g.layers[l] == nil {
			g.layers[l] = make(map[K]map[K]struct{})
		}

		// Skip if node already exists in this layer
		if _, exists := g.layers[l][key]; exists {
			continue
		}

		g.layers[l][key] = make(map[K]struct{})

		// If this is the first node in the layer, no connections to make
		if len(g.layers[l]) == 1 {
			continue
		}

		// Find nearest neighbors in this layer
		neighbors := g.findNeighbors(l, vector, g.M)

		// Add connections
		for _, neighbor := range neighbors {
			g.layers[l][key][neighbor] = struct{}{}
			g.layers[l][neighbor][key] = struct{}{}
		}
	}

	// Update node count
	g.nodeCount = len(g.layers[0])

	return nil
}

// randomLevel generates a random level for a new node
func (g *ParquetGraph[K]) randomLevel() int {
	level := 0
	maxLevel := int(math.Log(float64(g.nodeCount+1)) / math.Log(1.0/g.Ml))
	if maxLevel < 1 {
		maxLevel = 1
	}

	for level < maxLevel && g.Rng.Float64() < g.Ml {
		level++
	}

	return level
}

// findNeighbors finds the nearest neighbors for a node in a specific layer
func (g *ParquetGraph[K]) findNeighbors(layer int, vector []float32, count int) []K {
	// If layer is empty, return empty slice
	if len(g.layers[layer]) == 0 {
		return []K{}
	}

	// Get a random entry point
	var entryKey K
	for k := range g.layers[layer] {
		entryKey = k
		break
	}

	// Find nearest neighbors using priority queues (similar to searchLayer)
	// Create a temporary cache for vector retrievals during this search
	vectorCache := make(map[K][]float32)

	// Get entry point vector and cache it
	entryVector, err := g.vectorStore.GetVector(entryKey)
	if err != nil {
		return []K{}
	}
	vectorCache[entryKey] = entryVector

	entryDist := g.Distance(vector, entryVector)

	// Initialize visited set
	visited := map[K]bool{entryKey: true}

	// Initialize candidates min-heap (closest first for processing)
	candidates := &candidateMinHeap[K]{&searchCandidate[K]{key: entryKey, dist: entryDist}}
	heap.Init(candidates)

	// Initialize results max-heap (furthest first for easy pruning)
	results := &candidateMaxHeap[K]{&searchCandidate[K]{key: entryKey, dist: entryDist}}
	heap.Init(results)

	// Keep track of the worst (largest) distance in the result set
	worstDistance := entryDist

	// Process candidates
	for candidates.Len() > 0 && len(visited) < 2*count {
		// Get closest candidate
		current := heap.Pop(candidates).(*searchCandidate[K])

		// Early termination: if this candidate is farther than the worst result
		// and we have enough results, we're done
		if results.Len() >= count && current.dist > worstDistance {
			break
		}

		// Collect all neighbors that haven't been visited yet
		neighborsToProcess := make([]K, 0)
		for neighbor := range g.layers[layer][current.key] {
			if !visited[neighbor] {
				neighborsToProcess = append(neighborsToProcess, neighbor)
				visited[neighbor] = true
			}
		}

		// If no new neighbors, continue to next candidate
		if len(neighborsToProcess) == 0 {
			continue
		}

		// Batch retrieve vectors for all neighbors
		neighborVectors, err := g.vectorStore.GetVectorsBatch(neighborsToProcess)
		if err != nil {
			// If batch retrieval fails, fall back to individual retrieval
			for _, neighbor := range neighborsToProcess {
				if neighborVector, err := g.vectorStore.GetVector(neighbor); err == nil {
					vectorCache[neighbor] = neighborVector

					dist := g.Distance(vector, neighborVector)

					// Add to results if better than worst result or if we don't have enough results yet
					if results.Len() < count || dist < worstDistance {
						heap.Push(results, &searchCandidate[K]{key: neighbor, dist: dist})

						// If we have too many results, remove the worst one
						if results.Len() > count {
							heap.Pop(results)
							// Update worst distance
							worstDistance = (*results)[0].dist
						}
					}

					// Only add to candidates if it can potentially improve results
					if dist < worstDistance || results.Len() < count {
						heap.Push(candidates, &searchCandidate[K]{key: neighbor, dist: dist})
					}
				}
			}
		} else {
			// Process all retrieved vectors
			for neighbor, neighborVector := range neighborVectors {
				vectorCache[neighbor] = neighborVector

				dist := g.Distance(vector, neighborVector)

				// Add to results if better than worst result or if we don't have enough results yet
				if results.Len() < count || dist < worstDistance {
					heap.Push(results, &searchCandidate[K]{key: neighbor, dist: dist})

					// If we have too many results, remove the worst one
					if results.Len() > count {
						heap.Pop(results)
						// Update worst distance
						worstDistance = (*results)[0].dist
					}
				}

				// Only add to candidates if it can potentially improve results
				if dist < worstDistance || results.Len() < count {
					heap.Push(candidates, &searchCandidate[K]{key: neighbor, dist: dist})
				}
			}
		}
	}

	// Extract keys from results (sorted by distance)
	resultHeap := *results
	sort.Slice(resultHeap, func(i, j int) bool {
		return resultHeap[i].dist < resultHeap[j].dist
	})

	// Calculate result count (min of resultHeap length and count)
	resultCount := len(resultHeap)
	if resultCount > count {
		resultCount = count
	}

	keys := make([]K, 0, resultCount)
	for i := 0; i < resultCount; i++ {
		keys = append(keys, resultHeap[i].key)
	}

	return keys
}

// saveToParquet saves the graph structure to Parquet files
func (g *ParquetGraph[K]) saveToParquet() error {
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

// saveLayers saves the layer structure to Parquet
func (g *ParquetGraph[K]) saveLayers() error {
	// Create schema
	schema := g.storage.LayerSchema()

	// Create Arrow record batch
	layerIdBuilder := array.NewInt32Builder(g.storage.alloc)
	defer layerIdBuilder.Release()

	keyBuilder := array.NewBuilder(g.storage.alloc, getKeyType[K]())
	defer keyBuilder.Release()

	// Add data to builders
	for layerId, layer := range g.layers {
		for key := range layer {
			layerIdBuilder.Append(int32(layerId))
			appendToBuilder(keyBuilder, key)
		}
	}

	// Create record batch
	layerIdArr := layerIdBuilder.NewArray()
	defer layerIdArr.Release()

	keyArr := keyBuilder.NewArray()
	defer keyArr.Release()

	cols := []arrow.Array{layerIdArr, keyArr}
	batch := array.NewRecord(schema, cols, int64(layerIdBuilder.Len()))
	defer batch.Release()

	// Write to Parquet file
	file, err := os.Create(g.storage.layersFile)
	if err != nil {
		return fmt.Errorf("failed to create layers file: %w", err)
	}
	defer file.Close()

	writer, err := pqarrow.NewFileWriter(
		schema,
		file,
		g.storage.createWriterProperties(),
		g.storage.createArrowWriterProperties(),
	)
	if err != nil {
		return fmt.Errorf("failed to create Parquet writer: %w", err)
	}

	if err := writer.Write(batch); err != nil {
		return fmt.Errorf("failed to write layers: %w", err)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("failed to close writer: %w", err)
	}

	return nil
}

// saveNeighbors saves the neighbor connections to Parquet
func (g *ParquetGraph[K]) saveNeighbors() error {
	// Create schema
	schema := g.storage.NeighborSchema()

	// Create Arrow record batch
	layerIdBuilder := array.NewInt32Builder(g.storage.alloc)
	defer layerIdBuilder.Release()

	keyBuilder := array.NewBuilder(g.storage.alloc, getKeyType[K]())
	defer keyBuilder.Release()

	neighborKeyBuilder := array.NewBuilder(g.storage.alloc, getKeyType[K]())
	defer neighborKeyBuilder.Release()

	// Add data to builders
	for layerId, layer := range g.layers {
		for key, neighbors := range layer {
			for neighbor := range neighbors {
				layerIdBuilder.Append(int32(layerId))
				appendToBuilder(keyBuilder, key)
				appendToBuilder(neighborKeyBuilder, neighbor)
			}
		}
	}

	// Create record batch
	layerIdArr := layerIdBuilder.NewArray()
	defer layerIdArr.Release()

	keyArr := keyBuilder.NewArray()
	defer keyArr.Release()

	neighborKeyArr := neighborKeyBuilder.NewArray()
	defer neighborKeyArr.Release()

	cols := []arrow.Array{layerIdArr, keyArr, neighborKeyArr}
	batch := array.NewRecord(schema, cols, int64(layerIdBuilder.Len()))
	defer batch.Release()

	// Write to Parquet file
	file, err := os.Create(g.storage.neighborsFile)
	if err != nil {
		return fmt.Errorf("failed to create neighbors file: %w", err)
	}
	defer file.Close()

	writer, err := pqarrow.NewFileWriter(
		schema,
		file,
		g.storage.createWriterProperties(),
		g.storage.createArrowWriterProperties(),
	)
	if err != nil {
		return fmt.Errorf("failed to create Parquet writer: %w", err)
	}

	if err := writer.Write(batch); err != nil {
		return fmt.Errorf("failed to write neighbors: %w", err)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("failed to close writer: %w", err)
	}

	return nil
}

// appendToBuilder appends a key to the appropriate builder based on its type
func appendToBuilder[K cmp.Ordered](builder array.Builder, key K) {
	switch b := builder.(type) {
	case *array.Int32Builder:
		if k, ok := any(key).(int32); ok {
			b.Append(k)
		}
	case *array.Int64Builder:
		if k, ok := any(key).(int64); ok {
			b.Append(k)
		} else if k, ok := any(key).(int); ok {
			b.Append(int64(k))
		}
	case *array.Uint32Builder:
		if k, ok := any(key).(uint32); ok {
			b.Append(k)
		}
	case *array.Uint64Builder:
		if k, ok := any(key).(uint64); ok {
			b.Append(k)
		} else if k, ok := any(key).(uint); ok {
			b.Append(uint64(k))
		}
	case *array.Float32Builder:
		if k, ok := any(key).(float32); ok {
			b.Append(k)
		}
	case *array.Float64Builder:
		if k, ok := any(key).(float64); ok {
			b.Append(k)
		}
	case *array.StringBuilder:
		if k, ok := any(key).(string); ok {
			b.Append(k)
		}
	case *array.BinaryBuilder:
		if k, ok := any(key).([]byte); ok {
			b.Append(k)
		}
	default:
		// For other types, convert to string as fallback
		b.(*array.StringBuilder).Append(fmt.Sprintf("%v", key))
	}
}

// Search performs a k-nearest neighbor search
func (g *ParquetGraph[K]) Search(query []float32, k int) ([]hnsw.Node[K], error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if len(g.layers) == 0 {
		return []hnsw.Node[K]{}, nil
	}

	// Start from the top layer
	layer := len(g.layers) - 1

	// Get a random entry point from the top layer
	var entryKey K
	found := false
	for k := range g.layers[layer] {
		entryKey = k
		found = true
		break
	}

	// If no entry point found, return empty result
	if !found {
		return []hnsw.Node[K]{}, nil
	}

	// Search through layers
	for ; layer > 0; layer-- {
		result := g.searchLayer(layer, query, entryKey, 1)
		if len(result) == 0 {
			// If no results found at this layer, return empty result
			return []hnsw.Node[K]{}, nil
		}
		entryKey = result[0]
	}

	// Search base layer with ef = k
	result := g.searchLayer(0, query, entryKey, max(k, g.EfSearch))
	if len(result) == 0 {
		return []hnsw.Node[K]{}, nil
	}

	// Limit results to k
	if len(result) > k {
		result = result[:k]
	}

	// Convert to Node objects
	nodes := make([]hnsw.Node[K], 0, len(result))

	// Batch retrieve vectors for all results
	resultVectors, err := g.vectorStore.GetVectorsBatch(result)
	if err != nil {
		// Fall back to individual retrieval if batch fails
		for _, key := range result {
			vector, err := g.vectorStore.GetVector(key)
			if err != nil {
				continue
			}

			nodes = append(nodes, hnsw.Node[K]{
				Key:   key,
				Value: vector,
			})
		}
	} else {
		// Use batch results
		for _, key := range result {
			if vector, ok := resultVectors[key]; ok {
				nodes = append(nodes, hnsw.Node[K]{
					Key:   key,
					Value: vector,
				})
			}
		}
	}

	return nodes, nil
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// searchLayer performs a search within a specific layer using priority queues
func (g *ParquetGraph[K]) searchLayer(layer int, query []float32, entryKey K, ef int) []K {
	// Create a temporary cache for vector retrievals during this search
	vectorCache := make(map[K][]float32)

	// Get entry point vector and cache it
	entryVector, err := g.vectorStore.GetVector(entryKey)
	if err != nil {
		return []K{}
	}
	vectorCache[entryKey] = entryVector

	entryDist := g.Distance(query, entryVector)

	// Initialize visited set
	visited := map[K]bool{entryKey: true}

	// Initialize candidates min-heap (closest first for processing)
	candidates := &candidateMinHeap[K]{&searchCandidate[K]{key: entryKey, dist: entryDist}}
	heap.Init(candidates)

	// Initialize results max-heap (furthest first for easy pruning)
	results := &candidateMaxHeap[K]{&searchCandidate[K]{key: entryKey, dist: entryDist}}
	heap.Init(results)

	// Keep track of the worst (largest) distance in the result set
	worstDistance := entryDist

	// Process candidates
	for candidates.Len() > 0 {
		// Get closest candidate
		current := heap.Pop(candidates).(*searchCandidate[K])

		// Early termination: if this candidate is farther than the worst result
		// and we have enough results, we're done
		if results.Len() >= ef && current.dist > worstDistance {
			break
		}

		// Collect all neighbors that haven't been visited yet
		neighborsToProcess := make([]K, 0)
		for neighbor := range g.layers[layer][current.key] {
			if !visited[neighbor] {
				neighborsToProcess = append(neighborsToProcess, neighbor)
				visited[neighbor] = true
			}
		}

		// If no new neighbors, continue to next candidate
		if len(neighborsToProcess) == 0 {
			continue
		}

		// First check if any neighbors are already in the cache
		neighborsToFetch := make([]K, 0, len(neighborsToProcess))
		for _, neighbor := range neighborsToProcess {
			if vector, ok := vectorCache[neighbor]; ok {
				// Process cached vector
				dist := g.Distance(query, vector)

				// Add to results if better than worst result or if we don't have enough results yet
				if results.Len() < ef || dist < worstDistance {
					heap.Push(results, &searchCandidate[K]{key: neighbor, dist: dist})

					// If we have too many results, remove the worst one
					if results.Len() > ef {
						heap.Pop(results)
						// Update worst distance
						worstDistance = (*results)[0].dist
					}
				}

				// Only add to candidates if it can potentially improve results
				if dist < worstDistance || results.Len() < ef {
					heap.Push(candidates, &searchCandidate[K]{key: neighbor, dist: dist})
				}
			} else {
				neighborsToFetch = append(neighborsToFetch, neighbor)
			}
		}

		// If all neighbors were in cache, continue to next candidate
		if len(neighborsToFetch) == 0 {
			continue
		}

		// Batch retrieve vectors for neighbors not in cache
		neighborVectors, err := g.vectorStore.GetVectorsBatch(neighborsToFetch)
		if err != nil {
			// If batch retrieval fails, fall back to individual retrieval
			for _, neighbor := range neighborsToFetch {
				if neighborVector, err := g.vectorStore.GetVector(neighbor); err == nil {
					vectorCache[neighbor] = neighborVector

					dist := g.Distance(query, neighborVector)

					// Add to results if better than worst result or if we don't have enough results yet
					if results.Len() < ef || dist < worstDistance {
						heap.Push(results, &searchCandidate[K]{key: neighbor, dist: dist})

						// If we have too many results, remove the worst one
						if results.Len() > ef {
							heap.Pop(results)
							// Update worst distance
							worstDistance = (*results)[0].dist
						}
					}

					// Only add to candidates if it can potentially improve results
					if dist < worstDistance || results.Len() < ef {
						heap.Push(candidates, &searchCandidate[K]{key: neighbor, dist: dist})
					}
				}
			}
		} else {
			// Process all retrieved vectors
			for neighbor, neighborVector := range neighborVectors {
				vectorCache[neighbor] = neighborVector

				dist := g.Distance(query, neighborVector)

				// Add to results if better than worst result or if we don't have enough results yet
				if results.Len() < ef || dist < worstDistance {
					heap.Push(results, &searchCandidate[K]{key: neighbor, dist: dist})

					// If we have too many results, remove the worst one
					if results.Len() > ef {
						heap.Pop(results)
						// Update worst distance
						worstDistance = (*results)[0].dist
					}
				}

				// Only add to candidates if it can potentially improve results
				if dist < worstDistance || results.Len() < ef {
					heap.Push(candidates, &searchCandidate[K]{key: neighbor, dist: dist})
				}
			}
		}
	}

	// Extract keys from results (sorted by distance)
	resultHeap := *results
	sort.Slice(resultHeap, func(i, j int) bool {
		return resultHeap[i].dist < resultHeap[j].dist
	})

	keys := make([]K, len(resultHeap))
	for i, r := range resultHeap {
		keys[i] = r.key
	}

	return keys
}

// Delete removes a node from the graph
func (g *ParquetGraph[K]) Delete(key K) bool {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check if node exists in base layer
	if len(g.layers) == 0 || g.layers[0] == nil || g.layers[0][key] == nil {
		return false
	}

	// Remove from all layers
	for l := 0; l < len(g.layers); l++ {
		if g.layers[l] == nil || g.layers[l][key] == nil {
			continue
		}

		// Remove connections to this node
		for neighbor := range g.layers[l][key] {
			delete(g.layers[l][neighbor], key)
		}

		// Remove node
		delete(g.layers[l], key)
	}

	// Remove vector
	if g.vectorStore != nil {
		if err := g.vectorStore.DeleteVector(key); err != nil {
			// Log error but continue
			fmt.Printf("Error deleting vector: %v\n", err)
		}
	}

	// Update node count
	g.nodeCount = len(g.layers[0])

	// Save changes to Parquet
	if err := g.saveToParquet(); err != nil {
		// Log error but continue
		fmt.Printf("Error saving to Parquet: %v\n", err)
	}

	return true
}

// BatchDelete removes multiple nodes from the graph
func (g *ParquetGraph[K]) BatchDelete(keys []K) []bool {
	results := make([]bool, len(keys))

	for i, key := range keys {
		results[i] = g.Delete(key)
	}

	return results
}

// Len returns the number of nodes in the graph
func (g *ParquetGraph[K]) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return g.nodeCount
}

// Close releases resources used by the graph
func (g *ParquetGraph[K]) Close() error {
	return g.storage.Close()
}
