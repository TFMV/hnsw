package arrow

import (
	"cmp"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// ArrowStorageConfig defines configuration options for the Arrow storage
type ArrowStorageConfig struct {
	// Directory where Arrow files will be stored
	Directory string

	// Memory allocation strategy
	MemoryPoolSize int64

	// Whether to use memory mapping for files
	MemoryMap bool

	// Batch size for processing
	BatchSize int64

	// Number of worker goroutines for parallel operations
	NumWorkers int
}

// DefaultArrowStorageConfig returns the default configuration
func DefaultArrowStorageConfig() ArrowStorageConfig {
	return ArrowStorageConfig{
		Directory:      "hnsw_arrow_data",
		MemoryPoolSize: 1 << 30, // 1GB
		MemoryMap:      true,
		BatchSize:      64 * 1024 * 1024, // 64MB
		NumWorkers:     4,
	}
}

// ArrowStorage provides persistent storage for HNSW graph using Arrow columnar format
type ArrowStorage[K cmp.Ordered] struct {
	config ArrowStorageConfig
	alloc  memory.Allocator
	mu     sync.RWMutex

	// File paths
	vectorsFile   string
	layersFile    string
	neighborsFile string
	metadataFile  string

	// Memory pool for efficient memory management
	memPool *memory.GoAllocator
}

// NewArrowStorage creates a new Arrow storage instance
func NewArrowStorage[K cmp.Ordered](config ArrowStorageConfig) (*ArrowStorage[K], error) {
	if config.Directory == "" {
		config = DefaultArrowStorageConfig()
	}

	// Create directory if it doesn't exist
	if err := os.MkdirAll(config.Directory, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	// Initialize memory pool
	memPool := memory.NewGoAllocator()

	storage := &ArrowStorage[K]{
		config:        config,
		alloc:         memPool,
		memPool:       memPool,
		vectorsFile:   filepath.Join(config.Directory, "vectors.arrow"),
		layersFile:    filepath.Join(config.Directory, "layers.arrow"),
		neighborsFile: filepath.Join(config.Directory, "neighbors.arrow"),
		metadataFile:  filepath.Join(config.Directory, "metadata.arrow"),
	}

	return storage, nil
}

// Close releases resources used by the storage
func (as *ArrowStorage[K]) Close() error {
	as.mu.Lock()
	defer as.mu.Unlock()

	// Any cleanup operations would go here
	return nil
}

// VectorSchema returns the Arrow schema for vector storage
func (as *ArrowStorage[K]) VectorSchema(dims int) *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "key", Type: getKeyType[K]()},
			{Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)
}

// LayerSchema returns the Arrow schema for layer storage
func (as *ArrowStorage[K]) LayerSchema() *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "layer_id", Type: arrow.PrimitiveTypes.Int32},
			{Name: "key", Type: getKeyType[K]()},
		},
		nil,
	)
}

// NeighborSchema returns the Arrow schema for neighbor storage
func (as *ArrowStorage[K]) NeighborSchema() *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "layer_id", Type: arrow.PrimitiveTypes.Int32},
			{Name: "key", Type: getKeyType[K]()},
			{Name: "neighbor_key", Type: getKeyType[K]()},
		},
		nil,
	)
}

// MetadataSchema returns the Arrow schema for metadata storage
func (as *ArrowStorage[K]) MetadataSchema() *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "key", Type: arrow.BinaryTypes.String},
			{Name: "value", Type: arrow.BinaryTypes.String},
		},
		nil,
	)
}

// GetDirectory returns the directory path where Arrow files are stored
func (as *ArrowStorage[K]) GetDirectory() string {
	return as.config.Directory
}

// CreateRecordBuilder creates a new record builder for the given schema
func (as *ArrowStorage[K]) CreateRecordBuilder(schema *arrow.Schema) *array.RecordBuilder {
	return array.NewRecordBuilder(as.alloc, schema)
}

// getKeyType returns the Arrow data type for a given key type
func getKeyType[K cmp.Ordered]() arrow.DataType {
	var zero K
	switch any(zero).(type) {
	case int:
		return arrow.PrimitiveTypes.Int64
	case int32:
		return arrow.PrimitiveTypes.Int32
	case int64:
		return arrow.PrimitiveTypes.Int64
	case uint:
		return arrow.PrimitiveTypes.Uint64
	case uint32:
		return arrow.PrimitiveTypes.Uint32
	case uint64:
		return arrow.PrimitiveTypes.Uint64
	case float32:
		return arrow.PrimitiveTypes.Float32
	case float64:
		return arrow.PrimitiveTypes.Float64
	case string:
		return arrow.BinaryTypes.String
	case []byte:
		return arrow.BinaryTypes.Binary
	default:
		// For unsupported types, use string as fallback
		return arrow.BinaryTypes.String
	}
}

// Stats returns statistics about the storage
func (s *ArrowStorage[K]) Stats() map[string]interface{} {
	stats := make(map[string]interface{})

	// Add basic storage information
	stats["storage_dir"] = s.config.Directory
	stats["num_workers"] = s.config.NumWorkers
	stats["batch_size"] = s.config.BatchSize
	stats["memory_map"] = s.config.MemoryMap
	stats["memory_pool_size"] = s.config.MemoryPoolSize

	// Check if vector files exist and get their sizes
	if _, err := os.Stat(s.vectorsFile); err == nil {
		if info, err := os.Stat(s.vectorsFile); err == nil {
			stats["vectors_file_size"] = info.Size()
		}
	}

	if _, err := os.Stat(s.layersFile); err == nil {
		if info, err := os.Stat(s.layersFile); err == nil {
			stats["layers_file_size"] = info.Size()
		}
	}

	if _, err := os.Stat(s.neighborsFile); err == nil {
		if info, err := os.Stat(s.neighborsFile); err == nil {
			stats["neighbors_file_size"] = info.Size()
		}
	}

	return stats
}
