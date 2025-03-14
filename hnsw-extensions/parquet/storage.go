package parquet

import (
	"cmp"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/compress"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

// ParquetStorageConfig defines configuration options for the Parquet storage
type ParquetStorageConfig struct {
	// Directory where Parquet files will be stored
	Directory string

	// Compression codec to use (default: Snappy)
	Compression compress.Compression

	// Batch size for reading/writing (default: 64MB)
	BatchSize int64

	// Maximum row group length (default: 64MB)
	MaxRowGroupLength int64

	// Data page size (default: 1MB)
	DataPageSize int64

	// Whether to memory map files when reading
	MemoryMap bool
}

// DefaultParquetStorageConfig returns the default configuration
func DefaultParquetStorageConfig() ParquetStorageConfig {
	return ParquetStorageConfig{
		Directory:         "hnsw_parquet_data",
		Compression:       compress.Codecs.Snappy,
		BatchSize:         64 * 1024 * 1024, // 64MB
		MaxRowGroupLength: 64 * 1024 * 1024, // 64MB
		DataPageSize:      1 * 1024 * 1024,  // 1MB
		MemoryMap:         true,
	}
}

// ParquetStorage provides persistent storage for HNSW graph using Parquet files
type ParquetStorage[K cmp.Ordered] struct {
	config ParquetStorageConfig
	alloc  memory.Allocator
	mu     sync.RWMutex

	// File paths
	vectorsFile   string
	layersFile    string
	neighborsFile string
	metadataFile  string
}

// NewParquetStorage creates a new Parquet storage instance
func NewParquetStorage[K cmp.Ordered](config ParquetStorageConfig) (*ParquetStorage[K], error) {
	if config.Directory == "" {
		config = DefaultParquetStorageConfig()
	}

	// Create directory if it doesn't exist
	if err := os.MkdirAll(config.Directory, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	storage := &ParquetStorage[K]{
		config:        config,
		alloc:         memory.NewGoAllocator(),
		vectorsFile:   filepath.Join(config.Directory, "vectors.parquet"),
		layersFile:    filepath.Join(config.Directory, "layers.parquet"),
		neighborsFile: filepath.Join(config.Directory, "neighbors.parquet"),
		metadataFile:  filepath.Join(config.Directory, "metadata.parquet"),
	}

	return storage, nil
}

// Close releases resources used by the storage
func (ps *ParquetStorage[K]) Close() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Any cleanup operations would go here
	return nil
}

// createWriterProperties creates Parquet writer properties based on the configuration
func (ps *ParquetStorage[K]) createWriterProperties() *parquet.WriterProperties {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	return parquet.NewWriterProperties(
		parquet.WithCompression(compress.Codecs.Snappy),
		parquet.WithBatchSize(ps.config.BatchSize),
		parquet.WithAllocator(ps.alloc),
		parquet.WithVersion(parquet.V2_LATEST),
		parquet.WithDataPageSize(ps.config.DataPageSize),
		parquet.WithMaxRowGroupLength(ps.config.MaxRowGroupLength),
		parquet.WithCreatedBy("HNSW-TFMV"),
	)
}

// createArrowWriterProperties creates Arrow writer properties
func (ps *ParquetStorage[K]) createArrowWriterProperties() pqarrow.ArrowWriterProperties {
	return pqarrow.NewArrowWriterProperties(
		pqarrow.WithStoreSchema(),
	)
}

// createArrowReadProperties creates Arrow read properties
func (ps *ParquetStorage[K]) createArrowReadProperties() pqarrow.ArrowReadProperties {
	return pqarrow.ArrowReadProperties{
		Parallel:  true,
		BatchSize: int64(ps.config.BatchSize),
	}
}

// VectorSchema returns the Arrow schema for vector storage
func (ps *ParquetStorage[K]) VectorSchema(dims int) *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "key", Type: getKeyType[K]()},
			{Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)
}

// LayerSchema returns the Arrow schema for layer storage
func (ps *ParquetStorage[K]) LayerSchema() *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "layer_id", Type: arrow.PrimitiveTypes.Int32},
			{Name: "key", Type: getKeyType[K]()},
		},
		nil,
	)
}

// NeighborSchema returns the Arrow schema for neighbor storage
func (ps *ParquetStorage[K]) NeighborSchema() *arrow.Schema {
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
func (ps *ParquetStorage[K]) MetadataSchema() *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "key", Type: arrow.BinaryTypes.String},
			{Name: "value", Type: arrow.BinaryTypes.String},
		},
		nil,
	)
}

// GetDirectory returns the directory path where Parquet files are stored
func (ps *ParquetStorage[K]) GetDirectory() string {
	return ps.config.Directory
}
