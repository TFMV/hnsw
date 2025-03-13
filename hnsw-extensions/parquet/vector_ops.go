package parquet

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

// VectorStore handles storage and retrieval of vectors in Parquet format
type VectorStore[K cmp.Ordered] struct {
	storage *ParquetStorage[K]
	dims    int
	mu      sync.RWMutex
	cache   map[K][]float32 // Optional in-memory cache for frequently accessed vectors

	// For lazy updates
	dirty            bool
	pendingWrites    map[K][]float32
	pendingDeletes   map[K]bool
	flushInterval    time.Duration
	lastFlushTime    time.Time
	maxPendingWrites int
	flushInProgress  bool

	// Incremental updates
	incremental *IncrementalStore[K]

	// Base vectors cache
	baseVectorsMu      sync.RWMutex
	baseVectorsCache   map[K][]float32
	baseVectorsCached  bool
	baseVectorsLoading sync.Once
}

// NewVectorStore creates a new vector store
func NewVectorStore[K cmp.Ordered](storage *ParquetStorage[K], dims int, incremental *IncrementalStore[K]) *VectorStore[K] {
	vs := &VectorStore[K]{
		storage:           storage,
		dims:              dims,
		cache:             make(map[K][]float32, 10000), // Increase cache size to 10,000 vectors
		pendingWrites:     make(map[K][]float32),
		pendingDeletes:    make(map[K]bool),
		flushInterval:     5 * time.Minute, // Default flush interval
		lastFlushTime:     time.Now(),
		maxPendingWrites:  1000, // Default max pending writes before forced flush
		incremental:       incremental,
		baseVectorsCache:  make(map[K][]float32),
		baseVectorsCached: false,
	}

	// Start background flush goroutine
	go vs.backgroundFlush()

	return vs
}

// SetFlushInterval sets the interval for automatic flushing of pending changes
func (vs *VectorStore[K]) SetFlushInterval(interval time.Duration) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.flushInterval = interval
}

// SetMaxPendingWrites sets the maximum number of pending writes before a flush is forced
func (vs *VectorStore[K]) SetMaxPendingWrites(max int) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.maxPendingWrites = max
}

// backgroundFlush periodically flushes pending changes to disk
func (vs *VectorStore[K]) backgroundFlush() {
	ticker := time.NewTicker(30 * time.Second) // Check every 30 seconds
	defer ticker.Stop()

	for range ticker.C {
		vs.mu.RLock()
		timeSinceLastFlush := time.Since(vs.lastFlushTime)
		hasPendingChanges := len(vs.pendingWrites) > 0 || len(vs.pendingDeletes) > 0
		shouldFlush := hasPendingChanges && timeSinceLastFlush >= vs.flushInterval
		vs.mu.RUnlock()

		if shouldFlush {
			_ = vs.Flush() // Ignore errors in background flush
		}
	}
}

// Flush writes all pending changes to disk
func (vs *VectorStore[K]) Flush() error {
	vs.mu.Lock()
	if vs.flushInProgress || (!vs.dirty && len(vs.pendingWrites) == 0 && len(vs.pendingDeletes) == 0) {
		vs.mu.Unlock()
		return nil // Nothing to flush
	}

	// Mark flush in progress to prevent concurrent flushes
	vs.flushInProgress = true

	// Create local copies of pending changes
	pendingWrites := make(map[K][]float32)
	for k, v := range vs.pendingWrites {
		pendingWrites[k] = v
	}

	pendingDeletes := make(map[K]bool)
	for k, v := range vs.pendingDeletes {
		pendingDeletes[k] = v
	}

	// Clear pending changes
	vs.pendingWrites = make(map[K][]float32)
	vs.pendingDeletes = make(map[K]bool)
	vs.dirty = false
	vs.lastFlushTime = time.Now()
	vs.mu.Unlock()

	// Add changes to incremental store
	for k, v := range pendingWrites {
		if err := vs.incremental.AddChange(ChangeTypeAdd, k, v); err != nil {
			vs.mu.Lock()
			vs.flushInProgress = false
			vs.mu.Unlock()
			return err
		}
	}

	for k := range pendingDeletes {
		if err := vs.incremental.AddChange(ChangeTypeDelete, k, nil); err != nil {
			vs.mu.Lock()
			vs.flushInProgress = false
			vs.mu.Unlock()
			return err
		}
	}

	// Check if compaction should be performed
	if vs.incremental.ShouldCompact() {
		if err := vs.incremental.Compact(); err != nil {
			vs.mu.Lock()
			vs.flushInProgress = false
			vs.mu.Unlock()
			return err
		}
	}

	vs.mu.Lock()
	vs.flushInProgress = false
	vs.mu.Unlock()
	return nil
}

// StoreVectors stores multiple vectors in a single operation
func (vs *VectorStore[K]) StoreVectors(vectors map[K][]float32) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate dimensions
	for _, vector := range vectors {
		if len(vector) != vs.dims {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d", vs.dims, len(vector))
		}
	}

	vs.mu.Lock()

	// Add to pending writes
	for key, vector := range vectors {
		// Add to cache
		vs.cache[key] = vector
		vs.pendingWrites[key] = vector

		// Add to incremental store if available
		if vs.incremental != nil {
			if err := vs.incremental.AddChange(ChangeTypeAdd, key, vector); err != nil {
				vs.mu.Unlock()
				return fmt.Errorf("failed to add to incremental store: %w", err)
			}
		}
	}

	vs.dirty = true
	shouldFlush := len(vs.pendingWrites) >= vs.maxPendingWrites

	vs.mu.Unlock()

	// If we've reached the maximum pending writes, flush immediately
	if shouldFlush {
		return vs.Flush()
	}

	return nil
}

// GetVector retrieves a vector by key
func (vs *VectorStore[K]) GetVector(key K) ([]float32, error) {
	vs.mu.RLock()
	// Check cache first
	if vec, ok := vs.cache[key]; ok {
		vs.mu.RUnlock()
		return vec, nil
	}
	vs.mu.RUnlock()

	// Get base vector from Parquet file (if it exists)
	baseVector, err := vs.getVectorFromParquet(key)
	if err != nil && !os.IsNotExist(err) && err.Error() != "vector file does not exist" && err.Error() != "vector not found for key" {
		return nil, err
	}

	// Check incremental store
	vector, found, err := vs.incremental.GetVector(key, baseVector)
	if err != nil {
		return nil, err
	}

	if found {
		// If vector was deleted, return not found error
		if vector == nil {
			return nil, fmt.Errorf("vector not found for key")
		}

		// Add to cache
		vs.mu.Lock()
		vs.cache[key] = vector
		vs.mu.Unlock()

		return vector, nil
	}

	// If base vector exists, return it
	if baseVector != nil {
		// Add to cache
		vs.mu.Lock()
		vs.cache[key] = baseVector
		vs.mu.Unlock()

		return baseVector, nil
	}

	return nil, fmt.Errorf("vector not found for key")
}

// getVectorFromParquet retrieves a vector directly from the Parquet file
func (vs *VectorStore[K]) getVectorFromParquet(key K) ([]float32, error) {
	// Check if file exists
	if _, err := os.Stat(vs.storage.vectorsFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("vector file does not exist")
	}

	// Open Parquet file
	reader, err := file.OpenParquetFile(vs.storage.vectorsFile, vs.storage.config.MemoryMap)
	if err != nil {
		return nil, fmt.Errorf("failed to open Parquet file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		vs.storage.createArrowReadProperties(),
		vs.storage.alloc,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read all records (in a real implementation, you'd want to filter by key)
	recordReader, err := arrowReader.GetRecordReader(context.Background(), nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	for recordReader.Next() {
		record := recordReader.Record()

		// Get key and vector columns
		keyCol := record.Column(0)
		vectorCol := record.Column(1)

		// Iterate through records to find matching key
		for i := 0; i < int(record.NumRows()); i++ {
			recordKey := keyCol.GetOneForMarshal(i)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			if k == key {
				// Found matching key, extract vector
				listArray := vectorCol.(*array.List)
				valueArray := listArray.ListValues().(*array.Float32)

				start := int(listArray.Offsets()[i])
				end := int(listArray.Offsets()[i+1])

				vector := make([]float32, end-start)
				for j := start; j < end; j++ {
					vector[j-start] = valueArray.Value(j)
				}

				return vector, nil
			}
		}
	}

	return nil, fmt.Errorf("vector not found for key")
}

// GetVectorsBatch retrieves multiple vectors by key
func (vs *VectorStore[K]) GetVectorsBatch(keys []K) (map[K][]float32, error) {
	result := make(map[K][]float32)
	if len(keys) == 0 {
		return result, nil
	}

	// First, check cache for all keys
	vs.mu.RLock()
	missingKeys := make([]K, 0, len(keys))
	for _, key := range keys {
		if vec, ok := vs.cache[key]; ok {
			result[key] = vec
		} else {
			missingKeys = append(missingKeys, key)
		}
	}
	vs.mu.RUnlock()

	// If all keys were in cache, return immediately
	if len(missingKeys) == 0 {
		return result, nil
	}

	// Try to load all base vectors at once if not already cached
	vs.baseVectorsMu.RLock()
	baseVectorsCached := vs.baseVectorsCached
	vs.baseVectorsMu.RUnlock()

	if !baseVectorsCached {
		vs.baseVectorsLoading.Do(func() {
			// Load all base vectors into cache
			baseVectors, err := vs.getAllVectors()
			if err == nil {
				vs.baseVectorsMu.Lock()
				vs.baseVectorsCache = baseVectors
				vs.baseVectorsCached = true
				vs.baseVectorsMu.Unlock()
			}
		})
	}

	// Get base vectors from cache if available
	var baseVectors map[K][]float32
	vs.baseVectorsMu.RLock()
	if vs.baseVectorsCached {
		baseVectors = vs.baseVectorsCache
	}
	vs.baseVectorsMu.RUnlock()

	// If base vectors are not cached, get them from Parquet file
	if baseVectors == nil {
		var err error
		baseVectors, err = vs.getVectorsBatchFromParquet(missingKeys)
		if err != nil && err.Error() != "vector file does not exist" {
			return result, err
		}
	}

	// Use a mutex to protect concurrent writes to the result map
	var resultMu sync.Mutex

	// Process keys in parallel
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 8) // Limit concurrency to 8 goroutines

	for _, key := range missingKeys {
		wg.Add(1)
		semaphore <- struct{}{} // Acquire semaphore

		go func(key K) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release semaphore

			baseVector, hasBase := baseVectors[key]

			// Check incremental store for each key
			vector, found, err := vs.incremental.GetVector(key, baseVector)
			if err != nil {
				return // Skip this key on error
			}

			if found {
				// If vector was deleted, skip it
				if vector == nil {
					return
				}

				// Add to result and cache
				resultMu.Lock()
				result[key] = vector
				resultMu.Unlock()

				vs.mu.Lock()
				vs.cache[key] = vector
				vs.mu.Unlock()
			} else if hasBase {
				// Use base vector if no incremental changes
				resultMu.Lock()
				result[key] = baseVector
				resultMu.Unlock()

				vs.mu.Lock()
				vs.cache[key] = baseVector
				vs.mu.Unlock()
			}
		}(key)
	}

	wg.Wait() // Wait for all goroutines to complete

	return result, nil
}

// getVectorsBatchFromParquet retrieves multiple vectors directly from the Parquet file
func (vs *VectorStore[K]) getVectorsBatchFromParquet(keys []K) (map[K][]float32, error) {
	result := make(map[K][]float32)

	// Check if file exists
	if _, err := os.Stat(vs.storage.vectorsFile); os.IsNotExist(err) {
		return result, fmt.Errorf("vector file does not exist")
	}

	// Create a set of missing keys for faster lookup
	missingKeySet := make(map[K]bool, len(keys))
	for _, key := range keys {
		missingKeySet[key] = true
	}

	// Open Parquet file
	reader, err := file.OpenParquetFile(vs.storage.vectorsFile, vs.storage.config.MemoryMap)
	if err != nil {
		return result, fmt.Errorf("failed to open Parquet file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		vs.storage.createArrowReadProperties(),
		vs.storage.alloc,
	)
	if err != nil {
		return result, fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read all records
	ctx := context.Background()
	recordReader, err := arrowReader.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return result, fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	for recordReader.Next() {
		record := recordReader.Record()

		// Get key and vector columns
		keyCol := record.Column(0)
		vectorCol := record.Column(1)

		// Iterate through records to find matching keys
		for i := 0; i < int(record.NumRows()); i++ {
			recordKey := keyCol.GetOneForMarshal(i)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			// Check if this is one of our missing keys
			if missingKeySet[k] {
				// Extract vector
				listArray := vectorCol.(*array.List)
				valueArray := listArray.ListValues().(*array.Float32)

				start := int(listArray.Offsets()[i])
				end := int(listArray.Offsets()[i+1])

				vector := make([]float32, end-start)
				for j := start; j < end; j++ {
					vector[j-start] = valueArray.Value(j)
				}

				// Add to result
				result[k] = vector

				// Remove from missing set
				delete(missingKeySet, k)

				// If we found all missing keys, we can stop
				if len(missingKeySet) == 0 {
					break
				}
			}
		}

		// If we found all missing keys, we can stop
		if len(missingKeySet) == 0 {
			break
		}
	}

	return result, nil
}

// DeleteVector removes a vector from storage
func (vs *VectorStore[K]) DeleteVector(key K) error {
	// First, remove from cache under lock
	vs.mu.Lock()
	delete(vs.cache, key)
	delete(vs.pendingWrites, key)
	vs.pendingDeletes[key] = true
	vs.dirty = true

	// Add to incremental store
	err := vs.incremental.AddChange(ChangeTypeDelete, key, nil)

	vs.mu.Unlock()

	return err
}

// writeVectorsToFile writes all vectors to the Parquet file
func (vs *VectorStore[K]) writeVectorsToFile(vectors map[K][]float32) error {
	if len(vectors) == 0 {
		// If no vectors, create an empty file
		return vs.createEmptyFile()
	}

	// Create schema
	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "key", Type: getKeyType[K]()},
			{Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)

	// Create record builder
	recordBuilder := array.NewRecordBuilder(vs.storage.alloc, schema)
	defer recordBuilder.Release()

	keyBuilder := recordBuilder.Field(0)
	vectorBuilder := recordBuilder.Field(1).(*array.ListBuilder)
	valueBuilder := vectorBuilder.ValueBuilder().(*array.Float32Builder)

	// Add data
	for k, v := range vectors {
		// Add key based on its type
		appendToBuilder(keyBuilder, k)

		// Add vector
		vectorBuilder.Append(true)
		for _, val := range v {
			valueBuilder.Append(val)
		}
	}

	// Create record
	record := recordBuilder.NewRecord()
	defer record.Release()

	// Create file
	file, err := os.Create(vs.storage.vectorsFile)
	if err != nil {
		return fmt.Errorf("failed to create vectors file: %w", err)
	}
	defer file.Close()

	// Write record
	arrowWriter, err := pqarrow.NewFileWriter(
		schema,
		file,
		vs.storage.createWriterProperties(),
		vs.storage.createArrowWriterProperties(),
	)
	if err != nil {
		return fmt.Errorf("failed to create Arrow writer: %w", err)
	}

	if err := arrowWriter.Write(record); err != nil {
		arrowWriter.Close()
		return fmt.Errorf("failed to write record: %w", err)
	}

	if err := arrowWriter.Close(); err != nil {
		return fmt.Errorf("failed to close Arrow writer: %w", err)
	}

	return nil
}

// createEmptyFile creates an empty Parquet file
func (vs *VectorStore[K]) createEmptyFile() error {
	// Create schema
	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "key", Type: getKeyType[K]()},
			{Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)

	// Create record builder
	recordBuilder := array.NewRecordBuilder(vs.storage.alloc, schema)
	defer recordBuilder.Release()

	// Create empty record
	record := recordBuilder.NewRecord()
	defer record.Release()

	// Create file
	file, err := os.Create(vs.storage.vectorsFile)
	if err != nil {
		return fmt.Errorf("failed to create vectors file: %w", err)
	}
	defer file.Close()

	// Write record
	arrowWriter, err := pqarrow.NewFileWriter(
		schema,
		file,
		vs.storage.createWriterProperties(),
		vs.storage.createArrowWriterProperties(),
	)
	if err != nil {
		return fmt.Errorf("failed to create Arrow writer: %w", err)
	}

	if err := arrowWriter.Write(record); err != nil {
		arrowWriter.Close()
		return fmt.Errorf("failed to write record: %w", err)
	}

	if err := arrowWriter.Close(); err != nil {
		return fmt.Errorf("failed to close Arrow writer: %w", err)
	}

	return nil
}

// getAllVectors reads all vectors from the Parquet file
func (vs *VectorStore[K]) getAllVectors() (map[K][]float32, error) {
	vectors := make(map[K][]float32)

	// Check if file exists
	if _, err := os.Stat(vs.storage.vectorsFile); os.IsNotExist(err) {
		return vectors, nil
	}

	ctx := context.Background()

	// Open Parquet file
	reader, err := file.OpenParquetFile(vs.storage.vectorsFile, vs.storage.config.MemoryMap)
	if err != nil {
		return nil, fmt.Errorf("failed to open Parquet file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		vs.storage.createArrowReadProperties(),
		vs.storage.alloc,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read all records
	recordReader, err := arrowReader.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	// Copy cache to avoid holding lock during file read
	vs.mu.RLock()
	for k, v := range vs.cache {
		vectors[k] = v
	}
	vs.mu.RUnlock()

	for recordReader.Next() {
		record := recordReader.Record()

		// Get key and vector columns
		keyCol := record.Column(0)
		vectorCol := record.Column(1)

		// Iterate through records
		for i := 0; i < int(record.NumRows()); i++ {
			recordKey := keyCol.GetOneForMarshal(i)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			// Extract vector
			listArray := vectorCol.(*array.List)
			valueArray := listArray.ListValues().(*array.Float32)

			start := int(listArray.Offsets()[i])
			end := int(listArray.Offsets()[i+1])

			vector := make([]float32, end-start)
			for j := start; j < end; j++ {
				vector[j-start] = valueArray.Value(j)
			}

			vectors[k] = vector
		}
	}

	return vectors, nil
}

// Close releases resources used by the vector store
func (vs *VectorStore[K]) Close() error {
	// Flush any pending changes
	if err := vs.Flush(); err != nil {
		return fmt.Errorf("failed to flush pending changes: %w", err)
	}

	// Close incremental store
	if err := vs.incremental.Close(); err != nil {
		return fmt.Errorf("failed to close incremental store: %w", err)
	}

	return nil
}
