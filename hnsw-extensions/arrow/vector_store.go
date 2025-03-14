package arrow

import (
	"cmp"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
)

// VectorStore handles storage and retrieval of vectors in Arrow columnar format
type VectorStore[K cmp.Ordered] struct {
	storage *ArrowStorage[K]
	dims    int
	mu      sync.RWMutex
	cache   map[K][]float32 // In-memory cache for frequently accessed vectors

	// For lazy updates
	dirty            bool
	pendingWrites    map[K][]float32
	pendingDeletes   map[K]bool
	flushInterval    time.Duration
	lastFlushTime    time.Time
	maxPendingWrites int
	flushInProgress  bool

	// For parallel operations
	workerPool chan struct{}

	// For clean shutdown
	stopChan chan struct{}
}

// NewVectorStore creates a new vector store
func NewVectorStore[K cmp.Ordered](storage *ArrowStorage[K], dims int) *VectorStore[K] {
	vs := &VectorStore[K]{
		storage:          storage,
		dims:             dims,
		cache:            make(map[K][]float32, 10000), // Cache size of 10,000 vectors
		pendingWrites:    make(map[K][]float32),
		pendingDeletes:   make(map[K]bool),
		flushInterval:    5 * time.Minute, // Default flush interval
		lastFlushTime:    time.Now(),
		maxPendingWrites: 1000, // Default max pending writes before forced flush
		workerPool:       make(chan struct{}, storage.config.NumWorkers),
		stopChan:         make(chan struct{}),
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
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			vs.mu.RLock()
			shouldFlush := vs.dirty && (len(vs.pendingWrites) > 0 || len(vs.pendingDeletes) > 0) &&
				(time.Since(vs.lastFlushTime) > vs.flushInterval || len(vs.pendingWrites) >= vs.maxPendingWrites)
			vs.mu.RUnlock()

			if shouldFlush {
				if err := vs.Flush(); err != nil {
					// Log error but continue
					fmt.Printf("Error flushing vector store: %v\n", err)
				}
			}
		case <-vs.stopChan:
			return // Exit the goroutine when stop signal is received
		}
	}
}

// StoreVector stores a vector in memory and marks it for writing to disk
func (vs *VectorStore[K]) StoreVector(key K, vector []float32) error {
	if len(vector) != vs.dims {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", vs.dims, len(vector))
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()

	// Store in cache
	vs.cache[key] = vector

	// Mark for writing to disk
	vs.pendingWrites[key] = vector
	delete(vs.pendingDeletes, key) // Remove from deletes if it was there
	vs.dirty = true

	// Auto-flush if we have too many pending writes
	if len(vs.pendingWrites) >= vs.maxPendingWrites && !vs.flushInProgress {
		go func() {
			if err := vs.Flush(); err != nil {
				fmt.Printf("Error auto-flushing vector store: %v\n", err)
			}
		}()
	}

	return nil
}

// StoreVectors stores multiple vectors in memory and marks them for writing to disk
func (vs *VectorStore[K]) StoreVectors(vectors map[K][]float32) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	// Validate dimensions
	for k, v := range vectors {
		if len(v) != vs.dims {
			return fmt.Errorf("vector dimension mismatch for key %v: expected %d, got %d", k, vs.dims, len(v))
		}
	}

	// Store in cache and mark for writing
	for k, v := range vectors {
		vs.cache[k] = v
		vs.pendingWrites[k] = v
		delete(vs.pendingDeletes, k)
	}
	vs.dirty = true

	// Auto-flush if we have too many pending writes
	if len(vs.pendingWrites) >= vs.maxPendingWrites && !vs.flushInProgress {
		go func() {
			if err := vs.Flush(); err != nil {
				fmt.Printf("Error auto-flushing vector store: %v\n", err)
			}
		}()
	}

	return nil
}

// DeleteVector marks a vector for deletion
func (vs *VectorStore[K]) DeleteVector(key K) {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	// Remove from cache
	delete(vs.cache, key)

	// Mark for deletion
	vs.pendingDeletes[key] = true
	delete(vs.pendingWrites, key)
	vs.dirty = true
}

// DeleteVectors marks multiple vectors for deletion
func (vs *VectorStore[K]) DeleteVectors(keys []K) {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	// Remove from cache and mark for deletion
	for _, key := range keys {
		delete(vs.cache, key)
		vs.pendingDeletes[key] = true
		delete(vs.pendingWrites, key)
	}
	vs.dirty = true
}

// GetVector retrieves a vector by key
func (vs *VectorStore[K]) GetVector(key K) ([]float32, error) {
	// First check cache
	vs.mu.RLock()
	if vector, ok := vs.cache[key]; ok {
		vs.mu.RUnlock()
		return vector, nil
	}
	vs.mu.RUnlock()

	// Not in cache, try to load from disk
	vector, err := vs.getVectorFromArrow(key)
	if err != nil {
		return nil, err
	}

	// Add to cache
	vs.mu.Lock()
	vs.cache[key] = vector
	vs.mu.Unlock()

	return vector, nil
}

// GetVectorsBatch retrieves multiple vectors by key
func (vs *VectorStore[K]) GetVectorsBatch(keys []K) (map[K][]float32, error) {
	result := make(map[K][]float32)
	missingKeys := make([]K, 0, len(keys))

	// First check cache
	vs.mu.RLock()
	for _, key := range keys {
		if vector, ok := vs.cache[key]; ok {
			result[key] = vector
		} else {
			missingKeys = append(missingKeys, key)
		}
	}
	vs.mu.RUnlock()

	// If all keys were in cache, return early
	if len(missingKeys) == 0 {
		return result, nil
	}

	// Load missing keys from disk
	missingVectors, err := vs.getVectorsBatchFromArrow(missingKeys)
	if err != nil {
		return result, err
	}

	// Add missing vectors to cache and result
	vs.mu.Lock()
	for k, v := range missingVectors {
		vs.cache[k] = v
		result[k] = v
	}
	vs.mu.Unlock()

	return result, nil
}

// Flush writes pending changes to disk
func (vs *VectorStore[K]) Flush() error {
	vs.mu.Lock()
	if !vs.dirty || vs.flushInProgress {
		vs.mu.Unlock()
		return nil
	}

	vs.flushInProgress = true
	pendingWrites := make(map[K][]float32, len(vs.pendingWrites))
	for k, v := range vs.pendingWrites {
		pendingWrites[k] = v
	}
	pendingDeletes := make(map[K]bool, len(vs.pendingDeletes))
	for k, v := range vs.pendingDeletes {
		pendingDeletes[k] = v
	}
	vs.mu.Unlock()

	// Get all existing vectors
	allVectors, err := vs.getAllVectors()
	if err != nil {
		vs.mu.Lock()
		vs.flushInProgress = false
		vs.mu.Unlock()
		return fmt.Errorf("failed to get all vectors: %w", err)
	}

	// Apply pending changes
	for k, v := range pendingWrites {
		allVectors[k] = v
	}
	for k := range pendingDeletes {
		delete(allVectors, k)
	}

	// Write all vectors to disk
	if err := vs.WriteVectorsToFile(allVectors); err != nil {
		vs.mu.Lock()
		vs.flushInProgress = false
		vs.mu.Unlock()
		return fmt.Errorf("failed to write vectors to file: %w", err)
	}

	// Update state
	vs.mu.Lock()
	vs.pendingWrites = make(map[K][]float32)
	vs.pendingDeletes = make(map[K]bool)
	vs.dirty = false
	vs.lastFlushTime = time.Now()
	vs.flushInProgress = false
	vs.mu.Unlock()

	return nil
}

// getVectorFromArrow retrieves a vector from the Arrow file
func (vs *VectorStore[K]) getVectorFromArrow(key K) ([]float32, error) {
	// Check if file exists
	if _, err := os.Stat(vs.storage.vectorsFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("vector file does not exist")
	}

	// Open Arrow file
	file, err := os.Open(vs.storage.vectorsFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open Arrow file: %w", err)
	}
	defer file.Close()

	// Create Arrow reader
	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(vs.storage.alloc))
	if err != nil {
		return nil, fmt.Errorf("failed to create Arrow reader: %w", err)
	}
	defer reader.Close()

	// Read all records (in a real implementation, you'd want to filter by key)
	for i := 0; i < reader.NumRecords(); i++ {
		record, err := reader.Record(i)
		if err != nil {
			return nil, fmt.Errorf("failed to read record: %w", err)
		}
		defer record.Release()

		// Get key and vector columns
		keyCol := record.Column(0)
		vectorCol := record.Column(1)

		// Iterate through records to find matching key
		for j := 0; j < int(record.NumRows()); j++ {
			recordKey := GetArrayValue(keyCol, j)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			if k == key {
				// Found matching key, extract vector
				listArray := vectorCol.(*array.List)
				valueArray := listArray.ListValues().(*array.Float32)

				start := int(listArray.Offsets()[j])
				end := int(listArray.Offsets()[j+1])

				vector := make([]float32, end-start)
				for v := start; v < end; v++ {
					vector[v-start] = valueArray.Value(v)
				}

				return vector, nil
			}
		}
	}

	return nil, fmt.Errorf("vector not found for key")
}

// getVectorsBatchFromArrow retrieves multiple vectors from the Arrow file
func (vs *VectorStore[K]) getVectorsBatchFromArrow(keys []K) (map[K][]float32, error) {
	result := make(map[K][]float32)

	// Check if file exists
	if _, err := os.Stat(vs.storage.vectorsFile); os.IsNotExist(err) {
		return result, nil
	}

	// Create a set of missing keys for faster lookup
	missingKeySet := make(map[K]bool, len(keys))
	for _, key := range keys {
		missingKeySet[key] = true
	}

	// Open Arrow file
	file, err := os.Open(vs.storage.vectorsFile)
	if err != nil {
		return result, fmt.Errorf("failed to open Arrow file: %w", err)
	}
	defer file.Close()

	// Create Arrow reader
	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(vs.storage.alloc))
	if err != nil {
		return result, fmt.Errorf("failed to create Arrow reader: %w", err)
	}
	defer reader.Close()

	// Read all records
	for i := 0; i < reader.NumRecords(); i++ {
		record, err := reader.Record(i)
		if err != nil {
			return result, fmt.Errorf("failed to read record: %w", err)
		}
		defer record.Release()

		// Get key and vector columns
		keyCol := record.Column(0)
		vectorCol := record.Column(1)

		// Iterate through records to find matching keys
		for j := 0; j < int(record.NumRows()); j++ {
			recordKey := GetArrayValue(keyCol, j)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			if missingKeySet[k] {
				// Found matching key, extract vector
				listArray := vectorCol.(*array.List)
				valueArray := listArray.ListValues().(*array.Float32)

				start := int(listArray.Offsets()[j])
				end := int(listArray.Offsets()[j+1])

				vector := make([]float32, end-start)
				for v := start; v < end; v++ {
					vector[v-start] = valueArray.Value(v)
				}

				result[k] = vector
				delete(missingKeySet, k)

				// If we found all missing keys, we can stop
				if len(missingKeySet) == 0 {
					return result, nil
				}
			}
		}
	}

	return result, nil
}

// WriteVectorsToFile writes vectors to the Arrow file
func (vs *VectorStore[K]) WriteVectorsToFile(vectors map[K][]float32) error {
	if len(vectors) == 0 {
		// If no vectors, create an empty file
		return vs.CreateEmptyFile()
	}

	// Create schema
	schema := vs.storage.VectorSchema(vs.dims)

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
	writer, err := ipc.NewFileWriter(file, ipc.WithSchema(schema), ipc.WithAllocator(vs.storage.alloc))
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

// CreateEmptyFile creates an empty Arrow file with the appropriate schema
func (vs *VectorStore[K]) CreateEmptyFile() error {
	// Create schema
	schema := vs.storage.VectorSchema(vs.dims)

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
	writer, err := ipc.NewFileWriter(file, ipc.WithSchema(schema), ipc.WithAllocator(vs.storage.alloc))
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

// getAllVectors reads all vectors from the Arrow file
func (vs *VectorStore[K]) getAllVectors() (map[K][]float32, error) {
	vectors := make(map[K][]float32)

	// Check if file exists
	if _, err := os.Stat(vs.storage.vectorsFile); os.IsNotExist(err) {
		return vectors, nil
	}

	// Open Arrow file
	file, err := os.Open(vs.storage.vectorsFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open Arrow file: %w", err)
	}
	defer file.Close()

	// Create Arrow reader
	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(vs.storage.alloc))
	if err != nil {
		return nil, fmt.Errorf("failed to create Arrow reader: %w", err)
	}
	defer reader.Close()

	// Copy cache to avoid holding lock during file read
	vs.mu.RLock()
	for k, v := range vs.cache {
		vectors[k] = v
	}
	vs.mu.RUnlock()

	// Read all records
	for i := 0; i < reader.NumRecords(); i++ {
		record, err := reader.Record(i)
		if err != nil {
			return nil, fmt.Errorf("failed to read record: %w", err)
		}
		defer record.Release()

		// Get key and vector columns
		keyCol := record.Column(0)
		vectorCol := record.Column(1)

		// Iterate through records
		for j := 0; j < int(record.NumRows()); j++ {
			recordKey := GetArrayValue(keyCol, j)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			// Extract vector
			listArray := vectorCol.(*array.List)
			valueArray := listArray.ListValues().(*array.Float32)

			start := int(listArray.Offsets()[j])
			end := int(listArray.Offsets()[j+1])

			vector := make([]float32, end-start)
			for v := start; v < end; v++ {
				vector[v-start] = valueArray.Value(v)
			}

			vectors[k] = vector
		}
	}

	return vectors, nil
}

// Close releases resources used by the vector store
func (vs *VectorStore[K]) Close() error {
	// Signal the background goroutine to stop
	close(vs.stopChan)

	// Flush any pending changes
	if err := vs.Flush(); err != nil {
		return err
	}

	// Clear cache to free memory
	vs.mu.Lock()
	vs.cache = nil
	vs.pendingWrites = nil
	vs.pendingDeletes = nil
	vs.mu.Unlock()

	return nil
}
