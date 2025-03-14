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

// vectorPosition represents the position of a vector in the Arrow file
type vectorPosition struct {
	recordIndex int // Index of the record batch
	rowIndex    int // Index of the row within the record batch
}

// VectorStore handles storage and retrieval of vectors in Arrow columnar format
type VectorStore[K cmp.Ordered] struct {
	storage *ArrowStorage[K]
	dims    int
	mu      sync.RWMutex
	cache   map[K][]float32 // In-memory cache for frequently accessed vectors

	// Position cache for faster retrieval
	positionCache     sync.Map // Map[K]vectorPosition
	positionCacheOnce sync.Once
	positionCacheInit bool

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
		storage:           storage,
		dims:              dims,
		cache:             make(map[K][]float32, 10000), // Cache size of 10,000 vectors
		pendingWrites:     make(map[K][]float32),
		pendingDeletes:    make(map[K]bool),
		flushInterval:     5 * time.Minute, // Default flush interval
		lastFlushTime:     time.Now(),
		maxPendingWrites:  1000, // Default max pending writes before forced flush
		workerPool:        make(chan struct{}, storage.config.NumWorkers),
		stopChan:          make(chan struct{}),
		positionCacheInit: false,
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

	// Remove from position cache
	vs.positionCache.Delete(key)

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
		vs.positionCache.Delete(key)
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

	// Initialize position cache if needed
	vs.positionCacheOnce.Do(func() {
		if err := vs.initPositionCache(); err != nil {
			fmt.Printf("Failed to initialize position cache: %v\n", err)
		}
	})

	// Check if we have the position cached
	if pos, ok := vs.getPositionFromCache(key); ok {
		vector, err := vs.getVectorFromPosition(key, pos)
		if err == nil {
			// Add to cache
			vs.mu.Lock()
			vs.cache[key] = vector
			vs.mu.Unlock()
			return vector, nil
		}
	}

	// Not in position cache or error getting from position, try to load from disk
	vector, err := vs.scanFileForVector(key)
	if err != nil {
		return nil, err
	}

	// Add to cache
	vs.mu.Lock()
	vs.cache[key] = vector
	vs.mu.Unlock()

	return vector, nil
}

// getPositionFromCache gets the position of a vector from the position cache
func (vs *VectorStore[K]) getPositionFromCache(key K) (vectorPosition, bool) {
	if posInterface, ok := vs.positionCache.Load(key); ok {
		if pos, ok := posInterface.(vectorPosition); ok {
			return pos, true
		}
	}
	return vectorPosition{}, false
}

// initPositionCache initializes the position cache by scanning the Arrow file once
func (vs *VectorStore[K]) initPositionCache() error {
	fmt.Println("Initializing vector position cache...")

	// Check if file exists
	if _, err := os.Stat(vs.storage.vectorsFile); os.IsNotExist(err) {
		vs.positionCacheInit = true
		return nil
	}

	// Open Arrow file
	file, err := os.Open(vs.storage.vectorsFile)
	if err != nil {
		return fmt.Errorf("failed to open Arrow file: %w", err)
	}
	defer file.Close()

	// Create Arrow reader
	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(vs.storage.alloc))
	if err != nil {
		return fmt.Errorf("failed to create Arrow reader: %w", err)
	}
	defer reader.Close()

	// Read all records and build position cache
	for i := 0; i < reader.NumRecords(); i++ {
		record, err := reader.Record(i)
		if err != nil {
			return fmt.Errorf("failed to read record: %w", err)
		}

		// Get key column
		keyCol := record.Column(0)

		// Iterate through records to build position cache
		for j := 0; j < int(record.NumRows()); j++ {
			recordKey := GetArrayValue(keyCol, j)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			vs.positionCache.Store(k, vectorPosition{
				recordIndex: i,
				rowIndex:    j,
			})
		}

		record.Release()
	}

	vs.positionCacheInit = true
	fmt.Println("Position cache initialized")
	return nil
}

// getVectorFromPosition retrieves a vector from a specific position in the Arrow file
func (vs *VectorStore[K]) getVectorFromPosition(key K, pos vectorPosition) ([]float32, error) {

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

	// Check if record index is valid
	if pos.recordIndex >= reader.NumRecords() {
		return nil, fmt.Errorf("invalid record index: %d", pos.recordIndex)
	}

	// Read the specific record
	record, err := reader.Record(pos.recordIndex)
	if err != nil {
		return nil, fmt.Errorf("failed to read record: %w", err)
	}
	defer record.Release()

	// Check if row index is valid
	if pos.rowIndex >= int(record.NumRows()) {
		return nil, fmt.Errorf("invalid row index: %d", pos.rowIndex)
	}

	// Get key and vector columns
	keyCol := record.Column(0)
	vectorCol := record.Column(1)

	// Verify the key matches
	recordKey := GetArrayValue(keyCol, pos.rowIndex)
	k, err := convertArrowToKey[K](recordKey)
	if err != nil {
		return nil, fmt.Errorf("failed to convert key: %w", err)
	}

	if k != key {
		return nil, fmt.Errorf("key mismatch: expected %v, got %v", key, k)
	}

	// Extract vector
	listArray := vectorCol.(*array.List)
	valueArray := listArray.ListValues().(*array.Float32)

	start := int(listArray.Offsets()[pos.rowIndex])
	end := int(listArray.Offsets()[pos.rowIndex+1])

	vector := make([]float32, end-start)
	for v := start; v < end; v++ {
		vector[v-start] = valueArray.Value(v)
	}

	return vector, nil
}

// scanFileForVector scans the Arrow file for a vector with the given key
func (vs *VectorStore[K]) scanFileForVector(key K) ([]float32, error) {

	// Check if file exists
	if _, err := os.Stat(vs.storage.vectorsFile); os.IsNotExist(err) {
		errMsg := fmt.Sprintf("vector file does not exist for key %v", key)
		return nil, fmt.Errorf("%s", errMsg)
	}

	// Open Arrow file
	file, err := os.Open(vs.storage.vectorsFile)
	if err != nil {
		errMsg := fmt.Sprintf("failed to open Arrow file for key %v: %v", key, err)
		return nil, fmt.Errorf("%s", errMsg)
	}
	defer file.Close()

	// Create Arrow reader
	reader, err := ipc.NewFileReader(file, ipc.WithAllocator(vs.storage.alloc))
	if err != nil {
		errMsg := fmt.Sprintf("failed to create Arrow reader for key %v: %v", key, err)
		return nil, fmt.Errorf("%s", errMsg)
	}
	defer reader.Close()

	// Read all records
	for i := 0; i < reader.NumRecords(); i++ {
		record, err := reader.Record(i)
		if err != nil {
			errMsg := fmt.Sprintf("failed to read record %d for key %v: %v", i, key, err)
			return nil, fmt.Errorf("%s", errMsg)
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

				// Store position in cache
				vs.positionCache.Store(key, vectorPosition{
					recordIndex: i,
					rowIndex:    j,
				})

				return vector, nil
			}
		}
	}

	errMsg := fmt.Sprintf("vector not found for key %v", key)
	return nil, fmt.Errorf("%s", errMsg)
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

	// Initialize position cache if needed
	vs.positionCacheOnce.Do(func() {
		if err := vs.initPositionCache(); err != nil {
			fmt.Printf("Failed to initialize position cache: %v\n", err)
		}
	})

	// Try to get vectors from position cache
	remainingKeys := make([]K, 0, len(missingKeys))
	for _, key := range missingKeys {
		if pos, ok := vs.getPositionFromCache(key); ok {
			vector, err := vs.getVectorFromPosition(key, pos)
			if err == nil {
				// Add to result and cache
				result[key] = vector
				vs.mu.Lock()
				vs.cache[key] = vector
				vs.mu.Unlock()
			} else {
				remainingKeys = append(remainingKeys, key)
			}
		} else {
			remainingKeys = append(remainingKeys, key)
		}
	}

	// If there are still missing keys, scan the file
	if len(remainingKeys) > 0 {
		for _, key := range remainingKeys {
			vector, err := vs.scanFileForVector(key)
			if err == nil {
				// Add to result and cache
				result[key] = vector
				vs.mu.Lock()
				vs.cache[key] = vector
				vs.mu.Unlock()
			}
		}
	}

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

	// Reset position cache
	vs.positionCacheInit = false
	vs.positionCache = sync.Map{}
	vs.positionCacheOnce = sync.Once{}

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
