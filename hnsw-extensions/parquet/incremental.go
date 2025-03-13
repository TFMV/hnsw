package parquet

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

// ChangeType represents the type of change in the incremental log
type ChangeType int

const (
	ChangeTypeAdd ChangeType = iota
	ChangeTypeDelete
)

// Change represents a single change in the incremental log
type Change[K cmp.Ordered] struct {
	Type      ChangeType
	Key       K
	Vector    []float32 // Only used for ChangeTypeAdd
	Timestamp time.Time
}

// IncrementalConfig defines configuration options for incremental updates
type IncrementalConfig struct {
	// Maximum number of changes before compaction
	MaxChanges int

	// Maximum age of changes before compaction
	MaxAge time.Duration
}

// DefaultIncrementalConfig returns the default configuration for incremental updates
func DefaultIncrementalConfig() IncrementalConfig {
	return IncrementalConfig{
		MaxChanges: 1000,
		MaxAge:     1 * time.Hour,
	}
}

// IncrementalStore manages incremental updates for vector storage
type IncrementalStore[K cmp.Ordered] struct {
	storage    *ParquetStorage[K]
	config     IncrementalConfig
	mu         sync.RWMutex
	changes    []Change[K]
	lastChange time.Time
	dims       int

	// Paths for incremental files
	vectorChangesDir string
	currentLogFile   string
	logIndex         int

	// Cache for change log files (sorted by index in descending order)
	logFilesCache     []logFileInfo
	logFilesCacheMu   sync.RWMutex
	logFilesCacheTime time.Time
}

// logFileInfo represents information about a change log file
type logFileInfo struct {
	path  string
	index int
}

// NewIncrementalStore creates a new incremental store
func NewIncrementalStore[K cmp.Ordered](storage *ParquetStorage[K], config IncrementalConfig, dims int) (*IncrementalStore[K], error) {
	// Create directory for incremental changes
	vectorChangesDir := filepath.Join(storage.config.Directory, "vector_changes")
	if err := os.MkdirAll(vectorChangesDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create vector changes directory: %w", err)
	}

	store := &IncrementalStore[K]{
		storage:          storage,
		config:           config,
		changes:          make([]Change[K], 0, config.MaxChanges),
		vectorChangesDir: vectorChangesDir,
		dims:             dims,
	}

	// Find the highest log index
	entries, err := os.ReadDir(vectorChangesDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read vector changes directory: %w", err)
	}

	highestIndex := -1
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		// Parse log index from filename
		name := entry.Name()
		if len(name) < 10 || name[:6] != "vector" || name[len(name)-8:] != ".parquet" {
			continue
		}

		indexStr := name[6 : len(name)-8]
		index, err := strconv.Atoi(indexStr)
		if err != nil {
			continue
		}

		if index > highestIndex {
			highestIndex = index
		}
	}

	// Set log index and current log file
	store.logIndex = highestIndex + 1
	store.currentLogFile = filepath.Join(vectorChangesDir, fmt.Sprintf("vector%06d.parquet", store.logIndex))

	return store, nil
}

// AddChange adds a change to the incremental log
func (is *IncrementalStore[K]) AddChange(changeType ChangeType, key K, vector []float32) error {
	is.mu.Lock()
	defer is.mu.Unlock()

	// Add change to in-memory log
	is.changes = append(is.changes, Change[K]{
		Type:      changeType,
		Key:       key,
		Vector:    vector,
		Timestamp: time.Now(),
	})
	is.lastChange = time.Now()

	// Check if we need to flush changes
	if len(is.changes) >= is.config.MaxChanges {
		return is.flushChanges()
	}

	return nil
}

// flushChanges writes the current changes to a Parquet file
func (is *IncrementalStore[K]) flushChanges() error {
	if len(is.changes) == 0 {
		return nil
	}

	// Create schema
	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "change_type", Type: arrow.PrimitiveTypes.Int32},
			{Name: "key", Type: getKeyType[K]()},
			{Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
			{Name: "timestamp", Type: arrow.FixedWidthTypes.Timestamp_ms},
		},
		nil,
	)

	// Create record builder
	recordBuilder := array.NewRecordBuilder(is.storage.alloc, schema)
	defer recordBuilder.Release()

	changeTypeBuilder := recordBuilder.Field(0).(*array.Int32Builder)
	keyBuilder := recordBuilder.Field(1)
	vectorBuilder := recordBuilder.Field(2).(*array.ListBuilder)
	valueBuilder := vectorBuilder.ValueBuilder().(*array.Float32Builder)
	timestampBuilder := recordBuilder.Field(3).(*array.TimestampBuilder)

	// Add data to builders
	for _, change := range is.changes {
		changeTypeBuilder.Append(int32(change.Type))
		appendToBuilder(keyBuilder, change.Key)

		if change.Type == ChangeTypeAdd && change.Vector != nil {
			vectorBuilder.Append(true)
			for _, val := range change.Vector {
				valueBuilder.Append(val)
			}
		} else {
			vectorBuilder.AppendNull()
		}

		timestampBuilder.Append(arrow.Timestamp(change.Timestamp.UnixMilli()))
	}

	// Create record
	record := recordBuilder.NewRecord()
	defer record.Release()

	// Create file
	file, err := os.Create(is.currentLogFile)
	if err != nil {
		return fmt.Errorf("failed to create vector changes file: %w", err)
	}
	defer file.Close()

	// Write record
	arrowWriter, err := pqarrow.NewFileWriter(
		schema,
		file,
		is.storage.createWriterProperties(),
		is.storage.createArrowWriterProperties(),
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

	// Clear changes and increment log index
	is.changes = make([]Change[K], 0, is.config.MaxChanges)
	is.logIndex++
	is.currentLogFile = filepath.Join(is.vectorChangesDir, fmt.Sprintf("vector%06d.parquet", is.logIndex))

	return nil
}

// GetVector retrieves a vector by key, considering incremental changes
func (is *IncrementalStore[K]) GetVector(key K, baseVector []float32) ([]float32, bool, error) {
	is.mu.RLock()
	defer is.mu.RUnlock()

	// Check in-memory changes first (in reverse order to get the most recent change)
	for i := len(is.changes) - 1; i >= 0; i-- {
		change := is.changes[i]
		if change.Key == key {
			if change.Type == ChangeTypeDelete {
				return nil, true, nil // Vector was deleted
			} else if change.Type == ChangeTypeAdd {
				return change.Vector, true, nil // Vector was added or updated
			}
		}
	}

	// If not found in memory, check on-disk changes
	vector, found, err := is.getVectorFromChangeLogs(key)
	if err != nil {
		return nil, false, err
	}

	if found {
		return vector, true, nil
	}

	// If not found in changes, return the base vector
	return baseVector, false, nil
}

// getVectorFromChangeLogs retrieves a vector from the change logs
func (is *IncrementalStore[K]) getVectorFromChangeLogs(key K) ([]float32, bool, error) {
	// Get or update the cache of log files
	logFiles, err := is.getLogFiles()
	if err != nil {
		return nil, false, err
	}

	// Check each log file in order (already sorted in descending order)
	for _, logFile := range logFiles {
		vector, found, err := is.getVectorFromChangeLog(logFile.path, key)
		if err != nil {
			return nil, false, err
		}

		if found {
			return vector, true, nil
		}
	}

	return nil, false, nil
}

// getLogFiles returns the sorted list of change log files, using cache if available
func (is *IncrementalStore[K]) getLogFiles() ([]logFileInfo, error) {
	is.logFilesCacheMu.RLock()
	cacheAge := time.Since(is.logFilesCacheTime)
	if len(is.logFilesCache) > 0 && cacheAge < 5*time.Second {
		// Use cached result if it's recent
		result := is.logFilesCache
		is.logFilesCacheMu.RUnlock()
		return result, nil
	}
	is.logFilesCacheMu.RUnlock()

	// Need to update the cache
	is.logFilesCacheMu.Lock()
	defer is.logFilesCacheMu.Unlock()

	// Check again in case another goroutine updated the cache while we were waiting
	if len(is.logFilesCache) > 0 && time.Since(is.logFilesCacheTime) < 5*time.Second {
		return is.logFilesCache, nil
	}

	// List all change log files
	entries, err := os.ReadDir(is.vectorChangesDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read vector changes directory: %w", err)
	}

	// Process entries
	logFiles := make([]logFileInfo, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if len(name) < 10 || name[:6] != "vector" || name[len(name)-8:] != ".parquet" {
			continue
		}

		// Parse index
		indexStr := name[6 : len(name)-8]
		index, err := strconv.Atoi(indexStr)
		if err != nil {
			continue
		}

		logFiles = append(logFiles, logFileInfo{
			path:  filepath.Join(is.vectorChangesDir, name),
			index: index,
		})
	}

	// Sort in descending order to check most recent logs first
	sort.Slice(logFiles, func(i, j int) bool {
		return logFiles[i].index > logFiles[j].index
	})

	// Update cache
	is.logFilesCache = logFiles
	is.logFilesCacheTime = time.Now()

	return logFiles, nil
}

// getVectorFromChangeLog retrieves a vector from a specific change log file
func (is *IncrementalStore[K]) getVectorFromChangeLog(logFile string, key K) ([]float32, bool, error) {
	// Check if file exists
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		return nil, false, nil
	}

	// Open Parquet file
	reader, err := file.OpenParquetFile(logFile, is.storage.config.MemoryMap)
	if err != nil {
		return nil, false, fmt.Errorf("failed to open Parquet file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		is.storage.createArrowReadProperties(),
		is.storage.alloc,
	)
	if err != nil {
		return nil, false, fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read all records
	ctx := context.Background()
	recordReader, err := arrowReader.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return nil, false, fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	var latestVector []float32
	var latestTimestamp time.Time
	var found bool
	var isDeleted bool

	// Process all records in a single pass
	for recordReader.Next() {
		record := recordReader.Record()

		// Get columns
		changeTypeCol := record.Column(0).(*array.Int32)
		keyCol := record.Column(1)
		vectorCol := record.Column(2).(*array.List)
		timestampCol := record.Column(3).(*array.Timestamp)

		// Iterate through records
		for i := 0; i < int(record.NumRows()); i++ {
			recordKey := keyCol.GetOneForMarshal(i)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			// Only process the key we're looking for
			if k == key {
				// Get timestamp
				ts := time.Unix(0, int64(timestampCol.Value(i))*int64(time.Millisecond))

				// If this is a more recent change, update the result
				if !found || ts.After(latestTimestamp) {
					changeType := ChangeType(changeTypeCol.Value(i))
					found = true
					latestTimestamp = ts

					if changeType == ChangeTypeDelete {
						isDeleted = true
						latestVector = nil
					} else {
						isDeleted = false
						// Extract vector
						if vectorCol.IsNull(i) {
							latestVector = nil
						} else {
							start := int(vectorCol.Offsets()[i])
							end := int(vectorCol.Offsets()[i+1])
							valueArray := vectorCol.ListValues().(*array.Float32)

							vector := make([]float32, end-start)
							for j := start; j < end; j++ {
								vector[j-start] = valueArray.Value(j)
							}
							latestVector = vector
						}
					}
				}
			}
		}
	}

	if found && isDeleted {
		return nil, true, nil // Vector was deleted
	}

	return latestVector, found, nil
}

// Compact merges all changes into the base vector store and removes change logs
func (is *IncrementalStore[K]) Compact() error {
	is.mu.Lock()
	defer is.mu.Unlock()

	// Flush any pending changes
	if err := is.flushChanges(); err != nil {
		return fmt.Errorf("failed to flush changes: %w", err)
	}

	// Get all vectors from the base store
	baseVectors, err := is.getBaseVectors()
	if err != nil {
		return fmt.Errorf("failed to get base vectors: %w", err)
	}

	// Apply all changes from logs
	if err := is.applyChangesToVectors(baseVectors); err != nil {
		return fmt.Errorf("failed to apply changes: %w", err)
	}

	// Write updated vectors to the base store
	if err := is.writeVectorsToBaseStore(baseVectors); err != nil {
		return fmt.Errorf("failed to write vectors to base store: %w", err)
	}

	// Remove all change logs
	if err := is.removeChangeLogs(); err != nil {
		return fmt.Errorf("failed to remove change logs: %w", err)
	}

	// Reset log index
	is.logIndex = 0
	is.currentLogFile = filepath.Join(is.vectorChangesDir, fmt.Sprintf("vector%06d.parquet", is.logIndex))

	return nil
}

// getBaseVectors retrieves all vectors from the base store
func (is *IncrementalStore[K]) getBaseVectors() (map[K][]float32, error) {
	// Check if file exists
	if _, err := os.Stat(is.storage.vectorsFile); os.IsNotExist(err) {
		return make(map[K][]float32), nil
	}

	// Open Parquet file
	reader, err := file.OpenParquetFile(is.storage.vectorsFile, is.storage.config.MemoryMap)
	if err != nil {
		return nil, fmt.Errorf("failed to open Parquet file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		is.storage.createArrowReadProperties(),
		is.storage.alloc,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read all records
	ctx := context.Background()
	recordReader, err := arrowReader.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	vectors := make(map[K][]float32)

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

// applyChangesToVectors applies all changes from logs to the vectors map
func (is *IncrementalStore[K]) applyChangesToVectors(vectors map[K][]float32) error {
	// List all change log files
	entries, err := os.ReadDir(is.vectorChangesDir)
	if err != nil {
		return fmt.Errorf("failed to read vector changes directory: %w", err)
	}

	// Sort entries by log index (ascending)
	logFiles := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if len(name) < 10 || name[:6] != "vector" || name[len(name)-8:] != ".parquet" {
			continue
		}

		logFiles = append(logFiles, filepath.Join(is.vectorChangesDir, name))
	}

	// Apply changes from each log file
	for _, logFile := range logFiles {
		if err := is.applyChangesFromLog(logFile, vectors); err != nil {
			return fmt.Errorf("failed to apply changes from log %s: %w", logFile, err)
		}
	}

	return nil
}

// applyChangesFromLog applies changes from a specific log file to the vectors map
func (is *IncrementalStore[K]) applyChangesFromLog(logFile string, vectors map[K][]float32) error {
	// Check if file exists
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		return nil
	}

	// Open Parquet file
	reader, err := file.OpenParquetFile(logFile, is.storage.config.MemoryMap)
	if err != nil {
		return fmt.Errorf("failed to open Parquet file: %w", err)
	}
	defer reader.Close()

	// Create Arrow file reader
	arrowReader, err := pqarrow.NewFileReader(
		reader,
		is.storage.createArrowReadProperties(),
		is.storage.alloc,
	)
	if err != nil {
		return fmt.Errorf("failed to create Arrow reader: %w", err)
	}

	// Read all records
	ctx := context.Background()
	recordReader, err := arrowReader.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to get record reader: %w", err)
	}
	defer recordReader.Release()

	for recordReader.Next() {
		record := recordReader.Record()

		// Get columns
		changeTypeCol := record.Column(0).(*array.Int32)
		keyCol := record.Column(1)
		vectorCol := record.Column(2).(*array.List)

		// Iterate through records
		for i := 0; i < int(record.NumRows()); i++ {
			recordKey := keyCol.GetOneForMarshal(i)
			k, err := convertArrowToKey[K](recordKey)
			if err != nil {
				continue
			}

			changeType := ChangeType(changeTypeCol.Value(i))

			if changeType == ChangeTypeDelete {
				// Delete vector
				delete(vectors, k)
			} else if changeType == ChangeTypeAdd {
				// Add or update vector
				if vectorCol.IsNull(i) {
					continue
				}

				start := int(vectorCol.Offsets()[i])
				end := int(vectorCol.Offsets()[i+1])
				valueArray := vectorCol.ListValues().(*array.Float32)

				vector := make([]float32, end-start)
				for j := start; j < end; j++ {
					vector[j-start] = valueArray.Value(j)
				}

				vectors[k] = vector
			}
		}
	}

	return nil
}

// writeVectorsToBaseStore writes all vectors to the base store
func (is *IncrementalStore[K]) writeVectorsToBaseStore(vectors map[K][]float32) error {
	if len(vectors) == 0 {
		// Create an empty file
		return is.createEmptyBaseFile()
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
	recordBuilder := array.NewRecordBuilder(is.storage.alloc, schema)
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
	file, err := os.Create(is.storage.vectorsFile)
	if err != nil {
		return fmt.Errorf("failed to create vectors file: %w", err)
	}
	defer file.Close()

	// Write record
	arrowWriter, err := pqarrow.NewFileWriter(
		schema,
		file,
		is.storage.createWriterProperties(),
		is.storage.createArrowWriterProperties(),
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

// createEmptyBaseFile creates an empty base file
func (is *IncrementalStore[K]) createEmptyBaseFile() error {
	// Create schema
	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "key", Type: getKeyType[K]()},
			{Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)

	// Create record builder
	recordBuilder := array.NewRecordBuilder(is.storage.alloc, schema)
	defer recordBuilder.Release()

	// Create empty record
	record := recordBuilder.NewRecord()
	defer record.Release()

	// Create file
	file, err := os.Create(is.storage.vectorsFile)
	if err != nil {
		return fmt.Errorf("failed to create vectors file: %w", err)
	}
	defer file.Close()

	// Write record
	arrowWriter, err := pqarrow.NewFileWriter(
		schema,
		file,
		is.storage.createWriterProperties(),
		is.storage.createArrowWriterProperties(),
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

// removeChangeLogs removes all change log files
func (is *IncrementalStore[K]) removeChangeLogs() error {
	entries, err := os.ReadDir(is.vectorChangesDir)
	if err != nil {
		return fmt.Errorf("failed to read vector changes directory: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if len(name) < 10 || name[:6] != "vector" || name[len(name)-8:] != ".parquet" {
			continue
		}

		if err := os.Remove(filepath.Join(is.vectorChangesDir, name)); err != nil {
			return fmt.Errorf("failed to remove change log %s: %w", name, err)
		}
	}

	return nil
}

// ShouldCompact checks if compaction should be performed
func (is *IncrementalStore[K]) ShouldCompact() bool {
	is.mu.RLock()
	defer is.mu.RUnlock()

	// Check if there are enough changes
	if is.logIndex > 5 {
		return true
	}

	// Check if changes are old enough
	if !is.lastChange.IsZero() && time.Since(is.lastChange) > is.config.MaxAge {
		return true
	}

	return false
}

// Close flushes any pending changes and releases resources
func (is *IncrementalStore[K]) Close() error {
	is.mu.Lock()
	defer is.mu.Unlock()

	return is.flushChanges()
}
