package arrow

import (
	"cmp"
	"fmt"
	"sync"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
)

// ArrowAppender provides functionality to stream Arrow record batches directly into the HNSW graph
// with minimal copying of data, leveraging Arrow's columnar format for efficient vector operations.
type ArrowAppender[K cmp.Ordered] struct {
	index       *ArrowIndex[K]
	keyField    string
	vectorField string
	mu          sync.Mutex
	batchSize   int
}

// AppenderConfig defines configuration options for the Arrow appender
type AppenderConfig struct {
	// Name of the field in the Arrow schema that contains the key
	KeyField string

	// Name of the field in the Arrow schema that contains the vector
	// This should be a list/array field of float32 values
	VectorField string

	// Size of batches for processing
	BatchSize int
}

// DefaultAppenderConfig returns the default configuration for the appender
func DefaultAppenderConfig() AppenderConfig {
	return AppenderConfig{
		KeyField:    "key",
		VectorField: "vector",
		BatchSize:   1000,
	}
}

// NewArrowAppender creates a new appender for streaming Arrow data into the HNSW graph
func NewArrowAppender[K cmp.Ordered](index *ArrowIndex[K], config AppenderConfig) *ArrowAppender[K] {
	if config.KeyField == "" {
		config.KeyField = "key"
	}
	if config.VectorField == "" {
		config.VectorField = "vector"
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 1000
	}

	return &ArrowAppender[K]{
		index:       index,
		keyField:    config.KeyField,
		vectorField: config.VectorField,
		batchSize:   config.BatchSize,
	}
}

// ValidateSchema checks if the provided Arrow schema is compatible with the appender
func (a *ArrowAppender[K]) ValidateSchema(schema *arrow.Schema) error {
	// Check if key field exists
	keyIdx := -1
	for i, field := range schema.Fields() {
		if field.Name == a.keyField {
			keyIdx = i
			break
		}
	}
	if keyIdx == -1 {
		return fmt.Errorf("key field '%s' not found in schema", a.keyField)
	}
	keyField := schema.Field(keyIdx)

	// Check if vector field exists
	vectorIdx := -1
	for i, field := range schema.Fields() {
		if field.Name == a.vectorField {
			vectorIdx = i
			break
		}
	}
	if vectorIdx == -1 {
		return fmt.Errorf("vector field '%s' not found in schema", a.vectorField)
	}
	vectorField := schema.Field(vectorIdx)

	// Check if vector field is a list of float32
	if vectorField.Type.ID() != arrow.LIST {
		return fmt.Errorf("vector field '%s' must be a list type, got %s", a.vectorField, vectorField.Type.Name())
	}

	listType := vectorField.Type.(*arrow.ListType)
	if listType.Elem().ID() != arrow.FLOAT32 {
		return fmt.Errorf("vector field '%s' must contain float32 values, got %s", a.vectorField, listType.Elem().Name())
	}

	// Check if key field type is compatible with K
	var zero K
	switch any(zero).(type) {
	case int:
		if keyField.Type.ID() != arrow.INT64 && keyField.Type.ID() != arrow.INT32 {
			return fmt.Errorf("key field '%s' must be an integer type for int keys", a.keyField)
		}
	case int32:
		if keyField.Type.ID() != arrow.INT32 {
			return fmt.Errorf("key field '%s' must be an int32 type for int32 keys", a.keyField)
		}
	case int64:
		if keyField.Type.ID() != arrow.INT64 {
			return fmt.Errorf("key field '%s' must be an int64 type for int64 keys", a.keyField)
		}
	case uint:
		if keyField.Type.ID() != arrow.UINT64 && keyField.Type.ID() != arrow.UINT32 {
			return fmt.Errorf("key field '%s' must be an unsigned integer type for uint keys", a.keyField)
		}
	case uint32:
		if keyField.Type.ID() != arrow.UINT32 {
			return fmt.Errorf("key field '%s' must be a uint32 type for uint32 keys", a.keyField)
		}
	case uint64:
		if keyField.Type.ID() != arrow.UINT64 {
			return fmt.Errorf("key field '%s' must be a uint64 type for uint64 keys", a.keyField)
		}
	case float32:
		if keyField.Type.ID() != arrow.FLOAT32 {
			return fmt.Errorf("key field '%s' must be a float32 type for float32 keys", a.keyField)
		}
	case float64:
		if keyField.Type.ID() != arrow.FLOAT64 {
			return fmt.Errorf("key field '%s' must be a float64 type for float64 keys", a.keyField)
		}
	case string:
		if keyField.Type.ID() != arrow.STRING {
			return fmt.Errorf("key field '%s' must be a string type for string keys", a.keyField)
		}
	case []byte:
		if keyField.Type.ID() != arrow.BINARY {
			return fmt.Errorf("key field '%s' must be a binary type for []byte keys", a.keyField)
		}
	default:
		return fmt.Errorf("unsupported key type %T", zero)
	}

	return nil
}

// AppendRecord appends a single Arrow record to the HNSW graph
func (a *ArrowAppender[K]) AppendRecord(record arrow.Record) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Validate schema
	if err := a.ValidateSchema(record.Schema()); err != nil {
		return err
	}

	// Get key and vector columns
	keyIdx := record.Schema().FieldIndices(a.keyField)[0]
	vectorIdx := record.Schema().FieldIndices(a.vectorField)[0]

	keyCol := record.Column(keyIdx)
	vectorCol := record.Column(vectorIdx)

	// Process each row
	for i := 0; i < int(record.NumRows()); i++ {
		// Extract key
		keyValue := GetArrayValue(keyCol, i)
		key, err := convertArrowToKey[K](keyValue)
		if err != nil {
			return fmt.Errorf("failed to convert key at row %d: %w", i, err)
		}

		// Extract vector
		vector, err := a.extractVector(vectorCol, i)
		if err != nil {
			return fmt.Errorf("failed to extract vector at row %d: %w", i, err)
		}

		// Add to index
		if err := a.index.Add(key, vector); err != nil {
			return fmt.Errorf("failed to add vector at row %d: %w", i, err)
		}
	}

	return nil
}

// AppendBatch appends an Arrow record batch to the HNSW graph
func (a *ArrowAppender[K]) AppendBatch(batch arrow.Record) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Validate schema
	if err := a.ValidateSchema(batch.Schema()); err != nil {
		return err
	}

	// Get key and vector columns
	keyIdx := batch.Schema().FieldIndices(a.keyField)[0]
	vectorIdx := batch.Schema().FieldIndices(a.vectorField)[0]

	keyCol := batch.Column(keyIdx)
	vectorCol := batch.Column(vectorIdx)

	// Process in batches to avoid excessive memory usage
	numRows := int(batch.NumRows())
	for start := 0; start < numRows; start += a.batchSize {
		end := start + a.batchSize
		if end > numRows {
			end = numRows
		}

		keys := make([]K, end-start)
		vectors := make([][]float32, end-start)

		// Extract keys and vectors
		for i := start; i < end; i++ {
			idx := i - start

			// Extract key
			keyValue := GetArrayValue(keyCol, i)
			key, err := convertArrowToKey[K](keyValue)
			if err != nil {
				return fmt.Errorf("failed to convert key at row %d: %w", i, err)
			}
			keys[idx] = key

			// Extract vector
			vector, err := a.extractVector(vectorCol, i)
			if err != nil {
				return fmt.Errorf("failed to extract vector at row %d: %w", i, err)
			}
			vectors[idx] = vector
		}

		// Add batch to index
		errors := a.index.BatchAdd(keys, vectors)
		for i, err := range errors {
			if err != nil {
				return fmt.Errorf("failed to add vector at batch index %d: %w", i, err)
			}
		}
	}

	return nil
}

// AppendTable appends an Arrow table to the HNSW graph
func (a *ArrowAppender[K]) AppendTable(table arrow.Table) error {
	// Validate schema
	if err := a.ValidateSchema(table.Schema()); err != nil {
		return err
	}

	// Process each chunk/record batch
	for i := 0; i < int(table.NumCols()); i++ {
		chunks := table.Column(i).Data().Chunks()
		for j := 0; j < len(chunks); j++ {
			// Create a record from this chunk
			record := array.NewRecord(table.Schema(), []arrow.Array{chunks[j]}, int64(chunks[j].Len()))
			if err := a.AppendBatch(record); err != nil {
				return fmt.Errorf("failed to append chunk %d: %w", j, err)
			}
		}
	}

	return nil
}

// extractVector extracts a float32 vector from an Arrow array at the given index
func (a *ArrowAppender[K]) extractVector(arr arrow.Array, idx int) ([]float32, error) {
	if arr.IsNull(idx) {
		return nil, fmt.Errorf("vector at index %d is null", idx)
	}

	listArray, ok := arr.(*array.List)
	if !ok {
		return nil, fmt.Errorf("expected list array, got %T", arr)
	}

	start := int(listArray.Offsets()[idx])
	end := int(listArray.Offsets()[idx+1])
	length := end - start

	valueArray, ok := listArray.ListValues().(*array.Float32)
	if !ok {
		return nil, fmt.Errorf("expected float32 array for list values, got %T", listArray.ListValues())
	}

	// Create the vector with zero-copy if possible
	vector := make([]float32, length)
	for i := 0; i < length; i++ {
		vector[i] = valueArray.Value(start + i)
	}

	return vector, nil
}

// StreamRecords processes a channel of Arrow records and adds them to the HNSW graph
// This is useful for streaming data from a source like Arrow Flight or a file reader
func (a *ArrowAppender[K]) StreamRecords(records <-chan arrow.Record) error {
	for record := range records {
		if err := a.AppendBatch(record); err != nil {
			return err
		}
		// Release the record to free memory
		record.Release()
	}
	return nil
}

// StreamRecordsAsync processes a channel of Arrow records asynchronously
// It returns a channel that will receive any errors encountered during processing
func (a *ArrowAppender[K]) StreamRecordsAsync(records <-chan arrow.Record) <-chan error {
	errChan := make(chan error, 1)

	go func() {
		defer close(errChan)

		for record := range records {
			if err := a.AppendBatch(record); err != nil {
				errChan <- err
				// Release the record to free memory
				record.Release()
				return
			}
			// Release the record to free memory
			record.Release()
		}
	}()

	return errChan
}
