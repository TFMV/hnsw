package parquet

import (
	"cmp"
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
)

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

// convertArrowToKey converts an Arrow value to a key of type K
func convertArrowToKey[K cmp.Ordered](value interface{}) (K, error) {
	var zero K

	// Handle nil values
	if value == nil {
		return zero, fmt.Errorf("nil value cannot be converted to key")
	}

	// Try direct type assertion
	if k, ok := value.(K); ok {
		return k, nil
	}

	// Try conversion based on target type
	switch any(zero).(type) {
	case int:
		switch v := value.(type) {
		case int32:
			return any(int(v)).(K), nil
		case int64:
			return any(int(v)).(K), nil
		case float32:
			return any(int(v)).(K), nil
		case float64:
			return any(int(v)).(K), nil
		case string:
			var i int
			if _, err := fmt.Sscanf(v, "%d", &i); err != nil {
				return zero, fmt.Errorf("cannot convert string %q to int: %w", v, err)
			}
			return any(i).(K), nil
		}
	case int32:
		switch v := value.(type) {
		case int:
			return any(int32(v)).(K), nil
		case int64:
			return any(int32(v)).(K), nil
		case float32:
			return any(int32(v)).(K), nil
		case float64:
			return any(int32(v)).(K), nil
		case string:
			var i int32
			if _, err := fmt.Sscanf(v, "%d", &i); err != nil {
				return zero, fmt.Errorf("cannot convert string %q to int32: %w", v, err)
			}
			return any(i).(K), nil
		}
	case int64:
		switch v := value.(type) {
		case int:
			return any(int64(v)).(K), nil
		case int32:
			return any(int64(v)).(K), nil
		case float32:
			return any(int64(v)).(K), nil
		case float64:
			return any(int64(v)).(K), nil
		case string:
			var i int64
			if _, err := fmt.Sscanf(v, "%d", &i); err != nil {
				return zero, fmt.Errorf("cannot convert string %q to int64: %w", v, err)
			}
			return any(i).(K), nil
		}
	case uint:
		switch v := value.(type) {
		case int:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int %d to uint", v)
			}
			return any(uint(v)).(K), nil
		case int32:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int32 %d to uint", v)
			}
			return any(uint(v)).(K), nil
		case int64:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int64 %d to uint", v)
			}
			return any(uint(v)).(K), nil
		case uint32:
			return any(uint(v)).(K), nil
		case uint64:
			return any(uint(v)).(K), nil
		case string:
			var u uint
			if _, err := fmt.Sscanf(v, "%d", &u); err != nil {
				return zero, fmt.Errorf("cannot convert string %q to uint: %w", v, err)
			}
			return any(u).(K), nil
		}
	case uint32:
		switch v := value.(type) {
		case int:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int %d to uint32", v)
			}
			return any(uint32(v)).(K), nil
		case int32:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int32 %d to uint32", v)
			}
			return any(uint32(v)).(K), nil
		case int64:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int64 %d to uint32", v)
			}
			return any(uint32(v)).(K), nil
		case uint:
			return any(uint32(v)).(K), nil
		case uint64:
			return any(uint32(v)).(K), nil
		case string:
			var u uint32
			if _, err := fmt.Sscanf(v, "%d", &u); err != nil {
				return zero, fmt.Errorf("cannot convert string %q to uint32: %w", v, err)
			}
			return any(u).(K), nil
		}
	case uint64:
		switch v := value.(type) {
		case int:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int %d to uint64", v)
			}
			return any(uint64(v)).(K), nil
		case int32:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int32 %d to uint64", v)
			}
			return any(uint64(v)).(K), nil
		case int64:
			if v < 0 {
				return zero, fmt.Errorf("cannot convert negative int64 %d to uint64", v)
			}
			return any(uint64(v)).(K), nil
		case uint:
			return any(uint64(v)).(K), nil
		case uint32:
			return any(uint64(v)).(K), nil
		case string:
			var u uint64
			if _, err := fmt.Sscanf(v, "%d", &u); err != nil {
				return zero, fmt.Errorf("cannot convert string %q to uint64: %w", v, err)
			}
			return any(u).(K), nil
		}
	case float32:
		switch v := value.(type) {
		case int:
			return any(float32(v)).(K), nil
		case int32:
			return any(float32(v)).(K), nil
		case int64:
			return any(float32(v)).(K), nil
		case float64:
			return any(float32(v)).(K), nil
		case string:
			var f float32
			if _, err := fmt.Sscanf(v, "%f", &f); err != nil {
				return zero, fmt.Errorf("cannot convert string %q to float32: %w", v, err)
			}
			return any(f).(K), nil
		}
	case float64:
		switch v := value.(type) {
		case int:
			return any(float64(v)).(K), nil
		case int32:
			return any(float64(v)).(K), nil
		case int64:
			return any(float64(v)).(K), nil
		case float32:
			return any(float64(v)).(K), nil
		case string:
			var f float64
			if _, err := fmt.Sscanf(v, "%f", &f); err != nil {
				return zero, fmt.Errorf("cannot convert string %q to float64: %w", v, err)
			}
			return any(f).(K), nil
		}
	case string:
		return any(fmt.Sprintf("%v", value)).(K), nil
	case []byte:
		if s, ok := value.(string); ok {
			return any([]byte(s)).(K), nil
		}
	}

	return zero, fmt.Errorf("cannot convert %v (%T) to %T", value, value, zero)
}

// getArrayValue extracts a value from an Arrow array at the given index
func getArrayValue(arr arrow.Array, i int) interface{} {
	if arr.IsNull(i) {
		return nil
	}

	switch a := arr.(type) {
	case *array.Int32:
		return a.Value(i)
	case *array.Int64:
		return a.Value(i)
	case *array.Uint32:
		return a.Value(i)
	case *array.Uint64:
		return a.Value(i)
	case *array.Float32:
		return a.Value(i)
	case *array.Float64:
		return a.Value(i)
	case *array.String:
		return a.Value(i)
	case *array.Binary:
		return a.Value(i)
	default:
		return nil
	}
}

// convertKeyToArrow converts a key to its Arrow representation
func convertKeyToArrow[K cmp.Ordered](key K) interface{} {
	return key
}
