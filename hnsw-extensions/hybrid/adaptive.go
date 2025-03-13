package hybrid

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/TFMV/hnsw"
)

// QueryMetrics stores performance metrics for a single query
type QueryMetrics struct {
	Strategy      IndexType     // Strategy used for the query
	QueryVector   []float32     // The query vector
	Dimension     int           // Dimensionality of the vector
	K             int           // Number of neighbors requested
	Duration      time.Duration // Time taken to execute the query
	ResultCount   int           // Number of results returned
	Recall        float64       // Recall rate (if ground truth available)
	DistanceStats DistanceStats // Statistics about distances in the results
}

// DistanceStats captures statistics about distances in query results
type DistanceStats struct {
	Min      float32 // Minimum distance
	Max      float32 // Maximum distance
	Mean     float32 // Mean distance
	Variance float32 // Variance of distances
}

// StrategyStats tracks performance metrics for each strategy
type StrategyStats struct {
	TotalQueries      int             // Total number of queries executed with this strategy
	TotalDuration     time.Duration   // Total time spent on queries
	AvgDuration       time.Duration   // Average duration per query
	P95Duration       time.Duration   // 95th percentile duration
	AvgRecall         float64         // Average recall (if available)
	SuccessRate       float64         // Rate of successful queries
	LastUsed          time.Time       // When this strategy was last used
	RecentPerformance []time.Duration // Recent query durations (sliding window)
}

// AdaptiveConfig contains configuration for the adaptive strategy selector
type AdaptiveConfig struct {
	// Window size for moving averages
	WindowSize int

	// Weight factors for the decision function
	LatencyWeight     float64
	RecallWeight      float64
	SuccessRateWeight float64

	// Learning rate for adaptive thresholds
	LearningRate float64

	// Initial thresholds
	InitialExactThreshold int
	InitialDimThreshold   int

	// Exploration vs exploitation trade-off
	ExplorationFactor float64

	// Minimum number of samples before adapting
	MinSamplesForAdaptation int
}

// DefaultAdaptiveConfig returns default configuration for adaptive selection
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		WindowSize:              100,
		LatencyWeight:           0.6,
		RecallWeight:            0.3,
		SuccessRateWeight:       0.1,
		LearningRate:            0.05,
		InitialExactThreshold:   1000,
		InitialDimThreshold:     500,
		ExplorationFactor:       0.1,
		MinSamplesForAdaptation: 20,
	}
}

// indexTypeToString converts IndexType to string
func indexTypeToString(t IndexType) string {
	switch t {
	case ExactIndexType:
		return "Exact"
	case HNSWIndexType:
		return "HNSW"
	case LSHIndexType:
		return "LSH"
	case HybridIndexType:
		return "Hybrid"
	default:
		return "Unknown"
	}
}

// stringToIndexType converts string to IndexType
func stringToIndexType(s string) IndexType {
	switch s {
	case "Exact":
		return ExactIndexType
	case "HNSW":
		return HNSWIndexType
	case "LSH":
		return LSHIndexType
	case "Hybrid":
		return HybridIndexType
	default:
		return IndexType(0)
	}
}

// AdaptiveSelector implements dynamic strategy selection based on runtime metrics
type AdaptiveSelector[K cmp.Ordered] struct {
	// Configuration
	config AdaptiveConfig

	// Underlying indexes
	exactIndex  *ExactIndex[K]
	hnswIndex   *hnsw.Graph[K]
	lshIndex    *LSHIndex[K]
	partitioner *Partitioner[K]

	// Performance metrics for each strategy
	strategyStats map[string]*StrategyStats

	// Recent query metrics (circular buffer)
	recentQueries []QueryMetrics
	currentPos    int

	// Adaptive thresholds
	exactThreshold int // Dataset size threshold for exact search
	dimThreshold   int // Dimensionality threshold for LSH

	// Dataset statistics
	datasetSize  int
	avgDimension int

	// Query pattern detection
	queryDistribution map[string]int // Tracks query clusters

	// Mutex for thread safety
	mu sync.RWMutex
}

// NewAdaptiveSelector creates a new adaptive strategy selector
func NewAdaptiveSelector[K cmp.Ordered](
	exactIndex *ExactIndex[K],
	hnswIndex *hnsw.Graph[K],
	lshIndex *LSHIndex[K],
	partitioner *Partitioner[K],
	config AdaptiveConfig,
) *AdaptiveSelector[K] {
	if config.WindowSize <= 0 {
		config = DefaultAdaptiveConfig()
	}

	selector := &AdaptiveSelector[K]{
		config:            config,
		exactIndex:        exactIndex,
		hnswIndex:         hnswIndex,
		lshIndex:          lshIndex,
		partitioner:       partitioner,
		strategyStats:     make(map[string]*StrategyStats),
		recentQueries:     make([]QueryMetrics, config.WindowSize),
		queryDistribution: make(map[string]int),
		exactThreshold:    config.InitialExactThreshold,
		dimThreshold:      config.InitialDimThreshold,
	}

	// Initialize strategy stats
	for _, strategy := range []IndexType{ExactIndexType, HNSWIndexType, LSHIndexType, HybridIndexType} {
		selector.strategyStats[indexTypeToString(strategy)] = &StrategyStats{
			RecentPerformance: make([]time.Duration, 0, config.WindowSize),
			LastUsed:          time.Now(),
		}
	}

	return selector
}

// UpdateDatasetSize updates the dataset size in the selector
func (s *AdaptiveSelector[K]) UpdateDatasetSize(size int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.datasetSize = size
}

// SelectStrategy chooses the best strategy for a given query
func (s *AdaptiveSelector[K]) SelectStrategy(query []float32, k int) IndexType {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Get query dimension
	dimension := len(query)

	// Basic strategy selection based on current thresholds
	var selectedStrategy IndexType

	// Check if we should explore a random strategy
	if rand.Float64() < s.config.ExplorationFactor {
		// Exploration: try a random strategy occasionally
		strategies := []IndexType{ExactIndexType, HNSWIndexType, LSHIndexType, HybridIndexType}
		selectedStrategy = strategies[rand.Intn(len(strategies))]
	} else {
		// Exploitation: use heuristics to select the best strategy

		// Dataset size-based selection
		if s.datasetSize < s.exactThreshold {
			selectedStrategy = ExactIndexType
		} else if dimension > s.dimThreshold {
			// High-dimensional data
			selectedStrategy = LSHIndexType
		} else {
			// Check if we have query clusters
			if s.hasQueryClusters() {
				selectedStrategy = HybridIndexType // Use partitioning
			} else {
				selectedStrategy = HNSWIndexType
			}
		}

		// Override with performance-based selection if we have enough data
		if s.getTotalQueries() >= s.config.MinSamplesForAdaptation {
			performanceBasedStrategy := s.selectByPerformance()
			// Blend the two strategies (basic heuristic and performance-based)
			// For now, just use the performance-based one if available
			if performanceBasedStrategy != IndexType(0) {
				selectedStrategy = performanceBasedStrategy
			}
		}
	}

	return selectedStrategy
}

// RecordQueryMetrics records metrics for a completed query
func (s *AdaptiveSelector[K]) RecordQueryMetrics(metrics QueryMetrics) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Update recent queries circular buffer
	s.recentQueries[s.currentPos] = metrics
	s.currentPos = (s.currentPos + 1) % s.config.WindowSize

	// Update strategy stats
	strategyKey := indexTypeToString(metrics.Strategy)
	stats, ok := s.strategyStats[strategyKey]
	if !ok {
		stats = &StrategyStats{
			RecentPerformance: make([]time.Duration, 0, s.config.WindowSize),
		}
		s.strategyStats[strategyKey] = stats
	}

	stats.TotalQueries++
	stats.TotalDuration += metrics.Duration
	stats.AvgDuration = time.Duration(int64(stats.TotalDuration) / int64(stats.TotalQueries))
	stats.LastUsed = time.Now()

	// Update recent performance (sliding window)
	if len(stats.RecentPerformance) >= s.config.WindowSize {
		stats.RecentPerformance = stats.RecentPerformance[1:]
	}
	stats.RecentPerformance = append(stats.RecentPerformance, metrics.Duration)

	// Update P95 duration
	if len(stats.RecentPerformance) > 0 {
		durations := make([]time.Duration, len(stats.RecentPerformance))
		copy(durations, stats.RecentPerformance)
		sort.Slice(durations, func(i, j int) bool {
			return durations[i] < durations[j]
		})
		p95Index := int(float64(len(durations)) * 0.95)
		if p95Index >= len(durations) {
			p95Index = len(durations) - 1
		}
		stats.P95Duration = durations[p95Index]
	}

	// Update recall stats if available
	if metrics.Recall > 0 {
		stats.AvgRecall = (stats.AvgRecall*float64(stats.TotalQueries-1) + metrics.Recall) / float64(stats.TotalQueries)
	}

	// Update success rate
	if metrics.ResultCount > 0 {
		successRate := (stats.SuccessRate*float64(stats.TotalQueries-1) + 1.0) / float64(stats.TotalQueries)
		stats.SuccessRate = successRate
	} else {
		successRate := (stats.SuccessRate * float64(stats.TotalQueries-1)) / float64(stats.TotalQueries)
		stats.SuccessRate = successRate
	}

	// Track query distribution (simplified clustering)
	queryHash := s.hashQuery(metrics.QueryVector)
	s.queryDistribution[queryHash]++

	// Update dataset statistics
	s.datasetSize = max(s.datasetSize, metrics.ResultCount)
	s.avgDimension = metrics.Dimension

	// Adapt thresholds if we have enough data
	if s.getTotalQueries() >= s.config.MinSamplesForAdaptation {
		s.adaptThresholds()
	}
}

// adaptThresholds adjusts thresholds based on observed performance
func (s *AdaptiveSelector[K]) adaptThresholds() {
	// Compare exact vs HNSW performance for small datasets
	exactStats := s.strategyStats[indexTypeToString(ExactIndexType)]
	hnswStats := s.strategyStats[indexTypeToString(HNSWIndexType)]

	if exactStats.TotalQueries > 0 && hnswStats.TotalQueries > 0 {
		// If exact search is faster for current threshold, increase it
		if exactStats.AvgDuration < hnswStats.AvgDuration && s.datasetSize >= s.exactThreshold {
			s.exactThreshold = int(float64(s.exactThreshold) * (1 + s.config.LearningRate))
		} else if exactStats.AvgDuration > hnswStats.AvgDuration && s.datasetSize < s.exactThreshold {
			// If HNSW is faster, decrease the threshold
			s.exactThreshold = int(float64(s.exactThreshold) * (1 - s.config.LearningRate))
		}
	}

	// Compare LSH vs HNSW for high-dimensional data
	lshStats := s.strategyStats[indexTypeToString(LSHIndexType)]

	if lshStats.TotalQueries > 0 && hnswStats.TotalQueries > 0 {
		// If LSH is faster for high dimensions, decrease the threshold
		if lshStats.AvgDuration < hnswStats.AvgDuration && s.avgDimension <= s.dimThreshold {
			s.dimThreshold = int(float64(s.dimThreshold) * (1 - s.config.LearningRate))
		} else if lshStats.AvgDuration > hnswStats.AvgDuration && s.avgDimension > s.dimThreshold {
			// If HNSW is faster, increase the threshold
			s.dimThreshold = int(float64(s.dimThreshold) * (1 + s.config.LearningRate))
		}
	}
}

// selectByPerformance chooses the best strategy based on recent performance
func (s *AdaptiveSelector[K]) selectByPerformance() IndexType {
	var bestStrategy IndexType
	var bestScore float64 = -1

	for strategyStr, stats := range s.strategyStats {
		if stats.TotalQueries == 0 {
			continue
		}

		// Calculate a score based on latency, recall, and success rate
		latencyScore := 1.0 / float64(stats.AvgDuration)
		recallScore := stats.AvgRecall
		successScore := stats.SuccessRate

		// Weighted score
		score := s.config.LatencyWeight*latencyScore +
			s.config.RecallWeight*recallScore +
			s.config.SuccessRateWeight*successScore

		if score > bestScore {
			bestScore = score
			bestStrategy = stringToIndexType(strategyStr)
		}
	}

	return bestStrategy
}

// hasQueryClusters determines if queries form clusters
func (s *AdaptiveSelector[K]) hasQueryClusters() bool {
	// Simple heuristic: if a small number of query hashes account for
	// a large percentage of queries, we have clusters
	if len(s.queryDistribution) == 0 {
		return false
	}

	// Count total queries
	totalQueries := 0
	for _, count := range s.queryDistribution {
		totalQueries += count
	}

	// Find the top 3 clusters
	var topClusters []int
	for _, count := range s.queryDistribution {
		topClusters = append(topClusters, count)
	}

	sort.Slice(topClusters, func(i, j int) bool {
		return topClusters[i] > topClusters[j]
	})

	// Take top 3 or fewer
	clusterCount := min(3, len(topClusters))
	topClusterSum := 0
	for i := 0; i < clusterCount; i++ {
		topClusterSum += topClusters[i]
	}

	// If top clusters account for >50% of queries, we have clustering
	return float64(topClusterSum)/float64(totalQueries) > 0.5
}

// hashQuery creates a simple hash for a query vector
// This is a very basic clustering approach
func (s *AdaptiveSelector[K]) hashQuery(query []float32) string {
	// Simplify the vector by rounding to nearest 0.1
	simplified := make([]int, len(query))
	for i, v := range query {
		simplified[i] = int(math.Round(float64(v) * 10))
	}

	// Create a simple string hash
	var hash strings.Builder
	for _, v := range simplified {
		hash.WriteString(fmt.Sprintf("%d,", v))
	}
	return hash.String()
}

// getTotalQueries returns the total number of queries processed
func (s *AdaptiveSelector[K]) getTotalQueries() int {
	total := 0
	for _, stats := range s.strategyStats {
		total += stats.TotalQueries
	}
	return total
}

// GetStats returns the current statistics and thresholds
func (s *AdaptiveSelector[K]) GetStats() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := make(map[string]interface{})

	// Add thresholds
	stats["exact_threshold"] = s.exactThreshold
	stats["dimension_threshold"] = s.dimThreshold

	// Add strategy stats
	strategyStats := make(map[string]interface{})
	for strategy, sstats := range s.strategyStats {
		strategyStats[strategy] = map[string]interface{}{
			"total_queries": sstats.TotalQueries,
			"avg_duration":  sstats.AvgDuration.String(),
			"p95_duration":  sstats.P95Duration.String(),
			"avg_recall":    sstats.AvgRecall,
			"success_rate":  sstats.SuccessRate,
		}
	}
	stats["strategies"] = strategyStats

	// Add query distribution stats
	clusterStats := make(map[string]int)
	for hash, count := range s.queryDistribution {
		if count > 5 { // Only include significant clusters
			clusterStats[hash] = count
		}
	}
	stats["query_clusters"] = clusterStats

	return stats
}

// ResetStats resets all statistics
func (s *AdaptiveSelector[K]) ResetStats() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.strategyStats = make(map[string]*StrategyStats)
	for _, strategy := range []IndexType{ExactIndexType, HNSWIndexType, LSHIndexType, HybridIndexType} {
		s.strategyStats[indexTypeToString(strategy)] = &StrategyStats{
			RecentPerformance: make([]time.Duration, 0, s.config.WindowSize),
			LastUsed:          time.Now(),
		}
	}

	s.recentQueries = make([]QueryMetrics, s.config.WindowSize)
	s.currentPos = 0
	s.queryDistribution = make(map[string]int)

	// Reset thresholds to initial values
	s.exactThreshold = s.config.InitialExactThreshold
	s.dimThreshold = s.config.InitialDimThreshold
}
