package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/vectortypes"
)

// CustomVector represents a domain-specific vector type with additional metadata
type CustomVector struct {
	ID        string
	Embedding []float32
	Category  string
	Score     float64
	Tags      []string
}

func main() {
	fmt.Println("HNSW Optimized Distance Calculation Example")
	fmt.Println("===========================================")

	// Initialize random number generator
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Create sample data
	const (
		numVectors = 10000
		dimension  = 128
		k          = 10 // Number of nearest neighbors to find
	)

	// Generate categories and tags for our custom vectors
	categories := []string{"product", "article", "image", "video", "user"}
	tagSets := [][]string{
		{"tech", "gadget", "review"},
		{"news", "politics", "world"},
		{"nature", "landscape", "wildlife"},
		{"tutorial", "education", "howto"},
		{"profile", "social", "network"},
	}

	// Create custom vectors
	customVectors := make([]CustomVector, numVectors)
	for i := range customVectors {
		// Create random embedding
		embedding := make([]float32, dimension)
		for j := range embedding {
			embedding[j] = rng.Float32()
		}

		// Assign random category and tags
		categoryIdx := rng.Intn(len(categories))

		customVectors[i] = CustomVector{
			ID:        fmt.Sprintf("vec-%d", i),
			Embedding: embedding,
			Category:  categories[categoryIdx],
			Score:     rng.Float64() * 10.0,
			Tags:      tagSets[categoryIdx],
		}
	}

	// Create a query vector
	queryEmbedding := make([]float32, dimension)
	for i := range queryEmbedding {
		queryEmbedding[i] = rng.Float32()
	}
	queryVector := CustomVector{
		ID:        "query",
		Embedding: queryEmbedding,
		Category:  "query",
		Score:     10.0,
		Tags:      []string{"search"},
	}

	fmt.Println("\nApproach 1: Standard HNSW Graph")
	fmt.Println("-------------------------------")
	// Create a standard HNSW graph
	standardGraph := hnsw.NewGraph[string]()
	standardGraph.M = 16
	standardGraph.EfSearch = 100
	standardGraph.Distance = hnsw.CosineDistance

	// Add vectors to the graph
	startTime := time.Now()
	for _, cv := range customVectors {
		node := hnsw.MakeNode(cv.ID, cv.Embedding)
		if err := standardGraph.Add(node); err != nil {
			fmt.Printf("Error adding node: %v\n", err)
		}
	}
	fmt.Printf("Standard graph build time: %v\n", time.Since(startTime))

	// Search using standard approach
	startTime = time.Now()
	results, err := standardGraph.Search(queryEmbedding, k)
	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
	}
	fmt.Printf("Standard search time: %v\n", time.Since(startTime))
	fmt.Printf("Found %d results\n", len(results))

	fmt.Println("\nApproach 2: Using VectorDistance")
	fmt.Println("--------------------------------")
	// Create a custom distance function using VectorDistance
	customDistance := hnsw.NewVectorDistance(
		vectortypes.ContraMap[vectortypes.F32, CustomVector]{
			Surface: vectortypes.CreateSurface(hnsw.ToVectorTypesDistanceFunc(hnsw.CosineDistance)),
			ContraMap: func(cv CustomVector) vectortypes.F32 {
				return cv.Embedding
			},
		},
	)

	// Demonstrate direct distance calculation
	dist := customDistance.Distance(customVectors[0], customVectors[1])
	fmt.Printf("Distance between vector 0 and 1: %f\n", dist)

	// Create a custom graph implementation that uses our optimized distance
	type OptimizedGraph struct {
		vectors            []CustomVector
		distanceCalculator *hnsw.VectorDistance[CustomVector]
	}

	optimizedGraph := OptimizedGraph{
		vectors:            customVectors,
		distanceCalculator: customDistance,
	}

	// Implement a simple search function
	search := func(query CustomVector, k int) []CustomVector {
		// Calculate distances to all vectors
		type Result struct {
			Vector   CustomVector
			Distance float32
		}
		results := make([]Result, len(optimizedGraph.vectors))

		startTime := time.Now()
		for i, v := range optimizedGraph.vectors {
			results[i] = Result{
				Vector:   v,
				Distance: optimizedGraph.distanceCalculator.Distance(query, v),
			}
		}

		// Sort by distance (simple insertion sort for small k)
		for i := 1; i < len(results); i++ {
			j := i
			for j > 0 && results[j-1].Distance > results[j].Distance {
				results[j-1], results[j] = results[j], results[j-1]
				j--
			}
		}

		// Take top k results
		if len(results) > k {
			results = results[:k]
		}

		// Extract just the vectors
		vectors := make([]CustomVector, len(results))
		for i, r := range results {
			vectors[i] = r.Vector
		}

		searchTime := time.Since(startTime)
		fmt.Printf("Optimized search time: %v\n", searchTime)

		return vectors
	}

	// Perform search
	optimizedResults := search(queryVector, k)
	fmt.Printf("Found %d results\n", len(optimizedResults))

	fmt.Println("\nApproach 3: Integration with HNSW Graph")
	fmt.Println("--------------------------------------")
	// Create a standard HNSW graph but use our custom vectors
	// We'll need to map between our custom vectors and HNSW nodes

	// Create a map to store our custom vectors by ID
	vectorMap := make(map[string]CustomVector)
	for _, cv := range customVectors {
		vectorMap[cv.ID] = cv
	}

	// Create a new graph
	integratedGraph := hnsw.NewGraph[string]()
	integratedGraph.M = 16
	integratedGraph.EfSearch = 100
	integratedGraph.Distance = hnsw.CosineDistance

	// Add vectors to the graph
	startTime = time.Now()
	for _, cv := range customVectors {
		node := hnsw.MakeNode(cv.ID, cv.Embedding)
		if err := integratedGraph.Add(node); err != nil {
			fmt.Printf("Error adding node: %v\n", err)
		}
	}
	fmt.Printf("Integrated graph build time: %v\n", time.Since(startTime))

	// Search using the graph
	startTime = time.Now()
	integratedResults, err := integratedGraph.Search(queryEmbedding, k)
	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
	}
	searchTime := time.Since(startTime)
	fmt.Printf("Integrated search time: %v\n", searchTime)

	// Map results back to our custom vectors
	customResults := make([]CustomVector, len(integratedResults))
	for i, result := range integratedResults {
		customResults[i] = vectorMap[result.Key]
	}

	fmt.Printf("Found %d results\n", len(customResults))

	// Print some sample results
	fmt.Println("\nSample Results:")
	for i, result := range customResults[:3] {
		fmt.Printf("Result %d: ID=%s, Category=%s, Score=%.2f, Tags=%v\n",
			i+1, result.ID, result.Category, result.Score, result.Tags)
	}

	fmt.Println("\nBenefits of the Optimized Distance Approach:")
	fmt.Println("1. Type safety: Distance calculations are type-checked at compile time")
	fmt.Println("2. Abstraction: Distance functions can be composed and reused")
	fmt.Println("3. Performance: Reduces indirection and allows for specialized implementations")
	fmt.Println("4. Flexibility: Can be adapted to work with any vector-like type")
	fmt.Println("5. Domain-specific: Can incorporate domain knowledge into distance calculations")
}
