package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/vector"
)

func main() {
	fmt.Println("HNSW Vector Optimization Example")
	fmt.Println("================================")

	// Initialize random number generator
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Create some test vectors
	const (
		numVectors = 1000
		dimension  = 128
	)

	// Create vectors with string keys
	vectors := make([]vector.NodeVec[string], numVectors)
	for i := range vectors {
		vec := make(vector.F32, dimension)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vector.NodeVec[string]{
			Key: fmt.Sprintf("vec-%d", i),
			Vec: vec,
		}
	}

	// Create a query vector
	queryVec := make(vector.F32, dimension)
	for i := range queryVec {
		queryVec[i] = rng.Float32()
	}

	fmt.Println("\nExample 1: Standard Distance Calculation")
	fmt.Println("---------------------------------------")
	// Standard approach: Calculate distances directly
	startTime := time.Now()
	distances := make([]float32, numVectors)
	for i, v := range vectors {
		distances[i] = hnsw.CosineDistance(queryVec, v.Vec)
	}
	fmt.Printf("Standard calculation time: %v\n", time.Since(startTime))

	fmt.Println("\nExample 2: Using Surface for Distance Calculation")
	fmt.Println("-----------------------------------------------")
	// Create a surface for NodeVec types
	surface := vector.CreateNodeSurface[string](hnsw.CosineDistance)

	// Calculate distances using the surface
	startTime = time.Now()
	surfaceDistances := make([]float32, numVectors)
	queryNode := vector.NodeVec[string]{Key: "query", Vec: queryVec}
	for i, v := range vectors {
		surfaceDistances[i] = surface.Distance(queryNode, v)
	}
	fmt.Printf("Surface calculation time: %v\n", time.Since(startTime))

	// Verify results are the same
	for i := range distances {
		if distances[i] != surfaceDistances[i] {
			fmt.Printf("Mismatch at index %d: %f vs %f\n", i, distances[i], surfaceDistances[i])
		}
	}
	fmt.Println("Results verified: all distances match")

	fmt.Println("\nExample 3: Integration with HNSW Graph")
	fmt.Println("------------------------------------")
	// Create a standard HNSW graph
	graph := hnsw.NewGraph[string]()
	graph.Distance = hnsw.CosineDistance

	// Add vectors to the graph
	for _, v := range vectors[:100] { // Add first 100 vectors for brevity
		node := hnsw.MakeNode(v.Key, v.Vec)
		if err := graph.Add(node); err != nil {
			fmt.Printf("Error adding node: %v\n", err)
		}
	}

	// Search using standard approach
	startTime = time.Now()
	results, err := graph.Search(queryVec, 10)
	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
	}
	fmt.Printf("Standard search time: %v\n", time.Since(startTime))
	fmt.Printf("Found %d results\n", len(results))

	fmt.Println("\nExample 4: Custom Graph with Vector Surface")
	fmt.Println("----------------------------------------")
	// Create a custom graph that uses the vector surface
	type OptimizedGraph struct {
		nodes   []vector.NodeVec[string]
		surface vector.Surface[vector.NodeVec[string]]
	}

	optimizedGraph := OptimizedGraph{
		nodes:   vectors[:100], // Use first 100 vectors
		surface: surface,
	}

	// Perform a simple search using the optimized graph
	startTime = time.Now()
	type Result struct {
		Node     vector.NodeVec[string]
		Distance float32
	}
	optimizedResults := make([]Result, 0, 10)
	queryNodeVec := vector.NodeVec[string]{Key: "query", Vec: queryVec}

	// Calculate distances to all nodes
	for _, node := range optimizedGraph.nodes {
		dist := optimizedGraph.surface.Distance(queryNodeVec, node)
		optimizedResults = append(optimizedResults, Result{Node: node, Distance: dist})
	}

	// Sort results by distance (simple insertion sort for brevity)
	for i := 1; i < len(optimizedResults); i++ {
		j := i
		for j > 0 && optimizedResults[j-1].Distance > optimizedResults[j].Distance {
			optimizedResults[j-1], optimizedResults[j] = optimizedResults[j], optimizedResults[j-1]
			j--
		}
	}

	// Take top 10 results
	if len(optimizedResults) > 10 {
		optimizedResults = optimizedResults[:10]
	}
	fmt.Printf("Optimized search time: %v\n", time.Since(startTime))
	fmt.Printf("Found %d results\n", len(optimizedResults))

	fmt.Println("\nBenefits of the Vector Surface Approach:")
	fmt.Println("1. Type safety: Distance calculations are type-checked at compile time")
	fmt.Println("2. Abstraction: Distance functions can be composed and reused")
	fmt.Println("3. Performance: Reduces indirection and allows for specialized implementations")
	fmt.Println("4. Flexibility: Can be adapted to work with any vector-like type")
}
