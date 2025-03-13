package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/TFMV/hnsw"
)

// ProductVector represents a product in a recommendation system
type ProductVector struct {
	ID         string
	Embedding  []float32
	Category   string
	Price      float32
	Rating     float32
	Popularity float32
	InStock    bool
}

// WeightedDistanceParams defines weights for different components of the distance function
type WeightedDistanceParams struct {
	// Weight for the vector similarity component (0.0 to 1.0)
	VectorWeight float32

	// Weight for the category matching component (0.0 to 1.0)
	CategoryWeight float32

	// Weight for the price similarity component (0.0 to 1.0)
	PriceWeight float32

	// Weight for the rating component (0.0 to 1.0)
	RatingWeight float32

	// Weight for the popularity component (0.0 to 1.0)
	PopularityWeight float32

	// Penalty for out-of-stock items (added to distance)
	OutOfStockPenalty float32
}

// DefaultWeightedDistanceParams returns default parameters for the weighted distance function
func DefaultWeightedDistanceParams() WeightedDistanceParams {
	return WeightedDistanceParams{
		VectorWeight:      0.6,
		CategoryWeight:    0.1,
		PriceWeight:       0.1,
		RatingWeight:      0.1,
		PopularityWeight:  0.1,
		OutOfStockPenalty: 0.5,
	}
}

// CreateWeightedProductDistance creates a distance function that combines vector similarity with metadata
func CreateWeightedProductDistance(params WeightedDistanceParams) *hnsw.VectorDistance[ProductVector] {
	// Create a base vector distance function
	baseDistance := hnsw.ToVectorTypesDistanceFunc(hnsw.CosineDistance)

	// Create a custom distance function that combines vector similarity with metadata
	customDistanceFunc := func(a, b ProductVector) float32 {
		// Calculate vector similarity component
		vectorDist := baseDistance(a.Embedding, b.Embedding)

		// Calculate category matching component (0 if same category, 1 if different)
		var categoryDist float32
		if a.Category == b.Category {
			categoryDist = 0.0
		} else {
			categoryDist = 1.0
		}

		// Calculate price similarity component (normalized difference)
		priceDiff := a.Price - b.Price
		if priceDiff < 0 {
			priceDiff = -priceDiff
		}
		// Normalize price difference to [0, 1] range (assuming max price difference is 1000)
		priceDist := priceDiff / 1000.0
		if priceDist > 1.0 {
			priceDist = 1.0
		}

		// Calculate rating similarity component (inverse of rating)
		// Higher ratings should result in lower distance
		ratingDist := 1.0 - (b.Rating / 5.0)

		// Calculate popularity component (inverse of popularity)
		// Higher popularity should result in lower distance
		popularityDist := 1.0 - (b.Popularity / 100.0)

		// Apply out-of-stock penalty
		var stockPenalty float32
		if !b.InStock {
			stockPenalty = params.OutOfStockPenalty
		}

		// Combine all components with their respective weights
		weightedDist := params.VectorWeight*vectorDist +
			params.CategoryWeight*categoryDist +
			params.PriceWeight*priceDist +
			params.RatingWeight*ratingDist +
			params.PopularityWeight*popularityDist +
			stockPenalty

		return weightedDist
	}

	// Create a custom surface using our distance function
	customSurface := CustomProductSurface{
		DistFunc: customDistanceFunc,
	}

	// Create and return a VectorDistance with our custom surface
	return hnsw.NewVectorDistance[ProductVector](customSurface)
}

// CustomProductSurface implements the Surface interface for ProductVector
type CustomProductSurface struct {
	DistFunc func(a, b ProductVector) float32
}

// Distance implements the Surface interface
func (s CustomProductSurface) Distance(a, b ProductVector) float32 {
	return s.DistFunc(a, b)
}

func main() {
	fmt.Println("HNSW Custom Distance Function Example")
	fmt.Println("====================================")

	// Initialize random number generator
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Create sample data
	const (
		numProducts = 10000
		dimension   = 128
		k           = 10 // Number of nearest neighbors to find
	)

	// Generate categories for our products
	categories := []string{"electronics", "clothing", "books", "home", "sports"}

	// Create product vectors
	products := make([]ProductVector, numProducts)
	for i := range products {
		// Create random embedding
		embedding := make([]float32, dimension)
		for j := range embedding {
			embedding[j] = rng.Float32()
		}

		// Normalize the embedding (for cosine distance)
		var sum float32
		for j := range embedding {
			sum += embedding[j] * embedding[j]
		}
		norm := float32(1.0 / float32(sum))
		for j := range embedding {
			embedding[j] *= norm
		}

		// Assign random properties
		categoryIdx := rng.Intn(len(categories))

		products[i] = ProductVector{
			ID:         fmt.Sprintf("prod-%d", i),
			Embedding:  embedding,
			Category:   categories[categoryIdx],
			Price:      rng.Float32() * 1000.0,
			Rating:     rng.Float32() * 5.0,
			Popularity: rng.Float32() * 100.0,
			InStock:    rng.Float32() > 0.2, // 80% of products are in stock
		}
	}

	// Create a query product
	queryEmbedding := make([]float32, dimension)
	for i := range queryEmbedding {
		queryEmbedding[i] = rng.Float32()
	}
	// Normalize the query embedding
	var sum float32
	for i := range queryEmbedding {
		sum += queryEmbedding[i] * queryEmbedding[i]
	}
	norm := float32(1.0 / float32(sum))
	for i := range queryEmbedding {
		queryEmbedding[i] *= norm
	}

	queryProduct := ProductVector{
		ID:         "query",
		Embedding:  queryEmbedding,
		Category:   "electronics", // We're looking for electronics
		Price:      500.0,         // Mid-range price
		Rating:     4.5,           // High rating
		Popularity: 80.0,          // Popular
		InStock:    true,
	}

	// Create different parameter sets for comparison
	vectorOnlyParams := WeightedDistanceParams{
		VectorWeight:      1.0,
		CategoryWeight:    0.0,
		PriceWeight:       0.0,
		RatingWeight:      0.0,
		PopularityWeight:  0.0,
		OutOfStockPenalty: 0.0,
	}

	balancedParams := DefaultWeightedDistanceParams()

	metadataHeavyParams := WeightedDistanceParams{
		VectorWeight:      0.3,
		CategoryWeight:    0.2,
		PriceWeight:       0.2,
		RatingWeight:      0.2,
		PopularityWeight:  0.1,
		OutOfStockPenalty: 1.0,
	}

	// Create distance functions with different parameter sets
	vectorOnlyDistance := CreateWeightedProductDistance(vectorOnlyParams)
	balancedDistance := CreateWeightedProductDistance(balancedParams)
	metadataHeavyDistance := CreateWeightedProductDistance(metadataHeavyParams)

	// Function to perform search and print results
	performSearch := func(name string, distanceFunc *hnsw.VectorDistance[ProductVector]) {
		fmt.Printf("\n%s Search Results:\n", name)
		fmt.Printf("---------------------------\n")

		// Calculate distances to all products
		type Result struct {
			Product  ProductVector
			Distance float32
		}
		results := make([]Result, len(products))

		startTime := time.Now()
		for i, p := range products {
			results[i] = Result{
				Product:  p,
				Distance: distanceFunc.Distance(queryProduct, p),
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

		searchTime := time.Since(startTime)
		fmt.Printf("Search time: %v\n", searchTime)

		// Print results
		fmt.Printf("Top %d results:\n", k)
		fmt.Printf("%-10s %-15s %-10s %-10s %-10s %-10s %-10s\n",
			"Rank", "ID", "Category", "Price", "Rating", "Popularity", "In Stock")
		fmt.Printf("%-10s %-15s %-10s %-10s %-10s %-10s %-10s\n",
			"----", "--", "--------", "-----", "------", "----------", "--------")

		for i, r := range results {
			fmt.Printf("%-10d %-15s %-10s $%-9.2f %-10.1f %-10.1f %-10t\n",
				i+1, r.Product.ID, r.Product.Category, r.Product.Price,
				r.Product.Rating, r.Product.Popularity, r.Product.InStock)
		}

		// Calculate statistics
		sameCategory := 0
		inStock := 0
		avgRating := float32(0.0)
		avgPrice := float32(0.0)

		for _, r := range results {
			if r.Product.Category == queryProduct.Category {
				sameCategory++
			}
			if r.Product.InStock {
				inStock++
			}
			avgRating += r.Product.Rating
			avgPrice += r.Product.Price
		}

		avgRating /= float32(len(results))
		avgPrice /= float32(len(results))

		fmt.Printf("\nStatistics:\n")
		fmt.Printf("Same category: %d/%d (%.1f%%)\n", sameCategory, k, float32(sameCategory)/float32(k)*100)
		fmt.Printf("In stock: %d/%d (%.1f%%)\n", inStock, k, float32(inStock)/float32(k)*100)
		fmt.Printf("Average rating: %.1f/5.0\n", avgRating)
		fmt.Printf("Average price: $%.2f\n", avgPrice)
	}

	// Perform searches with different distance functions
	performSearch("Vector-Only", vectorOnlyDistance)
	performSearch("Balanced", balancedDistance)
	performSearch("Metadata-Heavy", metadataHeavyDistance)

	fmt.Println("\nConclusion:")
	fmt.Println("The example demonstrates how the new distance calculation approach allows for:")
	fmt.Println("1. Combining vector similarity with metadata for more relevant results")
	fmt.Println("2. Customizing the importance of different factors based on user preferences")
	fmt.Println("3. Implementing business rules (e.g., penalizing out-of-stock items)")
	fmt.Println("4. Creating domain-specific distance functions without modifying the core library")
}
