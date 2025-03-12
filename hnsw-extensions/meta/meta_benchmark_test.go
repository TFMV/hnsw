package meta

import (
	"testing"
	"time"

	"github.com/TFMV/hnsw"
)

// ProductMetadata represents metadata for a product.
type ProductMetadata struct {
	Name        string    `json:"name"`
	Category    string    `json:"category"`
	Price       float64   `json:"price"`
	Tags        []string  `json:"tags"`
	InStock     bool      `json:"inStock"`
	ReleaseDate time.Time `json:"releaseDate"`
}

func BenchmarkMetadataMarshaling(b *testing.B) {
	// Create sample metadata
	metadata := ProductMetadata{
		Name:        "Smartphone X",
		Category:    "Electronics",
		Price:       999.99,
		Tags:        []string{"smartphone", "5G", "camera"},
		InStock:     true,
		ReleaseDate: time.Date(2023, 1, 15, 0, 0, 0, 0, time.UTC),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})
		_, err := NewMetadataNode(node, metadata)
		if err != nil {
			b.Fatalf("Failed to create metadata node: %v", err)
		}
	}
}

func BenchmarkMetadataUnmarshaling(b *testing.B) {
	// Create a metadata node
	node := hnsw.MakeNode(1, []float32{0.1, 0.2, 0.3})
	metadata := ProductMetadata{
		Name:        "Smartphone X",
		Category:    "Electronics",
		Price:       999.99,
		Tags:        []string{"smartphone", "5G", "camera"},
		InStock:     true,
		ReleaseDate: time.Date(2023, 1, 15, 0, 0, 0, 0, time.UTC),
	}
	metadataNode, err := NewMetadataNode(node, metadata)
	if err != nil {
		b.Fatalf("Failed to create metadata node: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var retrievedMetadata ProductMetadata
		err := metadataNode.GetMetadataAs(&retrievedMetadata)
		if err != nil {
			b.Fatalf("Failed to get metadata: %v", err)
		}
	}
}

func BenchmarkMetadataSearch(b *testing.B) {
	// Create a graph and metadata store
	graph := hnsw.NewGraph[int]()
	store := NewMemoryMetadataStore[int]()
	metadataGraph := NewMetadataGraph(graph, store)

	// Add some nodes with metadata
	for i := 0; i < 100; i++ {
		node := hnsw.MakeNode(i, []float32{float32(i) * 0.01, float32(i) * 0.02, float32(i) * 0.03})
		metadata := ProductMetadata{
			Name:        "Product " + string(rune(i+65)),
			Category:    "Category " + string(rune(i%5+65)),
			Price:       float64(i) * 10.0,
			Tags:        []string{"tag1", "tag2", "tag3"},
			InStock:     i%2 == 0,
			ReleaseDate: time.Now().AddDate(0, 0, -i),
		}
		metadataNode, err := NewMetadataNode(node, metadata)
		if err != nil {
			b.Fatalf("Failed to create metadata node: %v", err)
		}
		err = metadataGraph.Add(metadataNode)
		if err != nil {
			b.Fatalf("Failed to add node: %v", err)
		}
	}

	query := []float32{0.5, 0.5, 0.5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results, err := metadataGraph.Search(query, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
		if len(results) == 0 {
			b.Fatal("No results found")
		}
	}
}

func BenchmarkMetadataSearchWithMetadataRetrieval(b *testing.B) {
	// Create a graph and metadata store
	graph := hnsw.NewGraph[int]()
	store := NewMemoryMetadataStore[int]()
	metadataGraph := NewMetadataGraph(graph, store)

	// Add some nodes with metadata
	for i := 0; i < 100; i++ {
		node := hnsw.MakeNode(i, []float32{float32(i) * 0.01, float32(i) * 0.02, float32(i) * 0.03})
		metadata := ProductMetadata{
			Name:        "Product " + string(rune(i+65)),
			Category:    "Category " + string(rune(i%5+65)),
			Price:       float64(i) * 10.0,
			Tags:        []string{"tag1", "tag2", "tag3"},
			InStock:     i%2 == 0,
			ReleaseDate: time.Now().AddDate(0, 0, -i),
		}
		metadataNode, err := NewMetadataNode(node, metadata)
		if err != nil {
			b.Fatalf("Failed to create metadata node: %v", err)
		}
		err = metadataGraph.Add(metadataNode)
		if err != nil {
			b.Fatalf("Failed to add node: %v", err)
		}
	}

	query := []float32{0.5, 0.5, 0.5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results, err := metadataGraph.Search(query, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
		if len(results) == 0 {
			b.Fatal("No results found")
		}

		// Retrieve metadata for each result
		for _, result := range results {
			var metadata ProductMetadata
			err := result.GetMetadataAs(&metadata)
			if err != nil {
				b.Fatalf("Failed to get metadata: %v", err)
			}
		}
	}
}
