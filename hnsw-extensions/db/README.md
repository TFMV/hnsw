# 🏹 Quiver

> **Note:** Quiver is an experimental vector database. While it's packed with exciting features, it's still finding its feet in the world. Feel free to play with it, but maybe don't bet your production system on it just yet!

## What is Quiver?

Quiver is a Go-based vector database that combines the best of HNSW (Hierarchical Navigable Small World) graphs with other cool search techniques. Think of it as your trusty sidekick for all things vector search!

## Why Choose Quiver?

- **🔍 Smart Search Strategy**: Quiver doesn't just use one search method - it combines HNSW with exact search, LSH, and data partitioning to find the best results
- **🏷️ Rich Metadata**: Attach JSON metadata to your vectors and filter search results based on it
- **🧩 Faceted Search**: Filter results based on attributes (we call them facets)
- **👎 Negative Examples**: Tell Quiver what you don't want to see in your results
- **📊 Analytics**: Peek under the hood with graph quality and performance metrics
- **💾 Durability**: Your data stays safe with Parquet-based storage
- **📦 Backup & Restore**: Create snapshots of your database and bring them back when needed
- **⚡ Batch Operations**: Add, update, and delete vectors in batches for lightning speed
- **🔗 Fluent Query API**: Write queries that read like plain English
- **🚀 Performance**: Quiver is built for speed without sacrificing accuracy
- **😌 Easy to Use**: A well-thought-out API that just makes sense

## Extensions

Quiver is built on a modular architecture with powerful extensions that enhance its capabilities:

### 🔀 Hybrid Extension

The hybrid extension is the secret sauce behind Quiver's intelligent search strategy. It dynamically selects the optimal search method based on your data and query characteristics:

#### Features

- **Adaptive Strategy Selection**: Automatically chooses between HNSW, exact search, and LSH based on dataset size, dimensionality, and query patterns
- **Learning from Queries**: Improves performance over time by analyzing query patterns and results
- **Strategy Composition**: Combines multiple strategies for better results (e.g., using LSH to pre-filter candidates before HNSW search)
- **Configurable Thresholds**: Fine-tune when to use each strategy based on your specific needs

### 📊 Parquet Extension

The Parquet extension provides durable, efficient storage for your vectors and graph structure using the Apache Parquet columnar format:

#### Features

- **Columnar Storage**: Efficient storage and retrieval of vectors with Apache Parquet
- **Memory Mapping**: Fast access to vectors without loading the entire dataset into memory
- **Compression**: Reduces storage requirements with configurable compression codecs
- **Batch Processing**: Optimized for reading and writing vectors in batches
- **Schema Evolution**: Supports changes to your data structure over time

## Tips for Best Performance

1. **Choose the Right Index Type**:
   - Small dataset (<1000 vectors)? Use `ExactIndexType` for perfect recall
   - Medium dataset? Use `HNSWIndexType` for a good balance
   - Large dataset? Use `HybridIndexType` for optimal performance

2. **Tune Your EfSearch Parameter**:
   - Higher values = more accurate but slower
   - Lower values = faster but potentially less accurate
   - Start with the default and adjust based on your needs

3. **Use Batch Operations**:
   - Always prefer `BatchAdd` over multiple `Add` calls
   - Same goes for `BatchDelete` vs multiple `Delete` calls
   - Run `OptimizeStorage` after large batch operations

4. **Be Smart About Backups**:
   - Schedule backups during quiet times
   - Keep backups in a separate location
   - Test your restore process regularly

5. **Delete with Caution**:
   - Deleting vectors can degrade graph quality
   - Consider marking vectors as inactive instead of deleting them
   - If you must delete, run `OptimizeStorage` afterward

## What's Coming Next?

Quiver is just getting started! Here's what we're working on:

1. **Distributed Search**: Search across multiple nodes
2. **Query Caching**: Speed up repeated queries
3. **Incremental Updates**: Update your index more efficiently
4. **Advanced Filtering**: More powerful filtering capabilities
5. **Vector Compression**: Reduce memory usage
6. **Cloud Storage**: Store your vectors in the cloud
7. **Incremental Backups**: Save space with smarter backups
8. **Streaming Updates**: Update your index in real-time

## License

Quiver is licensed under the MIT License - see the LICENSE file for details.

Happy vector searching! 🏹
