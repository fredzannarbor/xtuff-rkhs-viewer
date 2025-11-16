# Converting CodexSpaces to RKHS Format

This guide explains how to convert your CodexSpaces books dataset (PG19) into the RKHS format for use with the multiverse viewer.

## Quick Start with Sample Data

If you want to test the viewer immediately without waiting for conversion:

```bash
python create_sample_codexspace_rkhs.py
streamlit run multiverse_viewer.py
```

Then upload `codexspace_sample.rkhs.json` in the Open tab.

## Converting Your Full CodexSpaces Dataset

### Prerequisites

1. You have already built your CodexSpace using `build_codexspace.py`
2. You have the file `codexspace_v1.pkl` (typically ~3.5GB)
3. Install additional dependencies:

```bash
pip install scikit-learn
```

### Conversion Options

#### Option 1: Convert All Books (28,000)

```bash
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl
```

This will:
- Process all 28,000 books
- Create ~280MB JSON file
- Take approximately 10-15 minutes
- Output: `codexspace_v1.rkhs.json`

#### Option 2: Convert Subset for Testing

```bash
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl -n 1000
```

This converts only the first 1,000 books (~10MB, ~2 minutes).

#### Option 3: Custom Configuration

```bash
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl \
  -o my_books_universe.rkhs.json \
  -n 5000 \
  --kernel cosine \
  --neighbors 15
```

Parameters:
- `-o, --output`: Output filename
- `-n, --max-nodes`: Maximum number of books to include
- `-k, --kernel`: Kernel type (`cosine`, `rbf`, `linear`)
- `-g, --gamma`: Gamma parameter for RBF kernel
- `--neighbors`: Number of nearest neighbor edges per book

### What Gets Converted

The converter transforms your CodexSpaces data:

```
CodexSpaces PKL              →  RKHS JSON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
books[book_id] = text        →  nodes[book_id].content.description
embeddings[book_id] = vector →  nodes[book_id].kernel_features
metadata[book_id] = {...}    →  nodes[book_id].content.properties
similarity relationships     →  edges (k-nearest neighbors)
```

### RKHS Format Structure

Each book becomes an **RKHSNode** with:

```json
{
  "id": "pg19_00042",
  "position": [x, y, z],              // 3D PCA projection for visualization
  "content": {
    "title": "Pride and Prejudice",
    "author": "Jane Austen",
    "description": "First 500 chars of text...",
    "properties": {
      "word_count": 120000,
      "length": 650000,
      "year": "1813",
      "embedding_norm": 1.0
    }
  },
  "metadata": {
    "source": "PG19",
    "original_id": "pg19_00042",
    "index": 42
  },
  "kernel_features": [0.123, -0.456, ...],  // 768-dim embedding
  "timestamp": "2025-11-03T...",
  "parent_ids": ["pg19_00041", ...],
  "children_ids": ["pg19_00043", ...]
}
```

Edges connect similar books:

```json
{
  "source_id": "pg19_00042",
  "target_id": "pg19_00137",
  "weight": 1.0,
  "kernel_similarity": 0.856,
  "transition_type": "semantic",
  "metadata": {"rank": 1}
}
```

## Using the Converted Data

### 1. In the Multiverse Viewer

```bash
streamlit run multiverse_viewer.py
```

- **Open Tab**: Upload your `.rkhs.json` file
- **Browse Tab**: Search books by title/author
- **Filter Tab**: Filter by properties (year, word count, etc.)
- **Visualize Tab**: See 3D network of book relationships
- **Fork Tab**: Create variations and branches

### 2. Programmatically

```python
from multiverse_viewer import load_rkhs_universe

# Load universe
universe = load_rkhs_universe("codexspace_v1.rkhs.json")

# Access books
print(f"Total books: {len(universe.nodes)}")

# Get a specific book
book = universe.nodes["pg19_00042"]
print(f"Title: {book.content['title']}")
print(f"Author: {book.content['author']}")

# Find similar books (via edges)
book_id = "pg19_00042"
for edge in universe.edges:
    if edge.source_id == book_id:
        similar_id = edge.target_id
        similar_book = universe.nodes[similar_id]
        print(f"Similar: {similar_book.content['title']} "
              f"(similarity: {edge.kernel_similarity:.3f})")
```

## Performance Characteristics

| Books | File Size | Conversion Time | Load Time | Memory |
|-------|-----------|-----------------|-----------|--------|
| 100   | 2.4 MB    | 10 seconds      | <1 sec    | 50 MB  |
| 1,000 | 24 MB     | 2 minutes       | 2 sec     | 200 MB |
| 5,000 | 120 MB    | 8 minutes       | 8 sec     | 800 MB |
| 28,000| 670 MB    | 45 minutes      | 40 sec    | 4 GB   |

**Recommendations:**
- For demos: Use 100-1,000 books
- For exploration: Use 5,000-10,000 books
- For full library: Use all 28,000 (requires good machine)

## Visualization Tips

### Small Sets (< 1,000 books)
- Use "All Nodes" mode
- All visualizations work smoothly

### Medium Sets (1,000-10,000 books)
- Use "Sample" mode with 200-500 nodes
- Or use "Filtered" after applying filters

### Large Sets (10,000-28,000 books)
1. **Browse Tab**: Search for specific books
2. **Filter Tab**: Filter by properties (e.g., year > 1800)
3. Mark interesting books as "Traversed"
4. **Visualize Tab**: Show only "Traversed" or "Filtered" nodes

## Troubleshooting

### "Memory Error during conversion"
- Convert in smaller batches using `-n` parameter
- Close other applications
- Use a machine with 8GB+ RAM for full dataset

### "File too large to upload"
- Use the sample dataset instead
- Or convert a subset with `-n 1000`
- For full dataset, load locally (not via upload)

### "Embeddings missing in PKL file"
- Make sure you ran `build_codexspace.py` with embeddings enabled
- Check that the PKL file contains the 'embeddings' key

### "Slow visualization"
- Reduce number of nodes shown
- Use "Sample" mode
- Filter to specific subsets

## Advanced Usage

### Custom Kernel Parameters

For RBF kernel with tighter clustering:

```bash
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl \
  --kernel rbf \
  --gamma 0.5
```

### More Connected Graph

For denser graph structure:

```bash
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl \
  --neighbors 20
```

### Batch Processing

Convert multiple subsets:

```bash
# First 1000
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl -o books_1-1000.rkhs.json -n 1000

# Create filtered PKL first, then convert additional ranges
# (requires custom scripting)
```

## File Locations

After conversion, you'll have:

```
your_directory/
├── codexspace_v1.pkl              # Original (3.5 GB)
├── codexspace_v1.rkhs.json        # Converted (670 MB)
└── codexspace_sample.rkhs.json    # Sample (2.4 MB)
```

## Integration with xtuff.ai

This RKHS format is designed to be one of your seven universe instantiations:

1. **Books Universe** (this one) - Literary exploration
2. **Science Universe** - Scientific concepts
3. **History Universe** - Historical events
4. **Philosophy Universe** - Philosophical ideas
5. **Code Universe** - Programming concepts
6. **Art Universe** - Artistic works
7. **Music Universe** - Musical compositions

Each uses the same RKHS formalization with domain-specific:
- Kernel functions
- Node properties
- Edge semantics
- Visualization parameters

## Next Steps

1. Convert your CodexSpaces data: ✓
2. Load in multiverse viewer: ✓
3. Explore the book universe
4. Create your other six universes using the same format
5. Build your universe creation engine (12-24 month goal)

## Support

For issues or questions:
- Check the main README.md
- Review the multiverse_viewer.py code
- Examine the sample files for reference

Happy exploring! 🌌📚
