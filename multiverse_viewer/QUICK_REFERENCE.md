# CodexSpaces → RKHS Quick Reference

## 📦 What You Have

- **Format Name**: RKHS (Reproducing Kernel Hilbert Space)
- **File Extension**: `.rkhs.json`
- **Source**: CodexSpaces v1 / PG19 dataset
- **Books**: Up to 28,000

## 🚀 Quick Commands

### Test with Sample (Immediate)
```bash
python create_sample_codexspace_rkhs.py
streamlit run multiverse_viewer.py
# Upload: codexspace_sample.rkhs.json
```

### Convert Your Data (One-Time)
```bash
# Full dataset (~45 mins)
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl

# Quick test (1000 books, ~2 mins)
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl -n 1000

# Custom
python codexspaces_to_rkhs_converter.py codexspace_v1.pkl \
  -o my_universe.rkhs.json \
  -n 5000 \
  --kernel cosine \
  --neighbors 10
```

## 📊 Files Created

| File | Size | Books | Purpose |
|------|------|-------|---------|
| `codexspace_sample.rkhs.json` | 2.4 MB | 100 | Demo/testing |
| `codexspace_v1.rkhs.json` (1K) | 24 MB | 1,000 | Quick exploration |
| `codexspace_v1.rkhs.json` (full) | 670 MB | 28,000 | Full library |

## 🎯 Multiverse Viewer Tabs

1. **Open** - Load your .rkhs.json file
2. **Materialize** - Create new universes
3. **Browse** - Search books, view details
4. **Fork** - Create branches/variations
5. **Filter** - By year, author, properties
6. **Visualize** - 3D network, 2D projection
7. **Mathematics** - Kernel analysis, statistics

## 🔍 Key Features

### Visualization Modes
- **3D Network**: Full graph with position/force layouts
- **2D Projection**: Quick dimensional reduction
- **Kernel Matrix**: Heatmap of similarities

### Node Sets
- **All Nodes**: Everything (efficient up to 28K)
- **Sample**: Random subset (200-500 recommended)
- **Filtered**: After applying filters
- **Traversed**: Books you've marked
- **Forked**: Branches you've created

## 💡 Pro Tips

### For Small Sets (< 1,000)
- Use "All Nodes" mode
- Enable all visualizations
- Smooth performance

### For Medium Sets (1,000-10,000)
- Use "Sample" with 200-500 nodes
- Or filter first, then visualize
- Great for exploration

### For Large Sets (10,000-28,000)
1. Browse/search for specific books
2. Apply filters (year, author, etc.)
3. Mark interesting ones as "Traversed"
4. Visualize "Traversed" set only

### Smart Filtering
```
Year range: 1800-1900  →  ~8,000 books
+ Author contains "Dickens"  →  ~20 books
+ Word count > 50,000  →  ~10 books
→ Visualize this focused set!
```

## 🔢 RKHS Mathematics

### Kernel Types
- **Cosine**: K(x,y) = ⟨x,y⟩/(||x||||y||)  [default]
- **RBF**: K(x,y) = exp(-γ||x-y||²)
- **Linear**: K(x,y) = ⟨x,y⟩

### Node Structure
```json
{
  "position": [x, y, z],           // 3D viz coords
  "kernel_features": [768 dims],   // Full embedding
  "content": {
    "title": "Book Title",
    "author": "Author Name",
    "properties": {
      "year": "1813",
      "word_count": 120000
    }
  }
}
```

### Edge Structure
```json
{
  "source_id": "pg19_00042",
  "target_id": "pg19_00137", 
  "kernel_similarity": 0.856,
  "weight": 1.0,
  "transition_type": "semantic"
}
```

## 🎨 Your Seven Universes

This format supports all your xtuff.ai universes:

1. ✅ **Books** (CodexSpaces) - 28K works
2. ⏳ **Science** - Concepts & papers
3. ⏳ **History** - Events & timelines
4. ⏳ **Philosophy** - Ideas & arguments
5. ⏳ **Code** - Projects & patterns
6. ⏳ **Art** - Works & styles
7. ⏳ **Music** - Compositions & genres

Same format, different content!

## 📱 Usage Patterns

### Exploratory Research
1. Load full dataset
2. Search for topic
3. Explore similar books
4. Mark interesting path
5. Fork variations

### Focused Study
1. Filter by year/author
2. Visualize subset
3. Analyze clusters
4. Export findings

### Serendipitous Discovery
1. Random starting point
2. Follow similar edges
3. Mark traversed path
4. Visualize journey

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| File too large | Convert subset: `-n 1000` |
| Slow visualization | Use "Sample" mode (200-500) |
| Memory error | Close apps, use smaller dataset |
| Missing embeddings | Re-run build_codexspace.py |

## 📚 Documentation

- `README.md` - Main viewer documentation
- `CODEXSPACES_CONVERSION_GUIDE.md` - Detailed guide
- This file - Quick reference

## 🔗 Next Steps

1. ✅ Convert CodexSpaces → RKHS
2. ⬜ Load in multiverse viewer
3. ⬜ Explore book universe
4. ⬜ Create 6 more universes
5. ⬜ Build universe creation engine

---

**Format Version**: RKHS v1.0  
**Created**: November 2025  
**For**: xtuff.ai personal AI multiverses
