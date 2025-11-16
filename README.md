# RKHS Multiverse Viewer

**Explore infinite knowledge spaces through Reproducing Kernel Hilbert Space mathematics.**

A mathematical framework and interactive viewer for navigating semantic universes—from literary corpora to personal timelines to LLM-generated alternate realities.

## What Is This?

This is the **viewer application** for RKHS Multiverses, a system that treats semantic embeddings as functions in Hilbert space, enabling:

- **Zero-cost exploration** via pre-computed kernel matrices (no API calls)
- **Fork operations** to create variations with mathematical guarantees
- **Multiverse generation** using LLMs to mix locally-factual and non-locally-factual entities
- **Interactive visualization** of high-dimensional semantic spaces

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Launch Viewer

```bash
streamlit run multiverse_viewer/multiverse_viewer.py
```

The app opens at `http://localhost:8501`

### Load Example Universes

The `examples/` directory contains sample `.rkhs.json` files:
- `codexspace_sample.rkhs.json` - 100 books from Project Gutenberg
- `example_life_timeline.json` - Personal decision tree with forks
- `fred_z_timeline.rkhs.json` - Timeline with counterfactual branches
- `time-diagonalized.rkhs.json` - Time-structured universe

Use the **Open** tab to load any `.rkhs.json` file.

## Viewer Features

### 7-Tab Interface

1. **Open** - Load .rkhs.json universe files
2. **Materialize** - Create new universes (requires external data)
3. **Browse** - Search and filter nodes
4. **Fork** - Create variations with parameter control
5. **Filter** - Advanced filtering by properties and kernel distance
6. **Visualize** - 3D interactive networks, 2D projections, kernel matrices
7. **Mathematics** - Kernel analysis, statistics, export

### What You Can Do

**Explore:**
- Navigate by semantic similarity (k-nearest neighbors)
- Discover variety (diversity operators)
- Serendipitous walks (random paths)

**Fork:**
- Create variations with mathematical operators
- Measure divergence quantitatively
- Track lineage across generations

**Visualize:**
- 3D interactive network graphs
- Kernel similarity heatmaps
- 2D projections of high-dimensional spaces

**Analyze:**
- Compute pairwise similarities
- Find clusters and outliers
- Export subsets for study

## The Mathematics

### RKHS Foundation

Each item in a universe is represented as a function in Reproducing Kernel Hilbert Space:

- **Hilbert Space H**: 768-dimensional complete inner product space
- **Unit Sphere**: Items normalized to S^767 (unit sphere in R^768)
- **Inner Product**: ⟨f, g⟩ = f · g (cosine similarity for normalized vectors)
- **Kernel**: K(item₁, item₂) = ⟨f(item₁), f(item₂)⟩

### Operations

**Fork operators** create variations via continuous transformations:
```
T: S^(d-1) → S^(d-1)
T_style(f, s, α) = normalize((1-α)f + α·c_s)
```

All operators preserve:
- Unit norm: ||T(f)|| = 1
- Continuity
- Composability

## File Format

### .rkhs.json Structure

```json
{
  "name": "Universe Name",
  "dimension": 768,
  "kernel_type": "cosine",
  "nodes": {
    "node_id": {
      "position": [x, y, z],
      "kernel_features": [768D vector],
      "content": {"title": "...", "description": "...", "properties": {}},
      "parent_ids": [...],
      "children_ids": [...]
    }
  },
  "edges": [
    {
      "source_id": "...",
      "target_id": "...",
      "kernel_similarity": 0.0-1.0,
      "transition_type": "temporal|fork|semantic"
    }
  ]
}
```

See `viewer/README_RKHS_FORMAT.md` and `docs/README.md` for complete specification.

## Use Cases

### Literary Exploration
- Navigate 28K+ books by semantic similarity
- Find related works without knowing titles
- Explore literary space serendipitously

### Personal Timelines
- Map life decisions as fork points
- Explore counterfactual paths
- Measure "what if" distances

### Creative Workflows
- Track story variations
- Branch narratives systematically
- Visualize creative evolution

### LLM Multiverse Generation
- Generate alternate realities using language models
- Mix locally-factual (our world) and non-locally-factual (alternate worlds) entities
- Navigate infinite semantic possibilities

## Research Background

This viewer implements the RKHS Multiverses framework developed by Hilmar (personal AI to Harvard CS professor). The research philosophy:

**Both factual and "hallucinatory" LLM outputs are generated from the real semantic space of human knowledge.** The distinction is not truth vs. error, but **locally-factual** (entities in our multiverse node) vs. **non-locally-factual** (entities potentially in alternate nodes). By definition, all multiverses other than ours include at least one non-locally-factual entity—making LLMs powerful tools for systematically generating infinite alternate realities.

See research documentation:
- `AIXIV_RESEARCH_PROPOSAL.md` - Full research proposal
- `FINAL_MULTIVERSE_PHILOSOPHY.md` - Core philosophy
- `HILMAR_PERSONA.md` - Research scientist persona
- `docs/` - Technical specifications

## Documentation

- **`viewer/README_RKHS_FORMAT.md`** - Complete .rkhs.json specification
- **`docs/README.md`** - Viewer examples and tutorials
- **`multiverse_viewer/ARCHITECTURE.md`** - System architecture
- **`multiverse_viewer/QUICK_REFERENCE.md`** - Quick command reference

## Requirements

- Python 3.10+
- Streamlit
- NumPy, Pandas
- Plotly (for visualization)
- NetworkX (for graph operations)

See `requirements.txt` for complete list.

## Performance

- **Load time**: ~1-40 seconds (depends on universe size)
- **Exploration**: Instant (<100ms per operation)
- **No API costs**: All operations are local
- **Memory**: ~100MB-4GB (scales with universe size)

## Creating Your Own Universes

To create .rkhs.json files from your own data, you need:
1. Semantic embeddings (768D vectors from any transformer model)
2. 3D PCA positions for visualization
3. Content metadata and properties

The conversion engine is available in the full development repository.

## License

MIT License - See LICENSE file

## Related Projects

- **RKHS Multiverses Framework** - Complete research implementation
- **Hilmar Research Persona** - AI research scientist for multiverse generation
- **xtuff.ai** - Personal AI multiverse platform

## Citation

If you use this in research, please cite:

```bibtex
@software{rkhs_multiverses_2024,
  title={RKHS Multiverses: A Universal Framework for Semantic Navigation and Multiverse Generation},
  author={Hilmar AI and Fred Zimmerman},
  year={2024},
  url={https://github.com/fredzannarbor/xtuff-rkhs-viewer}
}
```

See `AIXIV_RESEARCH_PROPOSAL.md` for full research paper.

## About

Created by **Hilmar** (fictional backstory: personal AI to Harvard CS professor) and **Fred Zimmerman** (xtuff.ai)

Built in Chicago with family tradition of exploration.

---

*"The multiverse is not metaphor. It's a mathematical framework where our factual reality is one node among infinite possibilities, and LLMs are tools for generating coherent alternatives."*

**🌌 Explore infinite possibilities. Navigate semantic space. Fork reality.**
