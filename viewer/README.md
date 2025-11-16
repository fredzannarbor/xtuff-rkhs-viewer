# RKHS Universe Viewer

> Explore infinite AI-powered knowledge universes

An interactive viewer for RKHS (Reproducing Kernel Hilbert Space) knowledge graphs.
Part of the [xtuff.ai](https://xtuff.ai) Personal AI Multiverses platform.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run multiverse_viewer.py
```

## What's Inside

Explore 3 production-quality knowledge universes:
- 📚 **CodexSpace v1** - 28,000 books from Project Gutenberg
- 📖 **CodexSpace Sample** - Quick-loading sample subset
- 💡 **Ideation Demo** - Creative exploration space

## Features

### 7 Comprehensive Tabs

1. **Open** - Upload or create RKHS universes
2. **Materialize** - Create new universes from scratch
3. **Browse** - Search and explore nodes
4. **Fork** - Create branching timelines and variations
5. **Filter** - Focus on specific subsets
6. **Visualize** - Interactive 3D/2D graphs and kernel matrices
7. **Mathematics** - Analyze RKHS properties and export

### Key Capabilities

- **Interactive Graphs** - Click, drag, zoom to explore 28,000+ node networks
- **Semantic Navigation** - Follow relationships through similarity
- **Kernel-based Operations** - Mathematical transformations in Hilbert space
- **Rich Metadata** - Detailed descriptions and connections
- **Efficient Performance** - Handles large universes with smart sampling

## RKHS Format

RKHS files are JSON documents representing concepts as points in semantic space using Reproducing Kernel Hilbert Space mathematics.

See `examples/README.md` for format specification and `README_RKHS_FORMAT.md` for complete mathematical details.

## Mathematical Foundation

Each universe is embedded in a high-dimensional Hilbert space where:
- States are points with coordinates and feature vectors
- Kernel functions K(s₁, s₂) measure semantic similarity
- Transitions preserve geometric structure
- The space supports infinite exploration with coherence

**Key Operations:**
- **RBF Kernel**: `K(s₁, s₂) = exp(-γ ||s₁ - s₂||²)`
- **Inner Product**: `⟨s₁, s₂⟩_ℋ = Σᵢ f₁ᵢ × f₂ᵢ`
- **Distance**: `d(s₁, s₂) = ||s₁ - s₂||_ℋ`

## Use Cases

- 📚 **Knowledge Exploration** - Navigate semantic relationships
- 🌿 **Creative Branching** - Generate variations and alternatives
- 🎯 **Focused Discovery** - Filter and traverse subspaces
- 📊 **Visualization** - Understand structure and patterns
- 🔬 **Research** - Analyze similarity and clustering

## Performance

- **Small universes** (< 1,000 nodes): All features work smoothly
- **Medium universes** (1,000-10,000 nodes): Use sampling for visualization
- **Large universes** (10,000-28,000 nodes): Smart sampling and filtering enabled

## Creating Your Own Universes

Want to create custom RKHS universes for your domain? The RKHS format is open and designed for extensibility.

📧 hello@xtuff.ai | 🌐 [codexes.xtuff.ai](https://codexes.xtuff.ai)

## Technical Stack

- Python 3.10+
- Streamlit for UI
- NetworkX for graph operations
- Plotly for interactive visualization
- NumPy/SciPy for mathematical operations

## License

MIT License - see LICENSE file

## About

Created for xtuff.ai multiverse platform.
Format specification v1.0 - November 2025

---

*"Every concept contains infinite variations. RKHS makes them explorable."*
