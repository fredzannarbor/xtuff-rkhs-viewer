# RKHS Multiverse Viewer

## Format Name: **RKHS** (Reproducing Kernel Hilbert Space)

The RKHS format is a mathematical formalization for representing personal AI multiverses as structured explorable spaces where each state exists in a high-dimensional Hilbert space with kernel-based similarity measures.

## Format Specification

### Core Concept
Each multiverse is embedded in a Reproducing Kernel Hilbert Space (RKHS) where:
- States are points in a high-dimensional space ℋ
- Kernel functions K(s₁, s₂) measure semantic similarity
- Transitions preserve geometric structure
- The space supports infinite exploration while maintaining coherence

### File Format: `.rkhs.json`

```json
{
  "name": "Universe Name",
  "description": "Universe description",
  "dimension": 8,
  "kernel_type": "rbf|linear|polynomial|cosine",
  "kernel_params": {"gamma": 1.0},
  "nodes": {
    "node_id": {
      "id": "unique_identifier",
      "position": [x, y, z],
      "content": {
        "title": "Node Title",
        "description": "Content",
        "properties": {}
      },
      "metadata": {},
      "kernel_features": [f1, f2, ..., fn],
      "timestamp": "ISO8601",
      "parent_ids": [],
      "children_ids": []
    }
  },
  "edges": [
    {
      "source_id": "node_1",
      "target_id": "node_2",
      "weight": 1.0,
      "kernel_similarity": 0.95,
      "transition_type": "temporal|fork|semantic",
      "metadata": {}
    }
  ],
  "metadata": {},
  "version": "1.0"
}
```

### Key Components

**RKHSNode**: Represents a state in the multiverse
- `position`: 3D coordinates for visualization (projection of full space)
- `kernel_features`: Full dimensional feature vector for similarity computation
- `content`: Semantic content and properties
- `parent_ids/children_ids`: Graph structure

**RKHSEdge**: Represents transitions between states
- `kernel_similarity`: Computed K(s₁, s₂) value
- `transition_type`: Nature of the connection
- `weight`: Edge importance

**Kernel Types**:
- **RBF (Radial Basis Function)**: `K(s₁, s₂) = exp(-γ||s₁ - s₂||²)`
- **Linear**: `K(s₁, s₂) = ⟨s₁, s₂⟩`
- **Polynomial**: `K(s₁, s₂) = (⟨s₁, s₂⟩ + 1)^d`
- **Cosine**: `K(s₁, s₂) = ⟨s₁, s₂⟩ / (||s₁|| ||s₂||)`

## Installation

```bash
pip install -r requirements.txt
streamlit run multiverse_viewer.py
```

## Usage

### 1. Open Tab
- Upload existing `.rkhs.json` files
- Create sample universes (10 to 30,000 nodes)

### 2. Materialize Tab
- Create new universes from scratch
- Configure dimension, kernel type, and initial structure

### 3. Browse Tab
- Search and explore nodes
- View detailed node information
- Mark nodes as "traversed" for filtering

### 4. Fork Tab
- Create branching timelines
- Generate alternate states with controlled perturbation
- Build out exploration trees

### 5. Filter Tab
- Filter by traversal status
- Filter by node properties
- Filter by kernel distance from reference nodes
- Create focused subsets

### 6. Visualize Tab
- **3D Network**: Interactive network graph
  - Position-based layout (uses RKHS positions)
  - Force-directed layout (physics simulation)
  - Handles 28,000+ nodes efficiently
  
- **2D Projection**: Simplified view of first two dimensions

- **Kernel Matrix**: Heatmap of pairwise similarities

**Performance Modes**:
- **All Nodes**: Full universe (optimized for up to 28k nodes)
- **Sample**: Random subset for exploration
- **Filtered/Traversed/Forked**: Focused views (efficient for any size)

### 7. Mathematics Tab
- View RKHS properties and statistics
- Compute kernel similarities between states
- Analyze distance distributions
- Export universe files

## Mathematical Foundation

The RKHS formalization provides several advantages:

1. **Continuous Space**: Smooth interpolation between states
2. **Kernel Similarity**: Semantic distance measurement
3. **Infinite Expansion**: Add states while preserving structure
4. **Mathematical Rigor**: Well-defined operations and properties
5. **Dimensionality**: Flexible representation (typically 8-1000 dimensions)

### Key Equations

**Kernel Function** (RBF):
```
K(s₁, s₂) = exp(-γ ||s₁ - s₂||²)
```

**Inner Product**:
```
⟨s₁, s₂⟩_ℋ = Σᵢ f₁ᵢ × f₂ᵢ
```

**Distance**:
```
d(s₁, s₂) = ||s₁ - s₂||_ℋ = √(⟨s₁-s₂, s₁-s₂⟩_ℋ)
```

## Performance Characteristics

- **Small universes** (< 1,000 nodes): All features work smoothly
- **Medium universes** (1,000-10,000 nodes): Use sampling for visualization
- **Large universes** (10,000-28,000 nodes): 
  - Filtering recommended before visualization
  - Use "Sample" mode with 200-500 nodes
  - Kernel matrix limited to 100x100 for performance
  - Full graph operations still supported

## Use Cases

1. **Story/Narrative Exploration**: Branch narratives with coherent transitions
2. **Knowledge Spaces**: Scientific concepts with semantic relationships  
3. **Decision Trees**: Explore outcomes in high-dimensional possibility space
4. **Creative Exploration**: Generate and traverse artistic/creative variations
5. **Learning Paths**: Educational content with adaptive trajectories
6. **Game Worlds**: Procedurally connected states with meaningful structure

## Seven Universe Examples from xtuff.ai

This format is designed to support your existing universes:
- Each can be encoded with domain-specific kernel functions
- Dimension chosen based on concept complexity (typically 8-64)
- Properties tailored to domain (energy, coherence, plausibility, etc.)
- Visualization reveals semantic clustering and exploration patterns

## Future: Universe Creation Engine

The RKHS format is designed to support abstraction into a creation engine:
- **Template Definition**: Specify kernel type, dimension, and node schema
- **Content Generation**: Populate based on domain knowledge
- **Automatic Structuring**: Build graph based on semantic similarity
- **Export/Import**: Standard format for interchange

## File Extensions

- `.rkhs.json`: Full RKHS universe
- `.rkhs-lite.json`: Simplified format (positions only, no full features)
- `.rkhs-export.json`: Visualization-focused export

## License & Attribution

Created for xtuff.ai multiverse platform.
Format specification v1.0 - November 2025
