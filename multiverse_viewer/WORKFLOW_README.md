# RKHS Workflow Engine

General-purpose workflow engine for creating and managing state-based progressions in RKHS multiverses.

## Overview

The workflow engine allows you to:
- Define custom state progression workflows
- Create nodes that transition through states
- Fork nodes to explore variations
- Track lineage and provenance
- Visualize evolution in 3D space

**Key Feature:** Uses standard RKHS format - no special schema required!

## Quick Start

```python
from rkhs_workflow import (
    create_creative_writing_workflow,
    create_workflow_universe
)

# 1. Create a workflow universe
config = create_creative_writing_workflow()
universe, engine = create_workflow_universe(config)

# 2. Create initial node
logline_id = engine.create_node(
    content={
        "title": "My Story",
        "text": "A detective discovers they've been investigating their own murder."
    },
    state="logline"
)

# 3. Progress through states
summary_id = engine.transition(logline_id, "summary")
synopsis_id = engine.transition(summary_id, "synopsis")

# 4. Fork with variations
treatment_id = engine.transition(synopsis_id, "treatment")
forks = engine.fork(
    source_id=treatment_id,
    n_forks=5,
    fork_fn=my_variation_function
)

# 5. Save and visualize
save_workflow_universe(universe, "my_multiverse.rkhs.json")
```

## Core Concepts

### States

States represent stages in your workflow (e.g., logline, summary, manuscript).

```python
StateDefinition(
    name="logline",
    description="One sentence concept",
    typical_length_range=(10, 200)
)
```

### Transitions

Transitions define how to move from one state to another.

```python
TransitionDefinition(
    from_state="logline",
    to_state="summary",
    transition_type="expand",
    transform_fn=my_expand_function  # Optional
)
```

### Forking

Create multiple variations from a single node:

```python
fork_ids = engine.fork(
    source_id=node_id,
    n_forks=5,
    fork_fn=lambda text, idx: f"{text} [Variation {idx}]",
    fork_params=[
        {"setting": "space_station"},
        {"setting": "medieval_castle"},
        # ...
    ]
)
```

## Built-in Workflows

### Creative Writing
```python
from rkhs_workflow import create_creative_writing_workflow

config = create_creative_writing_workflow()
# States: logline → summary → synopsis → treatment → outline → draft → manuscript
```

### Code Evolution
```python
from rkhs_workflow import create_code_evolution_workflow

config = create_code_evolution_workflow()
# States: idea → spec → pseudocode → implementation → tested → optimized
```

## Creating Custom Workflows

```python
from rkhs_workflow import WorkflowConfig, StateDefinition, TransitionDefinition

# Define your states
states = [
    StateDefinition("idea", "Initial concept", (10, 100)),
    StateDefinition("design", "Detailed design", (500, 5000)),
    StateDefinition("prototype", "Working prototype", (1000, 50000)),
]

# Define transitions
transitions = [
    TransitionDefinition("idea", "design", "expand"),
    TransitionDefinition("design", "prototype", "implement"),
]

# Create config
config = WorkflowConfig(
    name="my_workflow",
    description="Custom workflow",
    states=states,
    transitions=transitions,
    initial_state="idea",
    terminal_states=["prototype"]
)

# Create universe
universe, engine = create_workflow_universe(config)
```

## Integration with LLMs

Replace mock functions with real LLM calls:

```python
def llm_expand(text: str, params: Dict) -> str:
    """Expand text using LLM"""
    from your_llm_library import call_llm

    prompt = f"Expand this logline into a summary:\n{text}"
    return call_llm(prompt, **params)

# Use in transitions
TransitionDefinition(
    from_state="logline",
    to_state="summary",
    transition_type="expand",
    transform_fn=llm_expand
)
```

## Embedding Integration

Add custom embedding function:

```python
def get_embedding(text: str) -> List[float]:
    """Generate embeddings for text"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode(text).tolist()

# Set as default
config.default_embedding_fn = get_embedding
```

## Querying the Multiverse

### Get lineage
```python
lineage = engine.get_lineage(node_id)
# Returns: [root_id, parent_id, grandparent_id, ..., node_id]
```

### Get descendants
```python
descendants = engine.get_descendants(node_id)
# Returns: [child1_id, child2_id, grandchild1_id, ...]
```

### Get state distribution
```python
distribution = engine.get_state_distribution()
# Returns: {"logline": 5, "summary": 3, "treatment": 2, ...}
```

## Visualization

The workflow engine integrates seamlessly with the RKHS viewer:

1. Save your universe:
   ```python
   save_workflow_universe(universe, "my_multiverse.rkhs.json")
   ```

2. Launch viewer:
   ```bash
   STREAMLIT_ENV=development streamlit run multiverse_viewer.py
   ```

3. Load and explore:
   - Use "Option 2: Load from Path" for large files
   - View 3D network to see state progressions (Z-axis)
   - Examine forking branches
   - Track content evolution

## Real-World Examples

### Creative Writing with Settings Variations

```python
# Start with a logline
logline_id = engine.create_node(
    content={"title": "Space Mystery", "text": "A murder on a space station..."},
    state="logline"
)

# Progress to treatment
treatment_id = engine.transition(logline_id, "summary")
treatment_id = engine.transition(treatment_id, "synopsis")
treatment_id = engine.transition(treatment_id, "treatment")

# Fork with 5 different settings
settings = ["orbital_station", "lunar_base", "mars_colony", "asteroid_mine", "generation_ship"]
forks = engine.fork(
    source_id=treatment_id,
    n_forks=5,
    fork_fn=lambda text, idx: rewrite_for_setting(text, settings[idx]),
    fork_params=[{"setting": s} for s in settings]
)

# Continue best fork to manuscript
best_fork = forks[0]
outline_id = engine.transition(best_fork, "outline")
draft_id = engine.transition(outline_id, "draft")
manuscript_id = engine.transition(draft_id, "manuscript")
```

### Research Paper Evolution

```python
# Define research workflow
states = [
    StateDefinition("question", "Research question", (50, 500)),
    StateDefinition("hypothesis", "Hypothesis", (100, 1000)),
    StateDefinition("methodology", "Research method", (500, 5000)),
    StateDefinition("experiment", "Experimental results", (1000, 10000)),
    StateDefinition("paper", "Full paper", (5000, 50000)),
]

transitions = [
    TransitionDefinition("question", "hypothesis", "formulate"),
    TransitionDefinition("hypothesis", "methodology", "design"),
    TransitionDefinition("methodology", "experiment", "execute"),
    TransitionDefinition("experiment", "paper", "write"),
]

config = WorkflowConfig(
    name="research",
    description="Research paper evolution",
    states=states,
    transitions=transitions,
    initial_state="question",
    terminal_states=["paper"]
)
```

## Best Practices

1. **Start small**: Begin with 2-3 states, add more as needed
2. **Use embeddings**: Provide a good embedding function for semantic search
3. **Track metadata**: Store prompts, parameters, and rationale in metadata
4. **Fork strategically**: Don't create too many forks at once
5. **Prune dead ends**: Remove unproductive branches to keep multiverse clean
6. **Save frequently**: Workflow universes can get large
7. **Visualize often**: Use the viewer to understand your multiverse structure

## Performance Tips

- For large multiverses (>1000 nodes), use sampling for visualization
- Batch embedding generation for efficiency
- Store universes as compressed JSON if file size is a concern
- Use the development mode for direct file path loading (large files)

## Architecture

The workflow engine is a thin layer over standard RKHS:
- **No schema changes** - uses existing `content`, `metadata`, `parent_ids`, `children_ids`
- **Pure functions** - no side effects, easy to test
- **Composable** - combine with other RKHS tools
- **Extensible** - add custom states, transitions, transforms

## Support

See `demo_creative_writing_workflow.py` for a complete working example.

For questions or issues, refer to the main RKHS documentation.
