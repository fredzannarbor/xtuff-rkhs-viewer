# Example RKHS Universes

This directory contains 3 production-quality knowledge universes demonstrating the RKHS format.

## Available Universes

1. **codexspace_v1.rkhs.json** - 28,000 books from Project Gutenberg embedded in Hilbert space (768MB)
2. **codexspace_sample.rkhs.json** - Sample subset for faster testing (2.4MB)
3. **ideation_demo.rkhs.json** - Creative ideation space demonstration (160KB)

## File Format

Each .rkhs.json file is a JSON document with this structure:

```json
{
  "name": "Universe Name",
  "description": "Brief description",
  "dimension": 768,
  "kernel_type": "rbf",
  "kernel_params": {"gamma": 1.0},
  "nodes": {
    "node_id": {
      "id": "unique_id",
      "position": [x, y, z],
      "content": {
        "title": "Node Title",
        "description": "Detailed explanation"
      },
      "metadata": {},
      "kernel_features": [0.1, 0.2, ...],
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
      "transition_type": "semantic",
      "metadata": {}
    }
  ],
  "metadata": {},
  "version": "1.0"
}
```

## Using These Files

Load any .rkhs.json file in the viewer to explore that universe. Use the **Open** tab to:
- Upload existing files
- Select from examples
- Create new sample universes

## Creating Your Own

Want to create custom universes for your domain?

📧 hello@xtuff.ai | 🌐 [codexes.xtuff.ai](https://codexes.xtuff.ai)

## Performance Notes

- **codexspace_v1.rkhs.json** (28,000 nodes): Use sampling mode for visualization
- **codexspace_sample.rkhs.json** (smaller subset): All features work smoothly
- **ideation_demo.rkhs.json** (small): Optimal for testing and demos
