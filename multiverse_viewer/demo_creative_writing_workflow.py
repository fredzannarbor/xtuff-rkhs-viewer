#!/usr/bin/env python3
"""
demo_creative_writing_workflow.py

Demonstrates using the RKHS workflow engine for creative writing.

Shows:
1. Creating a workflow universe
2. Adding initial loglines
3. Progressing through states (logline → treatment)
4. Forking with variations (5 different settings)
5. Continuing progression on forks
6. Visualizing the multiverse
"""

from rkhs_workflow import (
    create_creative_writing_workflow,
    create_workflow_universe,
    save_workflow_universe
)
from typing import Dict


# ============================================================================
# TRANSFORM FUNCTIONS (in real use, these would call LLMs)
# ============================================================================

def mock_expand_fn(text: str, params: Dict) -> str:
    """Mock function to expand text to next level"""
    expansion_type = params.get('expansion_type', 'generic')
    word_multiplier = params.get('word_multiplier', 5)

    # In real use, this would call an LLM
    return f"{text}\n\n[Expanded {word_multiplier}x with {expansion_type} expansion...]\n" + " ".join(["Content"] * (len(text.split()) * word_multiplier))


def mock_fork_fn(text: str, fork_idx: int) -> str:
    """Mock function to create variations"""
    settings = ["space_station", "medieval_castle", "modern_city", "underwater_base", "desert_colony"]
    setting = settings[fork_idx % len(settings)]

    # In real use, this would call an LLM to rewrite in different setting
    return f"{text}\n\n[Setting variation: {setting}]\n" + text


# ============================================================================
# DEMO WORKFLOW
# ============================================================================

def main():
    print("=" * 70)
    print("RKHS Creative Writing Workflow Demo")
    print("=" * 70)

    # 1. Create workflow universe
    print("\n1. Creating workflow universe...")
    config = create_creative_writing_workflow()
    universe, engine = create_workflow_universe(
        config,
        name="Creative Writing Multiverse",
        dimension=768
    )
    print(f"   ✓ Created universe with {len(config.states)} states")

    # 2. Add initial loglines
    print("\n2. Creating initial loglines...")
    loglines = [
        "A detective discovers they've been investigating their own murder.",
        "An AI falls in love with its creator, but cannot tell them.",
        "The last librarian on Earth must decide which books to save."
    ]

    logline_ids = []
    for i, logline in enumerate(loglines):
        node_id = engine.create_node(
            content={
                "title": f"Idea {i+1}",
                "text": logline,
                "genre": "sci-fi" if i < 2 else "dystopian"
            },
            state="logline"
        )
        logline_ids.append(node_id)
        print(f"   ✓ Created logline: {logline[:50]}...")

    # 3. Progress first logline through states
    print("\n3. Progressing first idea through states...")
    current_id = logline_ids[0]

    # logline → summary
    print("   - Expanding to summary...")
    summary_id = engine.transition(
        current_id,
        "summary",
        transform_params={"word_multiplier": 3}
    )

    # summary → synopsis
    print("   - Expanding to synopsis...")
    synopsis_id = engine.transition(
        summary_id,
        "synopsis",
        transform_params={"word_multiplier": 5}
    )

    # synopsis → treatment
    print("   - Expanding to treatment...")
    treatment_id = engine.transition(
        synopsis_id,
        "treatment",
        transform_params={"word_multiplier": 3}
    )
    print(f"   ✓ Progressed to treatment state")

    # 4. Fork the treatment with 5 different settings
    print("\n4. Forking treatment with 5 setting variations...")
    fork_ids = engine.fork(
        source_id=treatment_id,
        n_forks=5,
        fork_fn=mock_fork_fn,
        fork_params=[
            {"setting": "space_station", "era": "2300"},
            {"setting": "medieval_castle", "era": "1400"},
            {"setting": "modern_city", "era": "2024"},
            {"setting": "underwater_base", "era": "2100"},
            {"setting": "desert_colony", "era": "2200"}
        ]
    )
    print(f"   ✓ Created {len(fork_ids)} treatment variations")

    # 5. Progress one fork to outline
    print("\n5. Progressing space station variant to outline...")
    outline_id = engine.transition(
        fork_ids[0],
        "outline",
        transform_params={"detail_level": "high"}
    )
    print(f"   ✓ Created outline from first fork")

    # 6. Progress second logline differently
    print("\n6. Creating alternate progression for second idea...")
    alt_summary_id = engine.transition(logline_ids[1], "summary")
    alt_synopsis_id = engine.transition(alt_summary_id, "synopsis")
    print(f"   ✓ Second idea progressed to synopsis")

    # 7. Show statistics
    print("\n7. Multiverse Statistics:")
    print(f"   - Total nodes: {len(universe.nodes)}")
    print(f"   - Total edges: {len(universe.edges)}")

    state_dist = engine.get_state_distribution()
    print("\n   State distribution:")
    for state, count in state_dist.items():
        if count > 0:
            print(f"     • {state}: {count}")

    # 8. Show lineage
    print("\n8. Lineage of space station outline:")
    lineage = engine.get_lineage(outline_id)
    print(f"   Path length: {len(lineage)} nodes")
    for i, node_id in enumerate(lineage):
        node = universe.nodes[node_id]
        state = node.content.get('state', 'unknown')
        title = node.content.get('title', node_id[:20])
        print(f"     {i+1}. [{state}] {title}")

    # 9. Show descendants of first treatment
    print("\n9. Descendants of original treatment:")
    descendants = engine.get_descendants(treatment_id)
    print(f"   Total descendants: {len(descendants)}")
    for desc_id in descendants[:5]:
        node = universe.nodes[desc_id]
        state = node.content.get('state', 'unknown')
        setting = node.content.get('fork_params', {}).get('setting', 'N/A')
        print(f"     • [{state}] Setting: {setting}")

    # 10. Save universe
    output_path = "../creative_writing_demo.rkhs.json"
    print(f"\n10. Saving universe to {output_path}...")
    save_workflow_universe(universe, output_path)
    print(f"    ✓ Saved!")

    # 11. Instructions for viewing
    print("\n" + "=" * 70)
    print("To visualize this multiverse:")
    print("  1. Launch viewer: streamlit run multiverse_viewer.py")
    print(f"  2. Load file: {output_path}")
    print("  3. Go to 'Visualize' tab")
    print("  4. View the 3D network to see:")
    print("     - State progressions (vertical Z-axis)")
    print("     - Forking branches")
    print("     - Content evolution")
    print("=" * 70)


if __name__ == "__main__":
    main()
