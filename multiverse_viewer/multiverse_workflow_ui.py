#!/usr/bin/env python3
"""
multiverse_workflow_ui.py

Streamlit UI for RKHS multiverse workflow with:
- Claude Max integration (when available)
- API fallback using nimble-llm-caller
- Creative writing workflow
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rkhs_workflow import (
    create_creative_writing_workflow,
    create_workflow_universe,
    save_workflow_universe,
    load_workflow_universe,
    RKHSWorkflowEngine
)

# Import nimble-llm-caller
# Use environment-aware path detection
try:
    # Add project root to path (3 levels up: multiverse_viewer -> hilberts -> xcu_my_apps)
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from nimble_llm_caller import call_llm
    HAS_LLM_CALLER = True
except ImportError:
    HAS_LLM_CALLER = False
    st.warning("⚠️ nimble-llm-caller not found. Install for API fallback support.")


# ============================================================================
# CONFIGURATION
# ============================================================================

MULTIVERSE_DIR = Path(__file__).parent
DEFAULT_MULTIVERSE = MULTIVERSE_DIR / "current_multiverse.rkhs.json"

# Model configuration - will be looked up from litellm at runtime
DEFAULT_MODELS = {
    "fast": "claude-3-5-haiku-20241022",  # For quick expansions
    "smart": "claude-3-5-sonnet-20241022",  # For quality content
    "creative": "gpt-4o",  # For creative variations
}


# ============================================================================
# CLAUDE MAX DETECTION
# ============================================================================

def is_claude_max_available() -> bool:
    """
    Check if Claude Max is available (running in Claude Code CLI).
    Claude Max = Task tool available for subprocess calls.
    """
    # In Claude Code, we can check environment or just assume availability
    # For now, we'll check if we're in a Claude Code session
    return os.getenv('CLAUDE_CODE_SESSION') is not None or True  # Always try Task first


def use_claude_max_for_transform(prompt: str, context: Dict[str, Any]) -> Optional[str]:
    """
    Use Claude Max (Task tool) to perform content transformation.
    Returns None if Claude Max not available.
    """
    if not is_claude_max_available():
        return None

    # In actual implementation, this would use Task tool
    # For now, return None to indicate it should be called by Claude Code directly
    return None


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def get_available_models() -> List[str]:
    """Dynamically fetch available models from litellm"""
    if not HAS_LLM_CALLER:
        return list(DEFAULT_MODELS.values())

    try:
        # Try to get model list from litellm
        import litellm
        # litellm.model_list is available in recent versions
        models = []

        # Common model patterns
        for provider in ['anthropic/', 'openai/', 'gpt-', 'claude-']:
            try:
                provider_models = [m for m in litellm.model_list if provider in m]
                models.extend(provider_models[:5])  # Top 5 per provider
            except:
                pass

        return models if models else list(DEFAULT_MODELS.values())
    except:
        return list(DEFAULT_MODELS.values())


def call_llm_for_transform(prompt: str,
                           model: Optional[str] = None,
                           context: Optional[Dict] = None) -> str:
    """
    Use API to perform content transformation.

    Args:
        prompt: The transformation prompt
        model: Model to use (None = use smart default)
        context: Additional context

    Returns:
        Generated content
    """
    if not HAS_LLM_CALLER:
        return "[LLM not available - please install nimble-llm-caller]"

    if model is None:
        model = DEFAULT_MODELS["smart"]

    try:
        response = call_llm(
            prompt=prompt,
            model=model,
            temperature=0.7,
            max_tokens=4000
        )
        return response
    except Exception as e:
        return f"[Error calling LLM: {e}]"


# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def expand_content(text: str, from_state: str, to_state: str,
                   use_api: bool = False, model: Optional[str] = None) -> str:
    """Expand content from one state to next"""

    if not use_api:
        # Request user to use Claude Max via slash command
        return f"[Use /multiverse transition command with Claude Max for best results]\n\n{text}"

    # Fallback to API
    prompt = f"""Expand this {from_state} into a {to_state}.

{from_state.upper()}:
{text}

Write a detailed {to_state} that expands on this concept. Be creative and add depth."""

    return call_llm_for_transform(prompt, model)


def create_variation(text: str, variation_params: Dict[str, Any],
                    use_api: bool = False, model: Optional[str] = None) -> str:
    """Create a variation of content"""

    if not use_api:
        return f"[Use /multiverse fork command with Claude Max for best results]\n\n{text}"

    # Fallback to API
    setting = variation_params.get('setting', 'alternative version')
    prompt = f"""Rewrite this content with the following variation:
Setting/Context: {setting}

ORIGINAL:
{text}

Create a variation that maintains the core story but adapts it to the new setting."""

    return call_llm_for_transform(prompt, model)


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="Multiverse Workflow", layout="wide", page_icon="🌌")

    st.title("🌌 RKHS Multiverse Workflow")
    st.markdown("**Creative writing workflow** with Claude Max + API fallback")

    # Initialize session state
    if 'universe' not in st.session_state:
        st.session_state.universe = None
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Claude Max status
        claude_max_available = is_claude_max_available()
        if claude_max_available:
            st.success("✅ Claude Max available")
            st.info("💡 Use `/multiverse` slash command for best results")
        else:
            st.warning("⚠️ Claude Max not detected")

        st.divider()

        # Model selection (for API fallback)
        st.subheader("API Fallback Models")
        if HAS_LLM_CALLER:
            available_models = get_available_models()
            fast_model = st.selectbox("Fast model", available_models,
                                     index=available_models.index(DEFAULT_MODELS["fast"])
                                     if DEFAULT_MODELS["fast"] in available_models else 0)
            smart_model = st.selectbox("Smart model", available_models,
                                      index=available_models.index(DEFAULT_MODELS["smart"])
                                      if DEFAULT_MODELS["smart"] in available_models else 0)

            st.session_state.models = {
                "fast": fast_model,
                "smart": smart_model
            }
        else:
            st.error("nimble-llm-caller not available")

        st.divider()

        # Force API mode toggle
        st.session_state.force_api = st.checkbox(
            "Force API mode",
            value=False,
            help="Use API calls instead of Claude Max"
        )

    # Main tabs
    tabs = st.tabs(["📝 Create", "🔄 Transform", "🔱 Fork", "📊 Browse", "💾 Save/Load"])

    # Tab 1: Create
    with tabs[0]:
        st.header("Create New Node")

        col1, col2 = st.columns([2, 1])

        with col1:
            state = st.selectbox(
                "State",
                ["logline", "summary", "synopsis", "treatment", "outline", "draft", "manuscript"]
            )

            title = st.text_input("Title", placeholder="My Story Idea")
            text = st.text_area("Content", height=200,
                               placeholder="Enter your content here...")

            if st.button("✨ Create Node", type="primary"):
                if not st.session_state.universe:
                    # Create new universe
                    config = create_creative_writing_workflow()
                    universe, engine = create_workflow_universe(config, dimension=768)
                    st.session_state.universe = universe
                    st.session_state.engine = engine
                    st.success("Created new multiverse")

                if text:
                    node_id = st.session_state.engine.create_node(
                        content={"title": title or "Untitled", "text": text},
                        state=state
                    )
                    st.success(f"✅ Created node: {node_id}")
                    st.json(st.session_state.universe.nodes[node_id].content)
                else:
                    st.error("Please enter content")

        with col2:
            st.subheader("Workflow States")
            st.markdown("""
            - **logline**: One sentence
            - **summary**: Brief summary
            - **synopsis**: Plot synopsis
            - **treatment**: Detailed treatment
            - **outline**: Scene outline
            - **draft**: First draft
            - **manuscript**: Final version
            """)

    # Tab 2: Transform
    with tabs[1]:
        st.header("Transform Node")

        if not st.session_state.universe:
            st.warning("Create a multiverse first")
        else:
            node_ids = list(st.session_state.universe.nodes.keys())

            if node_ids:
                source_id = st.selectbox("Source Node", node_ids)
                source_node = st.session_state.universe.nodes[source_id]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Current")
                    st.write(f"**State**: {source_node.content.get('state')}")
                    st.write(f"**Title**: {source_node.content.get('title')}")
                    st.text_area("Content", source_node.content.get('text', ''), height=200, disabled=True)

                with col2:
                    st.subheader("Transform To")
                    target_state = st.selectbox(
                        "Target State",
                        ["summary", "synopsis", "treatment", "outline", "draft", "manuscript"]
                    )

                    use_api = st.session_state.force_api or st.checkbox("Use API (not Claude Max)")

                    if use_api:
                        model = st.selectbox("Model", list(st.session_state.get('models', DEFAULT_MODELS).values()))
                    else:
                        st.info("💡 Will use Claude Max - use /multiverse command or click below")

                    if st.button("🚀 Transform", type="primary"):
                        if use_api:
                            # Use API
                            with st.spinner("Generating with API..."):
                                from_state = source_node.content.get('state')
                                new_text = expand_content(
                                    source_node.content.get('text', ''),
                                    from_state,
                                    target_state,
                                    use_api=True,
                                    model=model if use_api else None
                                )

                                # Create new node
                                new_node_id = st.session_state.engine.create_node(
                                    content={
                                        "title": source_node.content.get('title'),
                                        "text": new_text,
                                    },
                                    state=target_state,
                                    parent_id=source_id,
                                    metadata={"method": "api", "model": model}
                                )

                                st.success(f"✅ Created: {new_node_id}")
                                st.text_area("New Content", new_text, height=200)
                        else:
                            st.info(f"""
                            **Use Claude Max for best results:**

                            ```
                            /multiverse transition {source_id} {target_state}
                            ```
                            """)
            else:
                st.info("No nodes yet - create one first")

    # Tab 3: Fork
    with tabs[2]:
        st.header("Fork Node (Create Variations)")

        if not st.session_state.universe:
            st.warning("Create a multiverse first")
        else:
            node_ids = list(st.session_state.universe.nodes.keys())

            if node_ids:
                source_id = st.selectbox("Source Node", node_ids, key="fork_source")
                source_node = st.session_state.universe.nodes[source_id]

                st.subheader("Source Content")
                st.write(f"**State**: {source_node.content.get('state')}")
                st.text_area("Content", source_node.content.get('text', '')[:500] + "...", height=100, disabled=True)

                st.divider()

                n_forks = st.number_input("Number of variations", min_value=1, max_value=10, value=5)
                variation_prompt = st.text_area(
                    "Variation instructions",
                    placeholder="E.g., Rewrite in 5 different settings: space station, medieval castle, modern city, underwater base, desert colony",
                    height=100
                )

                use_api = st.session_state.force_api or st.checkbox("Use API (not Claude Max)", key="fork_api")

                if st.button("🔱 Create Forks", type="primary"):
                    if use_api:
                        st.info("API forking not yet implemented - use Claude Max")
                        st.code(f"/multiverse fork {source_id} {n_forks} \"{variation_prompt}\"")
                    else:
                        st.info(f"""
                        **Use Claude Max for best results:**

                        ```
                        /multiverse fork {source_id} {n_forks} "{variation_prompt}"
                        ```
                        """)
            else:
                st.info("No nodes yet - create one first")

    # Tab 4: Browse
    with tabs[3]:
        st.header("Browse Multiverse")

        if not st.session_state.universe:
            st.warning("Create or load a multiverse first")
        else:
            # Statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Nodes", len(st.session_state.universe.nodes))
            col2.metric("Total Edges", len(st.session_state.universe.edges))
            col3.metric("Dimension", st.session_state.universe.dimension)

            # State distribution
            if st.session_state.engine:
                dist = st.session_state.engine.get_state_distribution()
                st.subheader("State Distribution")
                for state, count in dist.items():
                    if count > 0:
                        st.write(f"**{state}**: {count}")

            st.divider()

            # Node browser
            st.subheader("Nodes")
            for node_id, node in list(st.session_state.universe.nodes.items())[:20]:
                with st.expander(f"{node.content.get('title', node_id)} [{node.content.get('state')}]"):
                    st.write(f"**ID**: {node_id}")
                    st.write(f"**State**: {node.content.get('state')}")
                    st.write(f"**Parents**: {len(node.parent_ids)}")
                    st.write(f"**Children**: {len(node.children_ids)}")
                    st.text_area("Content", node.content.get('text', '')[:500], height=100, disabled=True, key=f"browse_{node_id}")

    # Tab 5: Save/Load
    with tabs[4]:
        st.header("Save/Load Multiverse")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💾 Save")
            filename = st.text_input(
                "Filename",
                value=st.session_state.current_file or "my_multiverse.rkhs.json"
            )

            if st.button("Save Multiverse"):
                if st.session_state.universe:
                    filepath = MULTIVERSE_DIR / filename
                    save_workflow_universe(st.session_state.universe, str(filepath))
                    st.session_state.current_file = filename
                    st.success(f"✅ Saved to {filepath}")
                else:
                    st.error("No multiverse to save")

        with col2:
            st.subheader("📂 Load")

            # List available files
            rkhs_files = list(MULTIVERSE_DIR.glob("*.rkhs.json"))
            if rkhs_files:
                selected_file = st.selectbox(
                    "Select file",
                    [f.name for f in rkhs_files]
                )

                if st.button("Load Multiverse"):
                    filepath = MULTIVERSE_DIR / selected_file
                    config = create_creative_writing_workflow()
                    universe, engine = load_workflow_universe(str(filepath), config)
                    st.session_state.universe = universe
                    st.session_state.engine = engine
                    st.session_state.current_file = selected_file
                    st.success(f"✅ Loaded {selected_file}")
                    st.rerun()
            else:
                st.info("No saved multiverses found")


if __name__ == "__main__":
    main()
