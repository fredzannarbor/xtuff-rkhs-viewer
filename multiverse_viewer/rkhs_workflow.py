"""
rkhs_workflow.py

General-purpose workflow engine for RKHS multiverse operations.

Supports:
- State progressions (A → B → C transitions)
- Forking (creating N variations from a node)
- Content transformations (LLM-based or custom)
- Embedding generation for variable-length content
- Automatic metadata tracking and provenance

Works with standard RKHS format - no schema modifications needed.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import hashlib
import json

# Import from existing multiverse_viewer
try:
    from multiverse_viewer import RKHSNode, RKHSEdge, RKHSUniverse
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from multiverse_viewer import RKHSNode, RKHSEdge, RKHSUniverse


# ============================================================================
# WORKFLOW CONFIGURATION
# ============================================================================

@dataclass
class StateDefinition:
    """Defines a state in a workflow"""
    name: str
    description: str
    typical_length_range: Tuple[int, int]  # (min_chars, max_chars)
    validation_fn: Optional[Callable[[str], bool]] = None
    metadata_schema: Optional[Dict[str, type]] = None


@dataclass
class TransitionDefinition:
    """Defines a transition between states"""
    from_state: str
    to_state: str
    transition_type: str
    transform_fn: Optional[Callable[[str, Dict], str]] = None  # (content, params) -> new_content
    embedding_fn: Optional[Callable[[str], List[float]]] = None
    default_params: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowConfig:
    """Complete workflow configuration"""
    name: str
    description: str
    states: List[StateDefinition]
    transitions: List[TransitionDefinition]
    initial_state: str
    terminal_states: List[str]
    default_embedding_fn: Optional[Callable[[str], List[float]]] = None


# ============================================================================
# WORKFLOW ENGINE
# ============================================================================

class RKHSWorkflowEngine:
    """
    General-purpose workflow engine for RKHS multiverses.

    Handles state progressions, forking, and content transformations
    while maintaining standard RKHS format.
    """

    def __init__(self,
                 universe: RKHSUniverse,
                 config: WorkflowConfig):
        """
        Initialize workflow engine.

        Args:
            universe: The RKHS universe to operate on
            config: Workflow configuration defining states and transitions
        """
        self.universe = universe
        self.config = config

        # Build state and transition lookup maps
        self.states = {s.name: s for s in config.states}
        self.transitions = {}
        for t in config.transitions:
            key = (t.from_state, t.to_state)
            self.transitions[key] = t

    def create_node(self,
                   content: Dict[str, Any],
                   state: str,
                   parent_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   auto_embed: bool = True) -> str:
        """
        Create a new node in the workflow.

        Args:
            content: Node content (must include 'text' key)
            state: Current state name
            parent_id: Optional parent node ID
            metadata: Optional additional metadata
            auto_embed: Whether to automatically generate embeddings

        Returns:
            New node ID
        """
        if state not in self.states:
            raise ValueError(f"Unknown state: {state}")

        # Generate node ID
        node_id = self._generate_node_id(content, state, parent_id)

        # Add state to content
        content_with_state = {**content, "state": state}

        # Generate embedding if needed
        if auto_embed:
            text = content.get('text', '')
            state_def = self.states[state]

            # Use state-specific or default embedding function
            embed_fn = self.config.default_embedding_fn
            if embed_fn:
                kernel_features = embed_fn(text)
            else:
                # Fallback: simple hash-based features
                kernel_features = self._hash_embedding(text)
        else:
            kernel_features = [0.0] * self.universe.dimension

        # Compute position (could be enhanced with UMAP/t-SNE)
        position = self._compute_position(kernel_features, state)

        # Build metadata
        node_metadata = {
            "state": state,
            "created_at": datetime.now().isoformat(),
            "workflow": self.config.name
        }
        if metadata:
            node_metadata.update(metadata)

        # Create node
        node = RKHSNode(
            id=node_id,
            position=position,
            content=content_with_state,
            metadata=node_metadata,
            kernel_features=kernel_features,
            timestamp=datetime.now().isoformat(),
            parent_ids=[parent_id] if parent_id else [],
            children_ids=[]
        )

        # Add to universe
        self.universe.nodes[node_id] = node

        # Create edge if parent exists
        if parent_id and parent_id in self.universe.nodes:
            parent_node = self.universe.nodes[parent_id]
            parent_node.children_ids.append(node_id)

            # Determine transition type
            parent_state = parent_node.content.get('state', 'unknown')
            transition_key = (parent_state, state)
            transition_type = self.transitions.get(transition_key, None)

            edge = RKHSEdge(
                source_id=parent_id,
                target_id=node_id,
                weight=1.0,
                kernel_similarity=self._compute_similarity(
                    parent_node.kernel_features,
                    kernel_features
                ),
                transition_type=transition_type.transition_type if transition_type else "custom",
                metadata={"created_at": datetime.now().isoformat()}
            )
            self.universe.edges.append(edge)

        return node_id

    def transition(self,
                   source_id: str,
                   target_state: str,
                   transform_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Transition a node to a new state.

        Args:
            source_id: Source node ID
            target_state: Target state name
            transform_params: Parameters for transformation function

        Returns:
            New node ID
        """
        if source_id not in self.universe.nodes:
            raise ValueError(f"Node not found: {source_id}")

        source_node = self.universe.nodes[source_id]
        source_state = source_node.content.get('state')

        if not source_state:
            raise ValueError(f"Node {source_id} has no state")

        # Find transition
        transition_key = (source_state, target_state)
        if transition_key not in self.transitions:
            raise ValueError(f"No transition defined from {source_state} to {target_state}")

        transition = self.transitions[transition_key]

        # Apply transformation
        source_text = source_node.content.get('text', '')
        params = {**(transition.default_params or {}), **(transform_params or {})}

        if transition.transform_fn:
            new_text = transition.transform_fn(source_text, params)
        else:
            # No transform function - copy text as-is
            new_text = source_text

        # Create new content
        new_content = {
            **source_node.content,
            'text': new_text,
            'state': target_state,
            'word_count': len(new_text.split()),
            'char_count': len(new_text)
        }

        # Create new node
        metadata = {
            "transition": transition.transition_type,
            "source_state": source_state,
            "transform_params": params
        }

        return self.create_node(new_content, target_state, source_id, metadata)

    def fork(self,
             source_id: str,
             n_forks: int,
             fork_fn: Callable[[str, int], str],
             fork_params: Optional[List[Dict[str, Any]]] = None,
             same_state: bool = True) -> List[str]:
        """
        Create multiple variations (forks) from a node.

        Args:
            source_id: Source node ID
            n_forks: Number of forks to create
            fork_fn: Function (text, fork_idx) -> new_text
            fork_params: Optional list of parameter dicts (one per fork)
            same_state: Whether forks stay in same state (True) or advance (False)

        Returns:
            List of new node IDs
        """
        if source_id not in self.universe.nodes:
            raise ValueError(f"Node not found: {source_id}")

        source_node = self.universe.nodes[source_id]
        source_state = source_node.content.get('state')
        source_text = source_node.content.get('text', '')

        if fork_params and len(fork_params) != n_forks:
            raise ValueError(f"fork_params length ({len(fork_params)}) != n_forks ({n_forks})")

        fork_ids = []

        for i in range(n_forks):
            # Generate forked content
            params = fork_params[i] if fork_params else {}
            forked_text = fork_fn(source_text, i)

            # Create new content
            new_content = {
                **source_node.content,
                'text': forked_text,
                'fork_index': i,
                'fork_params': params,
                'word_count': len(forked_text.split()),
                'char_count': len(forked_text)
            }

            # Create new node
            metadata = {
                "fork_index": i,
                "fork_of": source_id,
                "fork_params": params
            }

            fork_id = self.create_node(new_content, source_state, source_id, metadata)
            fork_ids.append(fork_id)

        return fork_ids

    def get_lineage(self, node_id: str) -> List[str]:
        """Get the full lineage from root to this node"""
        if node_id not in self.universe.nodes:
            return []

        node = self.universe.nodes[node_id]
        lineage = [node_id]

        while node.parent_ids:
            parent_id = node.parent_ids[0]  # Follow first parent
            lineage.insert(0, parent_id)
            if parent_id not in self.universe.nodes:
                break
            node = self.universe.nodes[parent_id]

        return lineage

    def get_descendants(self, node_id: str) -> List[str]:
        """Get all descendants of a node"""
        if node_id not in self.universe.nodes:
            return []

        descendants = []
        queue = [node_id]

        while queue:
            current_id = queue.pop(0)
            node = self.universe.nodes.get(current_id)
            if not node:
                continue

            for child_id in node.children_ids:
                descendants.append(child_id)
                queue.append(child_id)

        return descendants

    def get_state_distribution(self) -> Dict[str, int]:
        """Get count of nodes in each state"""
        distribution = {state: 0 for state in self.states.keys()}
        distribution['unknown'] = 0

        for node in self.universe.nodes.values():
            state = node.content.get('state', 'unknown')
            if state in distribution:
                distribution[state] += 1
            else:
                distribution['unknown'] += 1

        return distribution

    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================

    def _generate_node_id(self, content: Dict, state: str, parent_id: Optional[str]) -> str:
        """Generate unique node ID"""
        timestamp = datetime.now().isoformat()
        text_sample = content.get('text', '')[:100]
        id_string = f"{state}_{text_sample}_{timestamp}_{parent_id or 'root'}"
        hash_obj = hashlib.sha256(id_string.encode())
        return f"{state}_{hash_obj.hexdigest()[:12]}"

    def _hash_embedding(self, text: str, dim: Optional[int] = None) -> List[float]:
        """Generate deterministic hash-based embedding"""
        if dim is None:
            dim = self.universe.dimension

        # Simple hash-based embedding (for fallback only)
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to floats in [-1, 1]
        embedding = []
        for i in range(min(dim, len(hash_bytes))):
            val = (hash_bytes[i] / 255.0) * 2 - 1
            embedding.append(val)

        # Pad if needed
        while len(embedding) < dim:
            embedding.append(0.0)

        return embedding[:dim]

    def _compute_position(self, kernel_features: List[float], state: str) -> List[float]:
        """Compute 3D position from kernel features"""
        # Simple projection: use first 3 dimensions
        # Could be enhanced with PCA/UMAP/t-SNE
        position = kernel_features[:3] if len(kernel_features) >= 3 else kernel_features + [0.0] * (3 - len(kernel_features))

        # Add state-specific offset for visualization
        state_idx = list(self.states.keys()).index(state) if state in self.states else 0
        position[2] += state_idx * 0.5  # Z-axis encodes state progression

        return position

    def _compute_similarity(self, feat1: List[float], feat2: List[float]) -> float:
        """Compute cosine similarity"""
        f1 = np.array(feat1)
        f2 = np.array(feat2)

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(f1, f2) / (norm1 * norm2))


# ============================================================================
# PREDEFINED WORKFLOW CONFIGURATIONS
# ============================================================================

def create_creative_writing_workflow() -> WorkflowConfig:
    """
    Example: Creative writing workflow from logline to manuscript
    """
    states = [
        StateDefinition("logline", "One sentence concept", (10, 200)),
        StateDefinition("summary", "Brief summary", (50, 500)),
        StateDefinition("synopsis", "Plot synopsis", (200, 2000)),
        StateDefinition("treatment", "Detailed treatment", (1000, 5000)),
        StateDefinition("outline", "Scene-by-scene outline", (2000, 10000)),
        StateDefinition("draft", "First draft", (10000, 100000)),
        StateDefinition("manuscript", "Final manuscript", (10000, 150000)),
    ]

    transitions = [
        TransitionDefinition("logline", "summary", "expand"),
        TransitionDefinition("summary", "synopsis", "expand"),
        TransitionDefinition("synopsis", "treatment", "expand"),
        TransitionDefinition("treatment", "outline", "structure"),
        TransitionDefinition("outline", "draft", "write"),
        TransitionDefinition("draft", "manuscript", "revise"),
        # Allow backwards transitions for iteration
        TransitionDefinition("summary", "logline", "condense"),
        TransitionDefinition("synopsis", "summary", "condense"),
        TransitionDefinition("treatment", "synopsis", "condense"),
    ]

    return WorkflowConfig(
        name="creative_writing",
        description="Creative writing from concept to manuscript",
        states=states,
        transitions=transitions,
        initial_state="logline",
        terminal_states=["manuscript"]
    )


def create_code_evolution_workflow() -> WorkflowConfig:
    """
    Example: Code evolution workflow
    """
    states = [
        StateDefinition("idea", "Feature idea", (10, 200)),
        StateDefinition("spec", "Specification", (100, 2000)),
        StateDefinition("pseudocode", "Pseudocode", (200, 5000)),
        StateDefinition("implementation", "Working code", (500, 50000)),
        StateDefinition("tested", "Code with tests", (1000, 100000)),
        StateDefinition("optimized", "Optimized version", (1000, 100000)),
    ]

    transitions = [
        TransitionDefinition("idea", "spec", "specify"),
        TransitionDefinition("spec", "pseudocode", "design"),
        TransitionDefinition("pseudocode", "implementation", "implement"),
        TransitionDefinition("implementation", "tested", "test"),
        TransitionDefinition("tested", "optimized", "optimize"),
    ]

    return WorkflowConfig(
        name="code_evolution",
        description="Code evolution from idea to optimized implementation",
        states=states,
        transitions=transitions,
        initial_state="idea",
        terminal_states=["optimized"]
    )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_workflow_universe(config: WorkflowConfig,
                             name: Optional[str] = None,
                             dimension: int = 768) -> Tuple[RKHSUniverse, RKHSWorkflowEngine]:
    """
    Create a new RKHS universe configured for a workflow.

    Args:
        config: Workflow configuration
        name: Optional universe name (defaults to workflow name)
        dimension: Embedding dimension

    Returns:
        (universe, engine) tuple
    """
    universe = RKHSUniverse(
        name=name or config.name,
        description=config.description,
        dimension=dimension,
        kernel_type="cosine",
        kernel_params={},
        nodes={},
        edges=[],
        metadata={
            "workflow_type": config.name,
            "created_at": datetime.now().isoformat()
        }
    )

    engine = RKHSWorkflowEngine(universe, config)

    return universe, engine


def save_workflow_universe(universe: RKHSUniverse, filepath: str):
    """Save workflow universe to JSON"""
    from multiverse_viewer import save_rkhs_universe
    save_rkhs_universe(universe, filepath)


def load_workflow_universe(filepath: str, config: WorkflowConfig) -> Tuple[RKHSUniverse, RKHSWorkflowEngine]:
    """Load workflow universe from JSON"""
    from multiverse_viewer import load_rkhs_universe
    universe = load_rkhs_universe(filepath)
    engine = RKHSWorkflowEngine(universe, config)
    return universe, engine
