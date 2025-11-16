import streamlit as st
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import cosine
import networkx as nx
from pathlib import Path
import os
import sys
import warnings
import logging

# Suppress plotly deprecation warnings from Streamlit internal code
warnings.filterwarnings('ignore', message='.*keyword arguments.*deprecated.*config.*')

# Suppress Streamlit's deprecation warnings about plotly keyword arguments
class DeprecationWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "keyword arguments have been deprecated" not in msg and "Use `config` instead" not in msg

# Apply filter to streamlit logger and all subloggers
for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner']:
    logger = logging.getLogger(logger_name)
    logger.addFilter(DeprecationWarningFilter())

# Add parent directory to path for shared modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Authentication integration
try:
    from shared.auth import is_authenticated, get_user_info, authenticate, get_shared_auth
    AUTH_AVAILABLE = True
except ImportError as e:
    # Log the error for debugging
    import traceback
    print(f"Auth import error: {e}")
    print(f"Project root: {project_root}")
    print(f"sys.path: {sys.path[:3]}")
    traceback.print_exc()
    AUTH_AVAILABLE = False

# LLM integration
try:
    import litellm
    from dotenv import load_dotenv
    load_dotenv()
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ============================================================================
# RKHS FORMALIZATION DATA STRUCTURES
# ============================================================================

@dataclass
class RKHSNode:
    """Node in the Reproducing Kernel Hilbert Space multiverse"""
    id: str
    position: List[float]  # Coordinates in RKHS
    content: Dict[str, Any]  # Semantic content
    metadata: Dict[str, Any]
    kernel_features: List[float]  # Feature representation
    timestamp: str
    parent_ids: List[str]
    children_ids: List[str]
    
@dataclass
class RKHSEdge:
    """Edge representing transitions/relationships in RKHS"""
    source_id: str
    target_id: str
    weight: float
    kernel_similarity: float
    transition_type: str
    metadata: Dict[str, Any]

@dataclass
class RKHSUniverse:
    """Complete multiverse representation in RKHS format"""
    name: str
    description: str
    dimension: int  # Dimensionality of the RKHS
    kernel_type: str  # e.g., "rbf", "polynomial", "linear"
    kernel_params: Dict[str, float]
    nodes: Dict[str, RKHSNode]
    edges: List[RKHSEdge]
    metadata: Dict[str, Any]
    version: str = "1.0"

# ============================================================================
# ENVIRONMENT & URL UTILITIES
# ============================================================================

def is_production_env() -> bool:
    """Check if running in production environment"""
    import socket
    hostname = socket.gethostname()
    return (
        'book-publisher' in hostname.lower() or
        os.getenv('ENV') == 'production' or
        os.getenv('STREAMLIT_ENV') == 'production'
    )

def get_codexes_factory_url() -> str:
    """Get the correct Codexes Factory URL based on environment"""
    return "https://codexes.xtuff.ai" if is_production_env() else "http://localhost:8502"

def get_home_url() -> str:
    """Get the correct home/auth URL based on environment"""
    return "https://xtuff.ai" if is_production_env() else "http://localhost:8500"

# ============================================================================
# AUTHENTICATION UTILITIES
# ============================================================================

def get_tier_info(tier: str = "free") -> Dict[str, Any]:
    """Get tier limits and features"""
    tiers = {
        "free": {
            "name": "Free",
            "max_nodes": 100,
            "max_universes": 1,
            "max_node_size_kb": 10,
            "features": ["Browse and explore", "View example timelines", "Fork up to 100 nodes"],
            "price": "$0/month"
        },
        "consumer": {
            "name": "Consumer",
            "max_nodes": 1000,
            "max_universes": 10,
            "max_node_size_kb": 50,
            "features": ["Create unlimited forks", "Save up to 10 multiverses", "LLM-powered text operations", "Export to JSON"],
            "price": "$9.99/month"
        },
        "pro": {
            "name": "Pro",
            "max_nodes": 10000,
            "max_universes": 100,
            "max_node_size_kb": 500,
            "features": ["Advanced analytics", "Kernel matrix visualization", "API access", "Priority support", "Collaborative multiverses"],
            "price": "$49.99/month"
        },
        "enterprise": {
            "name": "Enterprise",
            "max_nodes": -1,  # Unlimited
            "max_universes": -1,  # Unlimited
            "max_node_size_kb": -1,  # Unlimited
            "features": ["Unlimited everything", "Custom integrations", "Dedicated support", "On-premise deployment", "SLA guarantees"],
            "price": "Custom pricing"
        }
    }
    return tiers.get(tier, tiers["free"])

def check_tier_limit(operation: str, current_count: int, tier: str = "free") -> bool:
    """Check if operation is within tier limits"""
    if not AUTH_AVAILABLE:
        return True

    tier_info = get_tier_info(tier)

    if operation == "nodes":
        max_allowed = tier_info["max_nodes"]
        if max_allowed == -1:  # Unlimited
            return True

        if current_count >= max_allowed:
            st.error(f"🚫 **Tier Limit Reached**")
            st.warning(f"Your {tier_info['name']} plan allows up to {max_allowed:,} nodes. You currently have {current_count:,} nodes.")
            st.info("💎 Upgrade your plan to create more nodes!")
            if st.button("⬆️ View Upgrade Options"):
                st.markdown(f"[View pricing plans]({get_home_url()}/pricing)")
            return False

    return True

def require_auth(operation: str = "edit or create content") -> bool:
    """Check if user is authenticated for write operations. Returns True if auth passed."""
    if not AUTH_AVAILABLE:
        # If auth system not available, allow operations (dev mode)
        return True

    if is_authenticated():
        return True

    # Show auth requirement
    st.warning(f"🔒 **Authentication Required**")
    st.info(f"You need to be logged in to {operation}. View-only access is available without login.")

    if st.button("🔑 Log In", key=f"login_{operation}"):
        # Redirect to auth page or show login form
        st.markdown(f"[Click here to log in]({get_home_url()})")

    return False

# ============================================================================
<<<<<<< HEAD
=======
# SAVE & PUBLISH UTILITY
# ============================================================================

def render_save_and_publish(universe: RKHSUniverse):
    """Render Save & Publish section for editing tabs"""
    st.divider()
    st.subheader("💾 Save & Publish Universe")

    save_col1, save_col2 = st.columns(2)

    with save_col1:
        st.markdown("**💾 Save Locally**")
        filename = st.text_input(
            "Filename",
            value=f"{universe.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
            help="Name for the saved universe file (local download)",
            key=f"save_filename_{id(universe)}"
        )

        if st.button("💾 Download Universe", use_container_width=True, key=f"download_{id(universe)}"):
            # Create download directly
            universe_json = json.dumps(
                {
                    'name': universe.name,
                    'description': universe.description,
                    'dimension': universe.dimension,
                    'kernel_type': universe.kernel_type,
                    'kernel_params': universe.kernel_params,
                    'nodes': {nid: asdict(node) for nid, node in universe.nodes.items()},
                    'edges': [asdict(edge) for edge in universe.edges],
                    'metadata': universe.metadata,
                    'version': universe.version
                },
                indent=2
            )
            st.download_button(
                "📥 Download JSON",
                universe_json,
                file_name=filename,
                mime="application/json",
                use_container_width=True,
                key=f"download_btn_{id(universe)}"
            )

    with save_col2:
        st.markdown("**🌐 Publish to Catalog**")
        if AUTH_AVAILABLE and is_authenticated():
            user_info = get_user_info()
            st.caption(f"Publishing as: {user_info['user_name']}")

            if st.button("🚀 Publish to Codexes Catalog", use_container_width=True, key=f"publish_{id(universe)}"):
                # Create catalog entry
                catalog_entry = {
                    'product_type': 'multiverse',
                    'name': universe.name,
                    'description': universe.description,
                    'nodes': len(universe.nodes),
                    'edges': len(universe.edges),
                    'username': user_info['username'],
                    'user_name': user_info['user_name'],
                    'published_date': datetime.now().isoformat(),
                    'dimension': universe.dimension,
                    'kernel_type': universe.kernel_type,
                    'file_path': f"multiverses/{user_info['username']}/{universe.name.replace(' ', '_').lower()}.rkhs.json"
                }

                # Save universe file
                catalog_base = project_root / "nimble" / "codexes-factory" / "resources" / "products" / "multiverses"
                user_dir = catalog_base / user_info['username']
                user_dir.mkdir(parents=True, exist_ok=True)

                universe_filename = f"{universe.name.replace(' ', '_').lower()}.rkhs.json"
                universe_path = user_dir / universe_filename

                # Save the universe
                save_rkhs_universe(universe, str(universe_path))

                # Save catalog entry
                catalog_entry_path = user_dir / f"{universe.name.replace(' ', '_').lower()}_catalog.json"
                with open(catalog_entry_path, 'w') as f:
                    json.dump(catalog_entry, f, indent=2)

                st.success(f"✅ Published to catalog!")
                st.json(catalog_entry)
                st.info(f"📁 Saved to: {universe_path.relative_to(project_root)}")
        else:
            st.info("🔒 Login required to publish to catalog")
            if st.button("🔑 Login to Publish", key=f"login_publish_{id(universe)}"):
                st.markdown(f"[Login here]({get_home_url()})")

# ============================================================================
>>>>>>> feature/feature/hilbert-ui-enhancements
# UTILITY FUNCTIONS
# ============================================================================

def compute_kernel_similarity(feat1: List[float], feat2: List[float], 
                              kernel_type: str = "rbf", gamma: float = 1.0) -> float:
    """Compute kernel similarity between two feature vectors"""
    feat1 = np.array(feat1)
    feat2 = np.array(feat2)
    
    if kernel_type == "rbf":
        return np.exp(-gamma * np.linalg.norm(feat1 - feat2) ** 2)
    elif kernel_type == "linear":
        return np.dot(feat1, feat2)
    elif kernel_type == "polynomial":
        return (np.dot(feat1, feat2) + 1) ** 2
    else:
        return 1.0 - cosine(feat1, feat2)

def load_rkhs_universe(file_path: str) -> Optional[RKHSUniverse]:
    """Load RKHS universe from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        nodes = {
            node_id: RKHSNode(**node_data) 
            for node_id, node_data in data['nodes'].items()
        }
        
        edges = [RKHSEdge(**edge_data) for edge_data in data['edges']]
        
        universe = RKHSUniverse(
            name=data['name'],
            description=data['description'],
            dimension=data['dimension'],
            kernel_type=data['kernel_type'],
            kernel_params=data['kernel_params'],
            nodes=nodes,
            edges=edges,
            metadata=data.get('metadata', {}),
            version=data.get('version', '1.0')
        )
        
        return universe
    except Exception as e:
        st.error(f"Error loading universe: {e}")
        return None

def save_rkhs_universe(universe: RKHSUniverse, file_path: str):
    """Save RKHS universe to JSON file"""
    data = {
        'name': universe.name,
        'description': universe.description,
        'dimension': universe.dimension,
        'kernel_type': universe.kernel_type,
        'kernel_params': universe.kernel_params,
        'nodes': {node_id: asdict(node) for node_id, node in universe.nodes.items()},
        'edges': [asdict(edge) for edge in universe.edges],
        'metadata': universe.metadata,
        'version': universe.version
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def apply_auto_filter_if_needed(universe: RKHSUniverse, threshold: int = 100) -> None:
    """Apply automatic filter to large universes for performance"""
    if len(universe.nodes) > threshold:
        # Get first N nodes (sorted by ID for consistency)
        node_ids = sorted(list(universe.nodes.keys()))[:threshold]
        st.session_state.filtered_nodes = set(node_ids)
        st.session_state.auto_filter_active = True
        st.info(
            f"ℹ️ **Auto-filter applied**: This universe has {len(universe.nodes):,} nodes. "
            f"Automatically limited to {threshold} nodes for performance. "
            f"You can adjust or clear this filter in the 🎯 Filter tab."
        )
    else:
        st.session_state.auto_filter_active = False

def get_user_multiverse_path() -> Optional[Path]:
    """Get the path to the user's saved multiverse, if authenticated"""
    if not AUTH_AVAILABLE or not is_authenticated():
        return None

    user_info = get_user_info()
    username = user_info.get("username", user_info.get("user_email", "")).replace("@", "_").replace(".", "_")

    if not username:
        return None

    # Store in multiverse_viewer directory
    user_data_dir = Path(__file__).parent / "user_data"
    user_data_dir.mkdir(exist_ok=True)

    return user_data_dir / f"{username}_multiverse.json"

def save_user_multiverse(universe: RKHSUniverse) -> None:
    """Save user's current multiverse (first 100 nodes only) for authenticated users"""
    user_path = get_user_multiverse_path()
    if not user_path:
        return

    # Limit to first 100 nodes for performance
    node_ids = sorted(list(universe.nodes.keys()))[:100]
    limited_nodes = {node_id: universe.nodes[node_id] for node_id in node_ids}

    # Filter edges to only those between kept nodes
    limited_edges = [
        edge for edge in universe.edges
        if edge.source_id in limited_nodes and edge.target_id in limited_nodes
    ]

    # Create limited universe
    limited_universe = RKHSUniverse(
        name=universe.name,
        description=universe.description,
        dimension=universe.dimension,
        kernel_type=universe.kernel_type,
        kernel_params=universe.kernel_params,
        nodes=limited_nodes,
        edges=limited_edges,
        metadata={**universe.metadata, "limited_to_100": True},
        version=universe.version
    )

    save_rkhs_universe(limited_universe, str(user_path))

def load_user_multiverse() -> Optional[RKHSUniverse]:
    """Load user's saved multiverse if available"""
    user_path = get_user_multiverse_path()
    if not user_path or not user_path.exists():
        return None

    return load_rkhs_universe(str(user_path))

def create_sample_universe(n_nodes: int = 100) -> RKHSUniverse:
    """Create a sample RKHS universe for demonstration"""
    nodes = {}
    edges = []
    
    # Create nodes in a 3D RKHS
    for i in range(n_nodes):
        theta = 2 * np.pi * i / n_nodes
        phi = np.pi * (i % 10) / 10
        
        position = [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi)
        ]
        
        kernel_features = position + [np.random.randn() for _ in range(5)]
        
        node = RKHSNode(
            id=f"node_{i:04d}",
            position=position,
            content={
                "title": f"State {i}",
                "description": f"Description of multiverse state {i}",
                "properties": {"energy": np.random.rand(), "entropy": np.random.rand()}
            },
            metadata={"created": datetime.now().isoformat(), "type": "state"},
            kernel_features=kernel_features,
            timestamp=datetime.now().isoformat(),
            parent_ids=[f"node_{(i-1):04d}"] if i > 0 else [],
            children_ids=[f"node_{(i+1):04d}"] if i < n_nodes - 1 else []
        )
        nodes[node.id] = node
        
        # Create edges
        if i > 0:
            source = nodes[f"node_{(i-1):04d}"]
            sim = compute_kernel_similarity(source.kernel_features, kernel_features)
            edge = RKHSEdge(
                source_id=source.id,
                target_id=node.id,
                weight=1.0,
                kernel_similarity=sim,
                transition_type="temporal",
                metadata={}
            )
            edges.append(edge)
    
    universe = RKHSUniverse(
        name="Sample Universe",
        description="A demonstration RKHS multiverse",
        dimension=8,
        kernel_type="rbf",
        kernel_params={"gamma": 1.0},
        nodes=nodes,
        edges=edges,
        metadata={"created": datetime.now().isoformat()}
    )
    
    return universe

# ============================================================================
# TEXT EXTRACTION AND LLM FUNCTIONS
# ============================================================================

def extract_text_from_nodes(nodes: Dict[str, RKHSNode], node_ids: Set[str]) -> str:
    """Extract all text content from selected nodes"""
    text_parts = []

    for node_id in sorted(node_ids):
        if node_id not in nodes:
            continue

        node = nodes[node_id]
        text_parts.append(f"=== {node_id} ===")

        # Title
        if 'title' in node.content:
            text_parts.append(f"Title: {node.content['title']}")

        # Description
        if 'description' in node.content:
            text_parts.append(f"Description: {node.content['description']}")

        # Properties
        if 'properties' in node.content:
            text_parts.append("Properties:")
            for key, value in node.content['properties'].items():
                text_parts.append(f"  {key}: {value}")

        text_parts.append("")  # Blank line between nodes

    return "\n".join(text_parts)

def morph_text_with_llm(text: str, prompt: str, model: str = "anthropic/claude-3-5-haiku-20241022") -> Optional[str]:
    """Apply LLM transformation to text"""
    if not LLM_AVAILABLE:
        return None

    try:
        full_prompt = f"""{prompt}

IMPORTANT: Provide ONLY the transformed text. Do not include:
- Introductory phrases ("Here's the transformed text:", "Sure, I can help with that", etc.)
- Explanations or commentary
- Pleasantries or acknowledgments
- Closing remarks

Just output the direct result of the transformation.

Text to transform:
{text}"""

        # Call litellm directly
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=4000,
            temperature=0.7
        )

        # Extract content from response
        if response and response.choices:
            return response.choices[0].message.content
        return None

    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return None

def apply_llm_morph_to_nodes(
    universe: RKHSUniverse,
    node_ids: Set[str],
    prompt: str,
    field: str = "description",
    model: str = "anthropic/claude-3-5-haiku-20241022",
    batch_size: int = 5
) -> int:
    """Apply LLM transformation to multiple nodes intelligently"""
    if not LLM_AVAILABLE:
        st.error("LLM features not available. Install nimble-llm-caller.")
        return 0

    modified_count = 0
    node_list = list(node_ids)

    # Process in batches for cost/performance
    for i in range(0, len(node_list), batch_size):
        batch = node_list[i:i+batch_size]

        for node_id in batch:
            if node_id not in universe.nodes:
                continue

            node = universe.nodes[node_id]

            # Get current text
            if field == "description":
                current_text = node.content.get('description', '')
            elif field == "title":
                current_text = node.content.get('title', '')
            else:
                continue

            if not current_text:
                continue

            # Transform with LLM
            transformed = morph_text_with_llm(current_text, prompt, model)

            if transformed:
                # Update node
                if field == "description":
                    node.content['description'] = transformed
                elif field == "title":
                    node.content['title'] = transformed

                modified_count += 1

    return modified_count

# ============================================================================
# LIFE TIMELINE FUNCTIONS
# ============================================================================

def get_life_domains() -> List[str]:
    """Get standard life domains"""
    return [
        "education",
        "career",
        "relationships",
        "family",
        "health",
        "life_milestone",
        "external_event",
        "personal_growth",
        "financial",
        "creative"
    ]

def get_domain_color(domain: str) -> str:
    """Get color for life domain"""
    colors = {
        "education": "#4A90E2",      # Blue
        "career": "#F5A623",          # Orange
        "relationships": "#E91E63",   # Pink
        "family": "#9B59B6",          # Purple
        "health": "#27AE60",          # Green
        "life_milestone": "#E74C3C",  # Red
        "external_event": "#95A5A6",  # Gray
        "personal_growth": "#3498DB",  # Light Blue
        "financial": "#F39C12",       # Gold
        "creative": "#1ABC9C"         # Teal
    }
    return colors.get(domain, "#7F8C8D")

def create_life_event_node(
    event_id: str,
    title: str,
    description: str,
    year: int,
    age: int,
    importance: float,
    happiness: float,
    domain: str,
    dimension: int,
    parent_ids: List[str] = None
) -> RKHSNode:
    """Create a life event node"""
    if parent_ids is None:
        parent_ids = []

    # Position: [time_progress, happiness, achievement]
    time_progress = (year - 1990) / 10.0  # Normalize to reasonable range
    position = [time_progress, happiness, importance]

    # Kernel features (extend to dimension)
    kernel_features = position + [0.0] * (dimension - 3)
    if dimension > 3:
        kernel_features[3] = importance
    if dimension > 4:
        kernel_features[4] = happiness

    node = RKHSNode(
        id=event_id,
        position=position,
        content={
            "title": title,
            "description": description,
            "properties": {
                "year": year,
                "age": age,
                "importance": importance,
                "happiness": happiness,
                "domain": domain
            }
        },
        metadata={
            "date": f"{year}-01-01",
            "type": domain
        },
        kernel_features=kernel_features,
        timestamp=datetime.now().isoformat(),
        parent_ids=parent_ids,
        children_ids=[]
    )

    return node

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_3d_network_viz(universe: RKHSUniverse,
                          selected_nodes: Optional[Set[str]] = None,
                          layout_type: str = "position",
                          color_by: str = "energy") -> go.Figure:
    """Create interactive 3D network visualization"""
    
    if selected_nodes is None:
        selected_nodes = set(universe.nodes.keys())
    
    # Filter nodes and edges
    nodes_to_show = {nid: node for nid, node in universe.nodes.items() 
                     if nid in selected_nodes}
    edges_to_show = [e for e in universe.edges 
                     if e.source_id in selected_nodes and e.target_id in selected_nodes]
    
    # Prepare node positions
    if layout_type == "position":
        node_positions = {nid: node.position[:3] for nid, node in nodes_to_show.items()}
    else:
        # Use networkx for alternative layouts
        G = nx.DiGraph()
        for nid in nodes_to_show.keys():
            G.add_node(nid)
        for edge in edges_to_show:
            G.add_edge(edge.source_id, edge.target_id)
        
        if len(G.nodes()) > 1:
            pos_2d = nx.spring_layout(G, dim=2, seed=42)
            node_positions = {nid: [pos[0], pos[1], 0] for nid, pos in pos_2d.items()}
        else:
            node_positions = {nid: [0, 0, 0] for nid in nodes_to_show.keys()}
    
    # Create edge traces
    edge_traces = []
    for edge in edges_to_show:
        if edge.source_id in node_positions and edge.target_id in node_positions:
            x0, y0, z0 = node_positions[edge.source_id]
            x1, y1, z1 = node_positions[edge.target_id]

            # Clamp alpha to reasonable range (0.1 to 1.0) for visibility
            alpha = max(0.1, min(1.0, edge.kernel_similarity))

            edge_trace = go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(
                    color=f'rgba(125, 125, 125, {alpha})',
                    width=edge.weight * 2
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_colors = []
    
    for nid, pos in node_positions.items():
        node = nodes_to_show[nid]
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_z.append(pos[2])

        # Enhanced hover text with more details
        text = f"<b>{node.content.get('title', nid)}</b><br>"
        text += f"<i>ID: {nid}</i><br>"
        text += "<br>"

        # Add description if available
        if 'description' in node.content and node.content['description']:
            desc = node.content['description']
            # Truncate long descriptions
            if len(desc) > 100:
                desc = desc[:97] + "..."
            text += f"Description: {desc}<br><br>"

        # Position in RKHS
        text += f"<b>RKHS Position:</b><br>"
        text += f"X: {pos[0]:.3f}, Y: {pos[1]:.3f}, Z: {pos[2]:.3f}<br><br>"

        # Properties
        if 'properties' in node.content:
            text += f"<b>Properties:</b><br>"
            for k, v in node.content['properties'].items():
                if isinstance(v, (int, float)):
                    # Format integers (or count-like values) without decimals
                    if isinstance(v, int) or (isinstance(v, float) and v == int(v)) or \
                       any(keyword in k.lower() for keyword in ['count', 'pages', 'words', 'tokens', 'length']):
                        text += f"{k.capitalize()}: {int(v)}<br>"
                    else:
                        text += f"{k.capitalize()}: {v:.3f}<br>"
                else:
                    text += f"{k.capitalize()}: {v}<br>"
            text += "<br>"

        # Relationships
        text += f"<b>Connections:</b><br>"
        text += f"Parents: {len(node.parent_ids)}<br>"
        text += f"Children: {len(node.children_ids)}<br>"

        # Timestamp
        if node.timestamp:
            text += f"<br>Created: {node.timestamp[:10]}"

        node_text.append(text)

        # Color by specified attribute
        if color_by == "domain":
            # Color by life domain
            domain = node.content.get('properties', {}).get('domain', 'unknown')
            node_colors.append(domain)
        elif color_by == "happiness":
            # Color by happiness level
            happiness = node.content.get('properties', {}).get('happiness', 0.5)
            node_colors.append(happiness)
        elif color_by == "importance":
            # Color by importance
            importance = node.content.get('properties', {}).get('importance', 0.5)
            node_colors.append(importance)
        else:
            # Default: energy
            if 'properties' in node.content and 'energy' in node.content['properties']:
                node_colors.append(node.content['properties']['energy'])
            else:
                node_colors.append(0.5)
    
    # Handle categorical vs numeric colors
    if color_by == "domain":
        # Map domains to numeric values for coloring
        unique_domains = list(set(node_colors))
        domain_to_num = {d: i for i, d in enumerate(unique_domains)}
        numeric_colors = [domain_to_num[d] for d in node_colors]

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            marker=dict(
                size=8,
                color=numeric_colors,
                colorscale='Plotly3',
                showscale=False,  # Don't show scale for categorical
                line=dict(color='white', width=0.5)
            ),
            text=node_text,
            hoverinfo='text',
            name='Nodes'
        )
    else:
        # Numeric coloring
        color_label = color_by.capitalize()
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            marker=dict(
                size=8,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=f"<b>{color_label}</b><br>(node property)",
                        side="right"
                    )
                ),
                line=dict(color='white', width=0.5)
            ),
            text=node_text,
            hoverinfo='text',
            name='Nodes'
        )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=f"{universe.name} - Network View ({len(nodes_to_show)} nodes)",
        showlegend=False,
        hovermode='closest',
        scene=dict(
            xaxis=dict(
                showbackground=False,
                title='<b>X Axis</b><br>(1st RKHS dimension)'
            ),
            yaxis=dict(
                showbackground=False,
                title='<b>Y Axis</b><br>(2nd RKHS dimension)'
            ),
            zaxis=dict(
                showbackground=False,
                title='<b>Z Axis</b><br>(3rd RKHS dimension)'
            ),
        ),
        height=700
    )
    
    return fig

def create_2d_projection(universe: RKHSUniverse, 
                        selected_nodes: Optional[Set[str]] = None) -> go.Figure:
    """Create 2D projection of RKHS space"""
    
    if selected_nodes is None:
        selected_nodes = set(universe.nodes.keys())
    
    nodes_to_show = {nid: node for nid, node in universe.nodes.items() 
                     if nid in selected_nodes}
    
    x_coords = []
    y_coords = []
    texts = []
    colors = []
    
    for nid, node in nodes_to_show.items():
        pos = node.position
        x_coords.append(pos[0])
        y_coords.append(pos[1] if len(pos) > 1 else 0)
        texts.append(node.content.get('title', nid))
        
        if 'properties' in node.content and 'energy' in node.content['properties']:
            colors.append(node.content['properties']['energy'])
        else:
            colors.append(0.5)
    
    fig = go.Figure(data=go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(
            size=10,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            line=dict(color='white', width=1)
        ),
        text=texts,
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{universe.name} - 2D Projection",
        xaxis_title="<b>1st RKHS Dimension</b> (semantic embedding space)",
        yaxis_title="<b>2nd RKHS Dimension</b> (semantic embedding space)",
        height=600,
        hovermode='closest'
    )
    
    return fig

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="xtuff.ai - Your Multiverse Explorer",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Hero section with gradient styling
    st.markdown("""
    <style>
    .hero-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #666;
        line-height: 1.6;
        max-width: 800px;
    }
    </style>

    <div class="hero-container">
        <div class="hero-title">🌌 Your Multiverse Explorer</div>
        <div class="hero-subtitle">
            Navigate infinite possibilities through Reproducing Kernel Hilbert Space.
            Visualize decisions, explore alternative timelines, and materialize ideas
            across parallel universes—all grounded in rigorous mathematical frameworks.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with navigation
    with st.sidebar:
        # Authentication Section
        st.markdown("### 👤 Account")

        if AUTH_AVAILABLE and is_authenticated():
            user_info = get_user_info()
            user_name = user_info.get("user_name", "User")
            user_tier = user_info.get("subscription_tier", "free")

            st.success(f"✅ Logged in as **{user_name}**")
            st.caption(f"Tier: **{user_tier.title()}**")

            # Show tier limits
            tier_info = get_tier_info(user_tier)
            st.caption(f"Node limit: {tier_info['max_nodes']:,}")

            if st.button("🔓 Logout", use_container_width=True):
                st.markdown(f"[Logout here]({get_home_url()}/logout)")

            if user_tier == "free":
                st.info("💎 Upgrade for unlimited nodes and advanced features!")
                if st.button("⬆️ Upgrade Plan", use_container_width=True):
                    st.markdown(f"[View Plans]({get_home_url()}/pricing)")
        else:
            st.info("🔓 **Browse freely** or login to save your work")

            # Show login/register form inline
            with st.expander("🔑 Login / Register", expanded=False):
                tab1, tab2 = st.tabs(["Login", "Register"])

                with tab1:
                    st.markdown("##### Login")
                    login_email = st.text_input("Email", key="login_email")
                    login_password = st.text_input("Password", type="password", key="login_password")

                    if st.button("🔑 Login", key="login_submit", use_container_width=True):
                        if not login_email or not login_password:
                            st.warning("⚠️ Please enter both email and password")
                        elif AUTH_AVAILABLE:
                            try:
                                result = authenticate(login_email, login_password)
                                if result:
                                    st.success("✅ Login successful!")
                                    st.rerun()
                                else:
                                    st.error("❌ Invalid credentials. Please check your email and password.")
                            except Exception as e:
                                st.error(f"Login error: {e}")
                                st.caption("If you don't have an account, please register via Codexes Factory.")
                        else:
                            st.error("⚠️ **Authentication System Unavailable**")
                            st.info("The authentication system could not be loaded. This is likely a temporary issue.")
                            codexes_url = get_codexes_factory_url()
                            st.markdown(f"""
                            **Alternative:**
                            - Visit [Codexes Factory]({codexes_url}) to log in
                            - Or refresh this page to try again
                            """)
                            st.caption(f"Debug: AUTH_AVAILABLE={AUTH_AVAILABLE}, project_root={project_root}")

                    st.markdown("---")
                    codexes_url = get_codexes_factory_url()
                    st.caption(f"Or login via [Codexes Factory]({codexes_url})")

                with tab2:
                    st.markdown("##### Create Your Account")
                    st.caption("Start exploring multiverses with a free account")

                    reg_name = st.text_input("Full Name", key="reg_name")
                    reg_email = st.text_input("Email Address", key="reg_email")
                    reg_password = st.text_input("Password", type="password", key="reg_password")
                    reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")

                    st.caption("Password must be at least 8 characters")

                    if st.button("✨ Create Account", key="register_submit", use_container_width=True):
                        # Validation
                        if not all([reg_name, reg_email, reg_password, reg_password_confirm]):
                            st.warning("⚠️ Please fill in all fields")
                        elif len(reg_password) < 8:
                            st.error("❌ Password must be at least 8 characters")
                        elif reg_password != reg_password_confirm:
                            st.error("❌ Passwords do not match")
                        elif "@" not in reg_email or "." not in reg_email:
                            st.error("❌ Please enter a valid email address")
                        elif AUTH_AVAILABLE:
                            try:
                                st.info("🔄 Creating your account...")

                                # Generate username from email (before @)
                                reg_username = reg_email.split('@')[0].lower()

                                # Register via shared auth system
                                shared_auth = get_shared_auth()
                                success = shared_auth.register_user(
                                    username=reg_username,
                                    password=reg_password,
                                    email=reg_email,
                                    name=reg_name,
                                    role="user"
                                )

                                if success:
                                    # Auto-login after successful registration
                                    auth_success, message = authenticate(reg_username, reg_password)
                                    if auth_success:
                                        user_info = get_user_info()
                                        st.session_state.username = user_info.get('username')
                                        st.session_state.user_name = user_info.get('user_name')
                                        st.session_state.user_email = user_info.get('user_email')
                                        st.session_state.user_role = user_info.get('user_role', 'user')
                                        st.success(f"✅ Account created! Welcome, {reg_name}!")
                                        st.rerun()
                                    else:
                                        st.success("✅ Account created! Please login.")
                                else:
                                    st.error("❌ Registration failed. Username or email may already exist.")
                            except Exception as e:
                                st.error(f"❌ Registration error: {e}")
                        else:
                            st.error("⚠️ **Authentication System Unavailable**")
                            codexes_url = get_codexes_factory_url()
                            st.info(f"Please try again later or visit [Codexes Factory]({codexes_url})")

                    st.markdown("---")
                    st.caption("Already have an account? Switch to the Login tab")

        st.markdown("---")

        st.markdown("### 🌐 Related Apps")
        codexes_url = get_codexes_factory_url()
        st.markdown(f"""
        📚 [Codexes Factory]({codexes_url}) - Book publishing platform
        """)

        st.caption("_Multiverse Viewer is your home page (port 8500)_")

        st.markdown("---")

        st.markdown("### 📖 About")
        st.markdown("""
        This tool uses **RKHS** (Reproducing Kernel Hilbert Space) to represent
        and navigate multiverses of possibilities. Each node represents a state,
        decision, or idea, positioned in a high-dimensional space where distance
        reflects semantic similarity.

        **Getting Started:**
        1. Explore the example timeline (auto-loaded)
        2. Fork nodes to create variations
        3. Filter and analyze your multiverse
        4. Create your own universes
        """)

    # Initialize session state
    if 'universe' not in st.session_state:
        st.session_state.universe = None
    if 'forked_nodes' not in st.session_state:
        st.session_state.forked_nodes = set()
    if 'traversed_nodes' not in st.session_state:
        st.session_state.traversed_nodes = set()
    if 'filtered_nodes' not in st.session_state:
        st.session_state.filtered_nodes = None
    if 'auto_filter_active' not in st.session_state:
        st.session_state.auto_filter_active = False
    if 'morph_nodes' not in st.session_state:
        st.session_state.morph_nodes = None

    # Auto-load multiverse on first run
    if 'initial_load_done' not in st.session_state:
        universe = None

        # Try to load user's saved multiverse first (if authenticated)
        if AUTH_AVAILABLE and is_authenticated():
            universe = load_user_multiverse()
            if universe:
                st.success("✅ Loaded your saved multiverse (first 100 nodes)")

        # Don't auto-load example - let user choose from Load tab
        # (Auto-loading small universes causes min_value errors in other tabs)

        # Set the loaded universe (only if user had one saved)
        if universe:
            st.session_state.universe = universe
            apply_auto_filter_if_needed(universe)

        st.session_state.initial_load_done = True

    # Create tabs - Load, Create, Explore are first for better orientation
    tabs = st.tabs([
        "🔍 Explore",
        "🌟 Create",
        "📂 Load",
        "🔍 Browse",
        "🔱 Fork",
        "🎯 Filter",
        "📝 Text/LLM",
        "📅 Life Timeline",
<<<<<<< HEAD
        "🔢 Mathematics"
=======
        "🔢 Maths"
>>>>>>> feature/feature/hilbert-ui-enhancements
    ])

    # ========================================================================
    # TAB 0: EXPLORE (HOME PAGE)
    # ========================================================================
    with tabs[0]:
        st.header("Explore Universe")

        if st.session_state.universe is None:
            st.warning("⚠️ No universe loaded. Go to the 📂 Load tab to get started!")
        else:
            universe = st.session_state.universe

<<<<<<< HEAD
            # Show universe stats at the top
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Total Nodes", len(universe.nodes))
            with stat_col2:
                st.metric("Total Edges", len(universe.edges))
            with stat_col3:
                forked_count = len(st.session_state.forked_nodes) if hasattr(st.session_state, 'forked_nodes') else 0
                st.metric("Forked Nodes", forked_count)

            st.divider()

            # Quick filter row
            filter_col1, filter_col2 = st.columns([3, 1])
            with filter_col1:
                quick_filter = st.text_input(
                    "🔍 Quick Filter",
                    placeholder="Search nodes by title or description...",
                    help="Type to filter nodes by title or description content"
                )
            with filter_col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if quick_filter and st.button("Clear"):
                    st.rerun()

            col1, col2, col3 = st.columns(3)

            with col1:
                viz_type = st.selectbox(
                    "Graph Type",
                    ["3D Network", "2D Projection", "Kernel Matrix"],
                    help="**3D Network**: Shows nodes as points in 3D space connected by edges. "
                         "Great for seeing spatial relationships and clusters.\n\n"
                         "**2D Projection**: Flattens the space to 2D for easier viewing. "
                         "Useful for identifying patterns when 3D is too complex.\n\n"
                         "**Kernel Matrix**: Heatmap showing similarity between all node pairs. "
                         "Reveals global structure and similarity patterns."
                )

            with col2:
                node_set = st.selectbox(
                    "Node Set",
                    ["All Nodes", "Filtered", "Traversed", "Forked", "Sample"],
                    help="**All Nodes**: Show every node in the universe (may be slow for large universes).\n\n"
                         "**Filtered**: Show only nodes matching your filter criteria (set in 🎯 Filter tab).\n\n"
                         "**Traversed**: Show nodes you've marked as visited (marked in 🔍 Browse tab).\n\n"
                         "**Forked**: Show only nodes you've created by forking (created in 🔱 Fork tab).\n\n"
                         "**Sample**: Randomly sample a subset of nodes for quick exploration."
                )

            with col3:
=======
            # Compact controls at top - one row
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

            with ctrl_col1:
                viz_type = st.selectbox(
                    "Graph Type",
                    ["2D Projection", "3D Network", "Kernel Matrix"],
                    help="**2D Projection**: Flattens the space to 2D for easier viewing. Default for quick exploration.\n\n"
                         "**3D Network**: Shows nodes as points in 3D space connected by edges.\n\n"
                         "**Kernel Matrix**: Heatmap showing similarity between all node pairs."
                )

            with ctrl_col2:
                node_set = st.selectbox(
                    "Node Set",
                    ["All Nodes", "Filtered", "Traversed", "Forked", "Sample"],
                    help="**All Nodes**: Show every node.\n\n"
                         "**Filtered**: Nodes matching filter criteria.\n\n"
                         "**Traversed**: Nodes you've marked.\n\n"
                         "**Forked**: Nodes you've created.\n\n"
                         "**Sample**: Random subset."
                )

            with ctrl_col3:
>>>>>>> feature/feature/hilbert-ui-enhancements
                if node_set == "Sample":
                    sample_size = st.number_input(
                        "Sample Size",
                        min_value=10,
                        max_value=min(1000, len(universe.nodes)),
                        value=min(200, len(universe.nodes)),
<<<<<<< HEAD
                        help="Number of random nodes to display. Smaller values load faster."
                    )
=======
                        help="Number of random nodes to display."
                    )
                else:
                    sample_size = 200  # Default for later use

            # Quick filter (optional)
            quick_filter = st.text_input(
                "🔍 Quick Search",
                placeholder="Search nodes by title or description...",
                help="Type to filter nodes by title or description content"
            )
>>>>>>> feature/feature/hilbert-ui-enhancements

            # Determine which nodes to show
            if node_set == "All Nodes":
                nodes_to_viz = None  # Show all
            elif node_set == "Filtered":
                nodes_to_viz = getattr(st.session_state, 'filtered_nodes', None)
            elif node_set == "Traversed":
                nodes_to_viz = st.session_state.traversed_nodes
            elif node_set == "Forked":
                nodes_to_viz = st.session_state.forked_nodes
            elif node_set == "Sample":
                all_ids = list(universe.nodes.keys())
                nodes_to_viz = set(np.random.choice(all_ids, size=min(sample_size, len(all_ids)), replace=False))

            # Apply quick filter if provided
            if quick_filter:
                filter_lower = quick_filter.lower()
                if nodes_to_viz is None:
                    # Filter from all nodes
                    filtered_ids = {
                        node_id for node_id, node in universe.nodes.items()
                        if filter_lower in node.content.get('title', '').lower() or
                           filter_lower in node.content.get('description', '').lower()
                    }
                    nodes_to_viz = filtered_ids
                else:
                    # Filter from existing set
                    nodes_to_viz = {
                        node_id for node_id in nodes_to_viz
                        if filter_lower in universe.nodes[node_id].content.get('title', '').lower() or
                           filter_lower in universe.nodes[node_id].content.get('description', '').lower()
                    }
                if nodes_to_viz:
                    st.info(f"🔍 Showing {len(nodes_to_viz)} nodes matching '{quick_filter}'")

<<<<<<< HEAD
            if nodes_to_viz is not None and len(nodes_to_viz) == 0:
                st.warning("⚠️ No nodes in selected set")
            else:
                # Create two panes: Table view and Graph view
                view_tabs = st.tabs(["📊 Graph View", "📋 Table View"])

                # Determine which nodes are being shown
                if nodes_to_viz is None:
                    display_nodes = list(universe.nodes.values())
                else:
                    display_nodes = [universe.nodes[nid] for nid in nodes_to_viz if nid in universe.nodes]

                # TAB: TABLE VIEW
                with view_tabs[1]:
=======
            # Determine which nodes are being shown
            if nodes_to_viz is None:
                display_nodes = list(universe.nodes.values())
            else:
                display_nodes = [universe.nodes[nid] for nid in nodes_to_viz if nid in universe.nodes]

            if nodes_to_viz is not None and len(nodes_to_viz) == 0:
                st.warning("⚠️ No nodes in selected set")
            else:
                # GRAPH RENDERING - Right at the top
                if viz_type == "2D Projection":
                    with st.spinner("Generating 2D projection..."):
                        fig = create_2d_projection(universe, nodes_to_viz)
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})
                    st.caption("💡 _Hover over nodes to inspect content_")

                elif viz_type == "3D Network":
                    col_layout, col_color = st.columns(2)

                    with col_layout:
                        layout = st.radio(
                            "Layout",
                            ["force-directed", "position"],
                            horizontal=True,
                            help="**position**: Use actual RKHS coordinates.\n\n**force-directed**: Physics simulation."
                        )

                    with col_color:
                        color_by = st.radio(
                            "Color By",
                            ["energy", "domain", "happiness", "importance"],
                            horizontal=True,
                            help="Color nodes by property value"
                        )

                    with st.spinner("Generating 3D network..."):
                        fig = create_3d_network_viz(universe, nodes_to_viz, layout, color_by)
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})
                    st.caption("💡 _Hover over nodes. Drag to rotate, scroll to zoom._")

                elif viz_type == "Kernel Matrix":
                    # Show kernel similarity matrix
                    if nodes_to_viz:
                        node_list = list(nodes_to_viz)[:100]
                    else:
                        node_list = list(universe.nodes.keys())[:100]

                    with st.spinner("Computing kernel matrix..."):
                        n = len(node_list)
                        kernel_matrix = np.zeros((n, n))

                        for i, nid1 in enumerate(node_list):
                            for j, nid2 in enumerate(node_list):
                                node1 = universe.nodes[nid1]
                                node2 = universe.nodes[nid2]
                                kernel_matrix[i, j] = compute_kernel_similarity(
                                    node1.kernel_features,
                                    node2.kernel_features,
                                    universe.kernel_type,
                                    universe.kernel_params.get('gamma', 1.0)
                                )

                        fig = go.Figure(data=go.Heatmap(
                            z=kernel_matrix,
                            x=node_list,
                            y=node_list,
                            colorscale='Viridis'
                        ))

                        fig.update_layout(
                            title="Kernel Similarity Matrix",
                            height=600,
                            xaxis_title="Node ID",
                            yaxis_title="Node ID"
                        )

                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})

                st.divider()

                # Stats below the graph
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Showing Nodes", len(display_nodes))
                with stat_col2:
                    st.metric("Total Edges", len(universe.edges))
                with stat_col3:
                    forked_count = len(st.session_state.forked_nodes) if hasattr(st.session_state, 'forked_nodes') else 0
                    st.metric("Forked Nodes", forked_count)

                # Table view in expander
                with st.expander("📋 View as Table", expanded=False):
>>>>>>> feature/feature/hilbert-ui-enhancements
                    st.subheader(f"Nodes in View ({len(display_nodes)})")

                    # Build dataframe
                    table_data = []
                    for node in display_nodes:
                        table_data.append({
                            'ID': node.id,
                            'Title': node.content.get('title', 'N/A'),
                            'Description': node.content.get('description', '')[:100] + ('...' if len(node.content.get('description', '')) > 100 else ''),
                            'Year': node.content.get('year', 'N/A'),
                            'Parents': len(node.parent_ids),
                            'Children': len(node.children_ids),
                            'LLM Generated': node.content.get('llm_generated', False),
                            'Timestamp': node.timestamp[:10] if len(node.timestamp) > 10 else node.timestamp
                        })

                    df = pd.DataFrame(table_data)
                    st.dataframe(df, width='stretch', height=600)

                    # Export option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download as CSV",
                        csv,
                        file_name=f"{universe.name}_nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

<<<<<<< HEAD
                # TAB: GRAPH VIEW
                with view_tabs[0]:
                    # Universe Description - Show above graph
                    with st.expander("📋 Universe Description", expanded=True):
                        if hasattr(universe, 'description') and universe.description:
                            st.markdown(universe.description)
                        else:
                            # Auto-generate description using Haiku
                            if LLM_AVAILABLE and st.button("🤖 Generate Description with AI"):
                                with st.spinner("Generating description..."):
                                    try:
                                        import random
                                        sample_nodes = random.sample(list(universe.nodes.values()), min(100, len(universe.nodes)))
                                        node_info = "\n".join([
                                            f"- {node.content.get('title', node.id)}: {node.content.get('description', '')[:100]}"
                                            for node in sample_nodes[:20]
                                        ])

                                        response = litellm.completion(
                                            model="claude-haiku-4-5-20251001",
                                            messages=[{
                                                "role": "user",
                                                "content": f"Analyze this multiverse and write a 2-3 sentence description:\n\n"
                                                           f"Universe: {universe.name}\n"
                                                           f"Nodes: {len(universe.nodes)}\n"
                                                           f"Sample content:\n{node_info}"
                                            }],
                                            max_tokens=150
                                        )
                                        description = response.choices[0].message.content
                                        st.markdown(description)
                                        if 'generated_descriptions' not in st.session_state:
                                            st.session_state.generated_descriptions = {}
                                        st.session_state.generated_descriptions[universe.name] = description
                                    except Exception as e:
                                        st.error(f"Could not generate description: {e}")
                            else:
                                if hasattr(st.session_state, 'generated_descriptions') and universe.name in st.session_state.generated_descriptions:
                                    st.markdown(st.session_state.generated_descriptions[universe.name])
                                else:
                                    st.info(f"**{universe.name}** - {len(universe.nodes)} nodes, {len(universe.edges)} edges")

                    st.divider()

                    # Render graph
                    if viz_type == "3D Network":
                        col_layout, col_color = st.columns(2)

                        with col_layout:
                            layout = st.radio(
                                "Layout",
                                ["force-directed", "position"],
                                horizontal=True,
                                help="**position**: Use the actual RKHS coordinates of each node. Shows true geometric relationships in the kernel space.\n\n"
                                     "**force-directed**: Apply physics simulation to spread nodes apart. Better for seeing network structure when nodes are clustered."
                            )

                        with col_color:
                            color_by = st.radio(
                                "Color By",
                                ["energy", "domain", "happiness", "importance"],
                                horizontal=True,
                                help="**energy**: Default property value.\n\n"
                                     "**domain**: Life domain (education, career, relationships, etc.). Great for life timelines!\n\n"
                                     "**happiness**: Emotional state/satisfaction level.\n\n"
                                     "**importance**: Significance of the event/node."
                            )

                        with st.spinner("Generating graph..."):
                            fig = create_3d_network_viz(universe, nodes_to_viz, layout, color_by)
                        st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})
                        st.caption("💡 _Hover over nodes to inspect content. Drag to rotate, scroll to zoom._")

                    elif viz_type == "2D Projection":
                        with st.spinner("Generating graph..."):
                            fig = create_2d_projection(universe, nodes_to_viz)
                        st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})
                        st.caption("💡 _Hover over nodes to inspect content_")

                    elif viz_type == "Kernel Matrix":
                        # Show kernel similarity matrix
                        if nodes_to_viz:
                            node_list = list(nodes_to_viz)[:100]  # Limit for performance
                        else:
                            node_list = list(universe.nodes.keys())[:100]

                        with st.spinner("Computing kernel matrix..."):
                            n = len(node_list)
                            kernel_matrix = np.zeros((n, n))

                            for i, nid1 in enumerate(node_list):
                                for j, nid2 in enumerate(node_list):
                                    node1 = universe.nodes[nid1]
                                    node2 = universe.nodes[nid2]
                                    kernel_matrix[i, j] = compute_kernel_similarity(
                                        node1.kernel_features,
                                        node2.kernel_features,
                                        universe.kernel_type,
                                        universe.kernel_params.get('gamma', 1.0)
                                    )

                            fig = go.Figure(data=go.Heatmap(
                                z=kernel_matrix,
                                x=node_list,
                                y=node_list,
                                colorscale='Viridis'
                            ))

                            fig.update_layout(
                                title="Kernel Similarity Matrix",
                                height=600,
                                xaxis_title="Node ID",
                                yaxis_title="Node ID"
                            )

                        st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})

                # Morph button - prepares currently visualized nodes for Text/LLM tab
                col_morph1, col_morph2 = st.columns([2, 1])
                with col_morph2:
                    if st.button("🔮 Morph These Nodes", use_container_width=True, help="Pre-select these nodes in Text/LLM tab for transformation"):
                        morph_set = nodes_to_viz if nodes_to_viz else set(universe.nodes.keys())
                        st.session_state.morph_nodes = morph_set
                        st.info(f"✨ {len(morph_set)} nodes ready! Go to **📝 Text/LLM** tab to transform them.")

                # Summary section below the graph
                st.divider()

                # Determine which nodes are being shown
                if nodes_to_viz is None:
                    summary_nodes = list(universe.nodes.values())
                else:
                    summary_nodes = [universe.nodes[nid] for nid in nodes_to_viz if nid in universe.nodes]

                # Add summary threshold configuration
                col_summary1, col_summary2 = st.columns([2, 1])
                with col_summary1:
                    st.markdown("### 📋 Graph Content Summary")
                with col_summary2:
                    summary_threshold = st.number_input(
                        "Sample threshold",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        help="If the graph has more nodes than this threshold, only a random sample will be included in the summary."
                    )

                # Display summary
                if len(summary_nodes) > summary_threshold:
                    # Sample for efficiency
                    import random
                    sampled_nodes = random.sample(summary_nodes, summary_threshold)
                    st.info(f"ℹ️ Showing summary of {len(summary_nodes)} nodes (sampled {summary_threshold} for efficiency)")
                    nodes_for_summary = sampled_nodes
                else:
                    nodes_for_summary = summary_nodes

                # Collect titles and descriptions
=======

                # Collect titles and descriptions from displayed nodes
>>>>>>> feature/feature/hilbert-ui-enhancements
                titles = []
                descriptions = []
                properties_summary = {}

<<<<<<< HEAD
                for node in nodes_for_summary:
=======
                for node in display_nodes[:100]:  # Limit to first 100 for summary
>>>>>>> feature/feature/hilbert-ui-enhancements
                    title = node.content.get('title', '')
                    if title:
                        titles.append(title)

                    desc = node.content.get('description', '')
                    if desc:
                        descriptions.append(desc)

                    # Aggregate properties
                    if 'properties' in node.content:
                        for k, v in node.content['properties'].items():
                            if isinstance(v, (int, float)):
                                if k not in properties_summary:
                                    properties_summary[k] = []
                                properties_summary[k].append(v)

<<<<<<< HEAD
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Nodes", len(summary_nodes))
                with col2:
                    edges_count = len([e for e in universe.edges
                                      if (nodes_to_viz is None or (e.source_id in nodes_to_viz and e.target_id in nodes_to_viz))])
                    st.metric("Edges", edges_count)
                with col3:
                    nodes_with_parents = sum(1 for n in nodes_for_summary if n.parent_ids)
                    st.metric("Nodes with Parents", nodes_with_parents)
                with col4:
                    nodes_with_children = sum(1 for n in nodes_for_summary if n.children_ids)
                    st.metric("Nodes with Children", nodes_with_children)
=======
>>>>>>> feature/feature/hilbert-ui-enhancements

                # Property statistics
                if properties_summary:
                    st.markdown("**Property Statistics:**")
                    prop_cols = st.columns(len(properties_summary))
                    for idx, (prop_name, values) in enumerate(properties_summary.items()):
                        with prop_cols[idx]:
                            st.metric(
                                f"{prop_name.capitalize()} (avg)",
                                f"{np.mean(values):.3f}",
                                delta=f"±{np.std(values):.3f}"
                            )

                # Show sample of items (titles and descriptions)
                if titles or descriptions:
                    with st.expander(f"📚 Sample Items ({min(10, len(titles))} of {len(titles)})", expanded=False):
                        for i, title in enumerate(titles[:10]):
                            st.markdown(f"**{i+1}. {title}**")
                            if i < len(descriptions) and descriptions[i]:
                                # Show first 100 chars of description
                                desc_preview = descriptions[i][:100]
                                if len(descriptions[i]) > 100:
                                    desc_preview += "..."
                                st.caption(desc_preview)

                st.divider()

                # Explanation of energy metric
                with st.expander("ℹ️ What does 'Energy' mean?", expanded=False):
                    st.markdown("""
                    **Energy** is a scalar property assigned to each node in the RKHS multiverse.

                    - **Visual Representation**: Nodes are colored by their energy value in the visualization
                    - **Scale**: Typically ranges from 0.0 (low energy, dark colors) to 1.0 (high energy, bright colors)
                    - **Interpretation**: Energy can represent various semantic properties depending on your universe:
                        - **Semantic intensity**: How "strongly" a concept is expressed
                        - **Activation level**: How active or prominent a state is
                        - **Distance from origin**: Magnitude in the RKHS (||x||)
                        - **Custom metrics**: Any domain-specific property you assign

                    In the **Viridis** color scale used here:
                    - **Purple/Dark**: Low energy values (≈0.0-0.3)
                    - **Green/Teal**: Medium energy values (≈0.3-0.7)
                    - **Yellow/Bright**: High energy values (≈0.7-1.0)

                    *Note: If a node doesn't have an energy property, it defaults to 0.5 (neutral).*
                    """)

                st.divider()

                # Export visualization data
                if st.button("💾 Export Visualization Data"):
                    export_data = {
                        'universe_name': universe.name,
                        'node_set': node_set,
                        'nodes': list(nodes_to_viz) if nodes_to_viz else list(universe.nodes.keys())
                    }
                    st.download_button(
                        "Download JSON",
                        json.dumps(export_data, indent=2),
                        file_name=f"viz_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
    
    # ========================================================================
    # TAB 3: BROWSE
    # ========================================================================
    with tabs[3]:
        st.header("Browse Universe")

        if st.session_state.universe is None:
            st.warning("⚠️ No universe loaded. Please open or create a universe first.")
        else:
            universe = st.session_state.universe

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nodes", len(universe.nodes))
            col2.metric("Edges", len(universe.edges))
            col3.metric("Dimension", universe.dimension)
            col4.metric("Kernel", universe.kernel_type)

            st.divider()

            # Node browser
            search = st.text_input(
                "🔍 Search nodes",
                "",
                help="Search for nodes by title or ID. Enter any text to filter the node list. "
                     "Search is case-insensitive and matches partial strings."
            )

            nodes_list = []
            for nid, node in universe.nodes.items():
                title = node.content.get('title', nid)
                if search.lower() in title.lower() or search.lower() in nid.lower():
                    nodes_list.append({
                        'ID': nid,
                        'Title': title,
                        'Parents': len(node.parent_ids),
                        'Children': len(node.children_ids),
                        'Timestamp': node.timestamp
                    })

            if nodes_list:
                df = pd.DataFrame(nodes_list)
                st.dataframe(df, width='stretch', height=400)

                # Node detail view
                selected_id = st.selectbox(
                    "Select node for details",
                    df['ID'].tolist(),
                    help="Choose a node from the filtered list to view its complete details, "
                         "including content, metadata, position, and relationships."
                )

                if selected_id:
                    node = universe.nodes[selected_id]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Content")
                        st.json(node.content)

                    with col2:
                        st.subheader("Metadata")
                        st.write(f"**Position:** {node.position[:3]}")
                        st.write(f"**Parents:** {len(node.parent_ids)}")
                        st.write(f"**Children:** {len(node.children_ids)}")
                        st.json(node.metadata)

                    # Mark as traversed
                    if st.button(f"📍 Mark as Traversed"):
                        st.session_state.traversed_nodes.add(selected_id)
                        st.success(f"Added {selected_id} to traversed nodes")
    
    # ========================================================================
    # TAB 4: FORK
    # ========================================================================
    with tabs[4]:
        st.header("Fork & Branch")

<<<<<<< HEAD
        if st.session_state.universe is None:
            st.warning("⚠️ No universe loaded.")
=======
        # Debug info
        if st.session_state.universe is None:
            st.warning("⚠️ No universe loaded.")
            st.info("🔍 Debug: session_state.universe is None. Please load a universe from the 📂 Load tab.")
            if hasattr(st.session_state, 'keys'):
                st.write("Available session state keys:", list(st.session_state.keys()))
>>>>>>> feature/feature/hilbert-ui-enhancements
        else:
            universe = st.session_state.universe

            st.subheader("Create Fork from Node")

            node_ids = list(universe.nodes.keys())
            source_node = st.selectbox(
                "Source Node",
                node_ids,
                help="Select the parent node from which to create fork(s). "
                     "The new node(s) will be created near this node in RKHS space."
            )

            if source_node:
                col1, col2 = st.columns(2)

                with col1:
                    fork_title = st.text_input(
                        "Fork Title",
                        value=f"Fork of {source_node}",
                        help="Title for the forked node(s). If creating multiple children, "
                             "each will be numbered (e.g., 'Fork of node_001 0', 'Fork of node_001 1', etc.)"
                    )
                    fork_desc = st.text_area(
                        "Description",
                        value="Branching timeline",
                        help="Description for the forked node(s). This helps document why and how the fork was created."
                    )

                with col2:
                    perturbation = st.slider(
                        "Position Perturbation",
                        0.0, 1.0, 0.1,
                        help="How far to move the forked node from the source in RKHS space. "
                             "0.0 = exact copy at same position, 0.1 = small variation (recommended), "
                             "0.5 = moderate variation, 1.0 = large variation. Higher values create more divergent forks."
                    )
                    n_children = st.number_input(
                        "Children to Create",
                        min_value=1,
                        max_value=100,
                        value=1,
                        help="Number of child nodes to fork from the source. "
                             "1 = single fork, 5+ = create a cluster of variations. "
                             "Each child will have slightly different random perturbations."
                    )

                if st.button("🔱 Create Fork"):
                    # Require authentication for fork operations
                    if not require_auth("create forks"):
                        st.stop()

                    # Check tier limits
                    user_tier = "free"
                    if AUTH_AVAILABLE and is_authenticated():
                        user_info = get_user_info()
                        user_tier = user_info.get("subscription_tier", "free")

                    current_node_count = len(universe.nodes)
                    new_total = current_node_count + n_children

                    if not check_tier_limit("nodes", new_total, user_tier):
                        st.stop()

                    source = universe.nodes[source_node]

                    for i in range(n_children):
                        new_id = f"fork_{source_node}_{i}_{datetime.now().timestamp()}"

                        # Perturb position
                        new_position = [
                            p + np.random.randn() * perturbation
                            for p in source.position
                        ]

                        new_features = [
                            f + np.random.randn() * perturbation
                            for f in source.kernel_features
                        ]

                        new_node = RKHSNode(
                            id=new_id,
                            position=new_position,
                            content={
                                "title": f"{fork_title} {i}",
                                "description": fork_desc,
                                "forked_from": source_node
                            },
                            metadata={"fork_time": datetime.now().isoformat()},
                            kernel_features=new_features,
                            timestamp=datetime.now().isoformat(),
                            parent_ids=[source_node],
                            children_ids=[]
                        )

                        universe.nodes[new_id] = new_node
                        source.children_ids.append(new_id)

                        # Create edge
                        sim = compute_kernel_similarity(
                            source.kernel_features,
                            new_features,
                            universe.kernel_type,
                            universe.kernel_params.get('gamma', 1.0)
                        )

                        edge = RKHSEdge(
                            source_id=source_node,
                            target_id=new_id,
                            weight=1.0,
                            kernel_similarity=sim,
                            transition_type="fork",
                            metadata={}
                        )
                        universe.edges.append(edge)

                        st.session_state.forked_nodes.add(new_id)

                    # Update session state explicitly to ensure changes persist
                    st.session_state.universe = universe
                    st.success(f"✅ Created {n_children} fork(s) from {source_node}")
<<<<<<< HEAD
                    st.info(f"Universe now has {len(universe.nodes)} total nodes. Go to 🔍 Explore tab to see them!")
=======
                    st.info(f"Universe now has {len(universe.nodes)} total nodes. Go to 🔍 Explore tab and select **'All Nodes'** or **'Forked'** to see them!")
>>>>>>> feature/feature/hilbert-ui-enhancements

                    # Auto-save user's multiverse if authenticated
                    save_user_multiverse(universe)
                    st.rerun()  # Force refresh to show updated metrics

            # LLM-Powered Path Generation
            st.divider()
            st.subheader("🤖 AI-Powered Fork Path Generation")
            st.markdown("Generate coherent alternate timeline paths using LLMs")

            if LLM_AVAILABLE:
                path_col1, path_col2 = st.columns(2)

                with path_col1:
                    fork_start_node = st.selectbox(
                        "Fork Starting Point",
                        node_ids,
                        key="fork_start",
                        help="The node where the alternate timeline diverges"
                    )

                with path_col2:
                    fork_end_node = st.selectbox(
                        "Target End State",
                        [nid for nid in node_ids if nid != fork_start_node],
                        key="fork_end",
                        help="The destination node to guide the alternate path generation"
                    )

                # Calculate path length from source to target
                if fork_start_node and fork_end_node:
                    # Find shortest path (simple BFS)
                    from collections import deque

                    def find_path_length(start, end, nodes, edges):
                        # Build adjacency list
                        adj = {}
                        for edge in edges:
                            if edge.source_id not in adj:
                                adj[edge.source_id] = []
                            adj[edge.source_id].append(edge.target_id)

                        queue = deque([(start, 0)])
                        visited = {start}

                        while queue:
                            node, dist = queue.popleft()
                            if node == end:
                                return dist
                            if node in adj:
                                for neighbor in adj[node]:
                                    if neighbor not in visited:
                                        visited.add(neighbor)
                                        queue.append((neighbor, dist + 1))
                        return -1  # No path found

                    default_steps = find_path_length(fork_start_node, fork_end_node, universe.nodes, universe.edges)
                    if default_steps == -1:
                        default_steps = 3

                    path_params_col1, path_params_col2 = st.columns(2)

                    with path_params_col1:
                        num_steps = st.number_input(
                            "Number of Steps",
                            min_value=1,
                            max_value=20,
                            value=max(1, default_steps) if default_steps > 0 else 3,
                            help=f"Number of intermediate nodes to generate. Original path has {default_steps} steps."
                        )

                    with path_params_col2:
                        num_fork_paths = st.number_input(
                            "Number of Alternate Paths",
                            min_value=1,
                            max_value=10,
                            value=1,
                            help="How many different alternate timeline paths to generate"
                        )

                    llm_model = st.selectbox(
                        "LLM Model",
                        ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", "gpt-4o-mini"],
                        help="Model to use for generating path content. Haiku 4.5 is fastest/cheapest, Sonnet 4.5 is highest quality."
                    )

                    if st.button("🚀 Generate Alternate Path(s)"):
                        if not require_auth("generate AI paths"):
                            st.stop()

                        with st.spinner(f"Generating {num_fork_paths} alternate path(s) with {num_steps} steps each..."):
                            try:
                                start_node = universe.nodes[fork_start_node]
                                end_node = universe.nodes[fork_end_node]

                                for path_idx in range(num_fork_paths):
                                    # Generate path using LLM
                                    # Detect universe type and adapt prompt accordingly
                                    universe_type = universe.metadata.get('type', 'unknown')
                                    subject_name = universe.metadata.get('subject', None)

                                    # Build context-aware prompt
                                    if universe_type == 'life_timeline' and subject_name:
                                        # Life timeline mode
                                        context_intro = f"""You are generating an alternate timeline for {subject_name}'s life.

IMPORTANT: {subject_name} is the same person throughout. Show what different events could have happened to {subject_name}."""
                                        event_type = "life events"
                                        title_instruction = f"brief event title (5-8 words, what {subject_name} does/experiences)"
                                        desc_instruction = f"event description (1-2 sentences about what happens to {subject_name})"
                                    else:
                                        # General multiverse mode (books, concepts, stories, etc.)
                                        context_intro = f"""You are generating an alternate path through a multiverse of related concepts/states.

Universe: {universe.name}
Type: {universe_type if universe_type != 'unknown' else 'conceptual space'}

Create a coherent progression that maintains thematic/semantic continuity."""
                                        event_type = "intermediate states/concepts"
                                        title_instruction = "brief title (5-8 words)"
                                        desc_instruction = "description (1-2 sentences)"

                                    prompt = f"""{context_intro}

Starting point: {start_node.content.get('title', fork_start_node)}
{start_node.content.get('description', '')}

Target point: {end_node.content.get('title', fork_end_node)}
{end_node.content.get('description', '')}

Generate {num_steps} {event_type} that create a coherent alternate path from starting point to target. Each should:
- Flow naturally from the previous one
- Be plausible/coherent given the start and end points
- Maintain thematic consistency with the universe type
- Progress logically (chronologically for timelines, thematically for concept spaces)

Return ONLY a JSON array of {num_steps} objects, each with:
- "title": {title_instruction}
- "description": {desc_instruction}
- "year": year/date (if applicable to this universe type)

Example format:
[{{"title": "...", "description": "...", "year": 1985}}, ...]"""

                                    response = litellm.completion(
                                        model=llm_model,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=2000,
                                        temperature=0.7 + (path_idx * 0.1)  # Vary temperature for diversity
                                    )

                                    # Parse response
                                    import json
                                    import re
                                    content = response.choices[0].message.content

                                    # Extract JSON array
                                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                                    if json_match:
                                        events = json.loads(json_match.group())
                                    else:
                                        st.error(f"Could not parse LLM response for path {path_idx + 1}")
                                        continue

                                    # Create nodes for each event
                                    prev_node_id = fork_start_node
                                    prev_node = start_node

                                    for step_idx, event in enumerate(events):
                                        new_id = f"llm_fork_p{path_idx}_s{step_idx}_{datetime.now().timestamp()}"

                                        # Interpolate position from start to end
                                        t = (step_idx + 1) / (num_steps + 1)
                                        new_position = [
                                            start_node.position[i] * (1 - t) + end_node.position[i] * t + np.random.randn() * 0.05
                                            for i in range(len(start_node.position))
                                        ]

                                        new_features = [
                                            start_node.kernel_features[i] * (1 - t) + end_node.kernel_features[i] * t + np.random.randn() * 0.05
                                            for i in range(len(start_node.kernel_features))
                                        ]

                                        new_node = RKHSNode(
                                            id=new_id,
                                            position=new_position,
                                            content={
                                                "title": event.get("title", f"Event {step_idx + 1}"),
                                                "description": event.get("description", ""),
                                                "year": event.get("year"),
                                                "path_index": path_idx,
                                                "step_index": step_idx,
                                                "llm_generated": True,
                                                "fork_from": fork_start_node,
                                                "target_end": fork_end_node
                                            },
                                            metadata={
                                                "llm_model": llm_model,
                                                "generation_time": datetime.now().isoformat()
                                            },
                                            kernel_features=new_features,
                                            timestamp=datetime.now().isoformat(),
                                            parent_ids=[prev_node_id],
                                            children_ids=[]
                                        )

                                        universe.nodes[new_id] = new_node
                                        prev_node.children_ids.append(new_id)

                                        # Create edge
                                        sim = compute_kernel_similarity(
                                            prev_node.kernel_features,
                                            new_features,
                                            universe.kernel_type,
                                            universe.kernel_params.get('gamma', 1.0)
                                        )

                                        edge = RKHSEdge(
                                            source_id=prev_node_id,
                                            target_id=new_id,
                                            weight=1.0,
                                            kernel_similarity=sim,
                                            transition_type="llm_fork",
                                            metadata={"llm_generated": True}
                                        )
                                        universe.edges.append(edge)
                                        st.session_state.forked_nodes.add(new_id)

                                        prev_node_id = new_id
                                        prev_node = new_node

                                    # Connect last node to end node
                                    prev_node.children_ids.append(fork_end_node)
                                    end_node.parent_ids.append(prev_node_id)

                                    edge = RKHSEdge(
                                        source_id=prev_node_id,
                                        target_id=fork_end_node,
                                        weight=1.0,
                                        kernel_similarity=compute_kernel_similarity(
                                            prev_node.kernel_features,
                                            end_node.kernel_features,
                                            universe.kernel_type,
                                            universe.kernel_params.get('gamma', 1.0)
                                        ),
                                        transition_type="llm_fork",
                                        metadata={"llm_generated": True}
                                    )
                                    universe.edges.append(edge)

                                # Update session state explicitly to ensure changes persist
                                st.session_state.universe = universe
                                st.success(f"✅ Generated {num_fork_paths} alternate path(s) with {num_steps} steps each!")
<<<<<<< HEAD
                                st.info(f"Universe now has {len(universe.nodes)} total nodes. Go to 🔍 Explore tab and select 'All Nodes' to see them!")
=======
                                st.info(f"Universe now has {len(universe.nodes)} total nodes. Go to 🔍 Explore tab and select **'All Nodes'** or **'Forked'** to see them!")
>>>>>>> feature/feature/hilbert-ui-enhancements
                                save_user_multiverse(universe)
                                st.rerun()  # Force refresh to show updated metrics

                            except Exception as e:
                                st.error(f"Error generating paths: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            else:
                st.warning("LLM features not available. Install litellm and configure API keys.")

            # Show forked nodes
            st.divider()
            if st.session_state.forked_nodes:
                st.subheader("Forked Nodes")
                st.write(f"Total forked: {len(st.session_state.forked_nodes)}")
                st.write(list(st.session_state.forked_nodes)[:20])
<<<<<<< HEAD
    
=======

            # Save & Publish at bottom of Fork tab
            if st.session_state.universe is not None:
                render_save_and_publish(st.session_state.universe)

>>>>>>> feature/feature/hilbert-ui-enhancements
    # ========================================================================
    # TAB 5: FILTER
    # ========================================================================
    with tabs[5]:
        st.header("Filter Nodes")

        if st.session_state.universe is None:
            st.warning("⚠️ No universe loaded.")
        else:
            universe = st.session_state.universe

            # Show auto-filter status
            if st.session_state.auto_filter_active:
                col_info, col_clear = st.columns([3, 1])
                with col_info:
                    st.info(
                        f"ℹ️ **Auto-filter active**: Showing {len(st.session_state.filtered_nodes)} of "
                        f"{len(universe.nodes):,} nodes for performance."
                    )
                with col_clear:
                    if st.button("Clear Auto-filter"):
                        st.session_state.filtered_nodes = None
                        st.session_state.auto_filter_active = False
                        st.rerun()
                st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                filter_type = st.selectbox(
                    "Filter Type",
                    ["All Nodes", "Traversed Only", "Forked Only", "By Property", "By Kernel Distance"],
                    help="**All Nodes**: Show every node (no filtering).\n\n"
                         "**Traversed Only**: Show nodes you marked as traversed in the Browse tab.\n\n"
                         "**Forked Only**: Show nodes you created using the Fork tab.\n\n"
                         "**By Property**: Filter by a numeric property value (e.g., energy between 0.3-0.7).\n\n"
                         "**By Kernel Distance**: Show nodes within a certain distance from a reference node in RKHS space."
                )

            with col2:
                if filter_type == "By Property":
                    prop_key = st.text_input(
                        "Property Key",
                        "energy",
                        help="Name of the property to filter by (e.g., 'energy', 'entropy'). "
                             "Must be a numeric property that exists in node.content.properties."
                    )
                    prop_min = st.number_input(
                        "Min Value",
                        value=0.0,
                        help="Minimum value for the property (inclusive). Nodes with property values below this will be excluded."
                    )
                    prop_max = st.number_input(
                        "Max Value",
                        value=1.0,
                        help="Maximum value for the property (inclusive). Nodes with property values above this will be excluded."
                    )

                elif filter_type == "By Kernel Distance":
                    ref_node = st.selectbox(
                        "Reference Node",
                        list(universe.nodes.keys()),
                        help="Choose a node to use as the center point. The filter will show nodes within "
                             "the specified distance from this reference node in RKHS space."
                    )
                    max_distance = st.slider(
                        "Max Distance",
                        0.0, 5.0, 1.0,
                        help="Maximum Euclidean distance from the reference node in RKHS space. "
                             "Smaller values (0.5-1.0) = nearby nodes only. Larger values (2.0-5.0) = broader neighborhood. "
                             "Distance is calculated as ||node1.features - node2.features||."
                    )

            with col3:
                # Adjust min_value for small universes
                min_limit = min(10, len(universe.nodes))
                limit = st.number_input(
                    "Max Results",
                    min_value=min_limit,
                    max_value=len(universe.nodes),
                    value=min(100, len(universe.nodes)),
                    help="Maximum number of nodes to include in the filter results. "
                         "Helps keep visualizations responsive by limiting the node count. "
                         "First N matching nodes will be included."
                )

            if st.button("🎯 Apply Filter"):
                filtered = set()

                if filter_type == "All Nodes":
                    filtered = set(universe.nodes.keys())

                elif filter_type == "Traversed Only":
                    filtered = st.session_state.traversed_nodes

                elif filter_type == "Forked Only":
                    filtered = st.session_state.forked_nodes

                elif filter_type == "By Property":
                    for nid, node in universe.nodes.items():
                        if 'properties' in node.content:
                            val = node.content['properties'].get(prop_key)
                            if val is not None and prop_min <= val <= prop_max:
                                filtered.add(nid)

                elif filter_type == "By Kernel Distance":
                    ref = universe.nodes[ref_node]
                    for nid, node in universe.nodes.items():
                        dist = np.linalg.norm(
                            np.array(ref.kernel_features) - np.array(node.kernel_features)
                        )
                        if dist <= max_distance:
                            filtered.add(nid)

                filtered = list(filtered)[:limit]
                st.session_state.filtered_nodes = set(filtered)
                st.session_state.auto_filter_active = False  # User applied manual filter

                st.success(f"✅ Filtered to {len(filtered)} nodes")
                st.write(filtered[:50])
    
    # ========================================================================
    # TAB 6: TEXT/LLM OPERATIONS
    # ========================================================================
    with tabs[6]:
        st.header("Text Extraction & LLM Operations")

        if st.session_state.universe is None:
            st.warning("⚠️ No universe loaded.")
        else:
            universe = st.session_state.universe

            st.markdown("""
            Extract text from selected nodes or apply LLM transformations to modify content.
            This is perfect for:
            - Extracting descriptions from life events
            - Rewriting event descriptions in different styles
            - Summarizing or expanding content
            - Translating or rephrasing
            """)

            st.divider()

            # Select node set
            col1, col2 = st.columns(2)

            with col1:
                # Check if morph_nodes is available
                options = ["All Nodes", "Filtered", "Traversed", "Forked"]
                default_idx = 0
                if st.session_state.morph_nodes:
                    options.insert(0, "Morphed Nodes (from Viz)")
                    default_idx = 0
                    st.info("✨ Using nodes from visualization! Click 'Morphed Nodes' above.")

                text_node_set = st.selectbox(
                    "Select Nodes",
                    options,
                    index=default_idx,
                    help="Choose which nodes to extract text from or transform. 'Morphed Nodes' are pre-selected from the visualization."
                )

            # Determine nodes
            if text_node_set == "Morphed Nodes (from Viz)":
                selected_nodes = st.session_state.morph_nodes or set()
            elif text_node_set == "All Nodes":
                selected_nodes = set(universe.nodes.keys())
            elif text_node_set == "Filtered":
                selected_nodes = getattr(st.session_state, 'filtered_nodes', set())
            elif text_node_set == "Traversed":
                selected_nodes = st.session_state.traversed_nodes
            elif text_node_set == "Forked":
                selected_nodes = st.session_state.forked_nodes
            else:
                selected_nodes = set()

            with col2:
                st.metric("Selected Nodes", len(selected_nodes))

            st.divider()

            # Two operation modes
            operation = st.radio(
                "Operation",
                ["Extract Text", "LLM Morph"],
                horizontal=True,
                help="**Extract Text**: Copy all text content from selected nodes.\n\n"
                     "**LLM Morph**: Use AI to transform text in selected nodes."
            )

            if operation == "Extract Text":
                st.subheader("📝 Extract Text")

                if st.button("Extract Text from Selected Nodes"):
                    if not selected_nodes:
                        st.warning("No nodes selected")
                    else:
                        extracted_text = extract_text_from_nodes(universe.nodes, selected_nodes)

                        st.text_area(
                            f"Extracted Text ({len(selected_nodes)} nodes)",
                            extracted_text,
                            height=400
                        )

                        # Download button
                        st.download_button(
                            "📥 Download as Text File",
                            extracted_text,
                            file_name=f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )

                        # Clear text option
                        st.divider()
                        st.subheader("🗑️ Clear Text")
                        st.warning("⚠️ This will delete all titles and descriptions from selected nodes!")

                        field_to_clear = st.radio(
                            "Clear which field?",
                            ["title", "description", "both"],
                            help="Choose which text fields to clear from the selected nodes"
                        )

                        if st.button("Clear Text from Selected Nodes", type="secondary"):
                            cleared_count = 0
                            for node_id in selected_nodes:
                                if node_id in universe.nodes:
                                    node = universe.nodes[node_id]
                                    if field_to_clear in ["title", "both"]:
                                        node.content['title'] = ""
                                    if field_to_clear in ["description", "both"]:
                                        node.content['description'] = ""
                                    cleared_count += 1

                            st.success(f"✅ Cleared {field_to_clear} from {cleared_count} nodes")

            else:  # LLM Morph
                st.subheader("🤖 LLM Morph")

                if not LLM_AVAILABLE:
                    st.error("❌ LLM features not available. Install nimble-llm-caller:")
                    st.code("pip install nimble-llm-caller")
                else:
                    col1, col2 = st.columns(2)

                    with col1:
                        field = st.selectbox(
                            "Field to Transform",
                            ["description", "title"],
                            help="Which text field should the LLM transform?"
                        )

                    with col2:
                        model = st.selectbox(
                            "Model",
                            ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022"],
                            help="**Haiku**: Fast and cheap, good for simple transformations.\n\n"
                                 "**Sonnet**: More capable, better for complex rewriting."
                        )

                    prompt = st.text_area(
                        "Transformation Prompt",
                        "Rewrite this in a more dramatic, cinematic style:",
                        height=100,
                        help="Describe how you want the LLM to transform the text. "
                             "This prompt will be applied to each node's content."
                    )

                    batch_size = st.slider(
                        "Batch Size",
                        1, 20, 5,
                        help="Number of nodes to process in each batch. Larger = faster but more expensive."
                    )

                    st.divider()

                    # Cost estimate
                    estimated_cost = len(selected_nodes) * 0.001  # Rough estimate
                    st.caption(f"Estimated cost: ~${estimated_cost:.3f} (approximate)")

                    if st.button("🚀 Apply LLM Transformation", type="primary"):
                        if not selected_nodes:
                            st.warning("No nodes selected")
                        else:
                            with st.spinner(f"Transforming {len(selected_nodes)} nodes..."):
                                modified_count = apply_llm_morph_to_nodes(
                                    universe,
                                    selected_nodes,
                                    prompt,
                                    field,
                                    model,
                                    batch_size
                                )

                            st.success(f"✅ Transformed {modified_count} nodes")

                            # Show sample
                            if modified_count > 0:
                                st.subheader("Sample Results")
                                sample_node_id = list(selected_nodes)[0]
                                if sample_node_id in universe.nodes:
                                    sample_node = universe.nodes[sample_node_id]
                                    st.markdown(f"**{sample_node.content.get('title', sample_node_id)}**")
                                    st.write(sample_node.content.get('description', ''))
<<<<<<< HEAD
    
=======

            # Save & Publish at bottom of Text/LLM tab
            if st.session_state.universe is not None:
                render_save_and_publish(st.session_state.universe)

>>>>>>> feature/feature/hilbert-ui-enhancements
    # ========================================================================
    # TAB 7: LIFE TIMELINE
    # ========================================================================
    with tabs[7]:
        st.header("📅 Life Timeline Editor")

        if st.session_state.universe is None:
            st.warning("⚠️ No universe loaded.")
        else:
            universe = st.session_state.universe

            st.markdown("""
            Create and edit life events for personal timelines. Perfect for exploring alternative life paths!
            """)

            st.divider()

            # Event Creator
            st.subheader("✨ Create New Life Event")

            with st.form("create_event"):
                col1, col2 = st.columns(2)

                with col1:
                    event_id = st.text_input(
                        "Event ID",
                        placeholder="college_2020",
                        help="Unique identifier (e.g., college_2020, wedding_2025)"
                    )

                    title = st.text_input(
                        "Event Title",
                        placeholder="Graduated from University",
                        help="Short title for the event"
                    )

                    description = st.text_area(
                        "Description",
                        placeholder="Completed BS in Computer Science with honors...",
                        help="Detailed description of what happened"
                    )

                    domain = st.selectbox(
                        "Life Domain",
                        get_life_domains(),
                        help="Category of life event"
                    )

                with col2:
                    year = st.number_input(
                        "Year",
                        min_value=1900,
                        max_value=2100,
                        value=2020,
                        help="Year the event occurred"
                    )

                    age = st.number_input(
                        "Age",
                        min_value=0,
                        max_value=120,
                        value=22,
                        help="Your age when this happened"
                    )

                    importance = st.slider(
                        "Importance",
                        0.0, 1.0, 0.8, 0.1,
                        help="How significant was this event? 0=minor, 1=life-defining"
                    )

                    happiness = st.slider(
                        "Happiness",
                        0.0, 1.0, 0.7, 0.1,
                        help="How happy/satisfied were you? 0=unhappy, 1=very happy"
                    )

                    # Parent selection
                    parent_options = ["None"] + list(universe.nodes.keys())
                    parent = st.selectbox(
                        "Parent Event",
                        parent_options,
                        help="Which event led to this one?"
                    )

                submitted = st.form_submit_button("Create Event")

                if submitted:
                    if not event_id or not title:
                        st.error("Event ID and Title are required")
                    elif event_id in universe.nodes:
                        st.error(f"Event ID '{event_id}' already exists")
                    else:
                        # Create node
                        parent_ids = [] if parent == "None" else [parent]
                        new_node = create_life_event_node(
                            event_id,
                            title,
                            description,
                            year,
                            age,
                            importance,
                            happiness,
                            domain,
                            universe.dimension,
                            parent_ids
                        )

                        # Add to universe
                        universe.nodes[event_id] = new_node

                        # Update parent's children
                        if parent_ids:
                            parent_node = universe.nodes[parent_ids[0]]
                            parent_node.children_ids.append(event_id)

                        st.success(f"✅ Created event: {title}")
                        st.rerun()

            st.divider()

            # Event Editor
            st.subheader("✏️ Edit Existing Event")

            if universe.nodes:
                edit_node_id = st.selectbox(
                    "Select Event to Edit",
                    list(universe.nodes.keys()),
                    format_func=lambda x: f"{x}: {universe.nodes[x].content.get('title', 'Untitled')}"
                )

                if edit_node_id:
                    edit_node = universe.nodes[edit_node_id]

                    with st.form("edit_event"):
                        col1, col2 = st.columns(2)

                        with col1:
                            new_title = st.text_input(
                                "Title",
                                value=edit_node.content.get('title', '')
                            )

                            new_description = st.text_area(
                                "Description",
                                value=edit_node.content.get('description', ''),
                                height=150
                            )

                        with col2:
                            props = edit_node.content.get('properties', {})

                            new_importance = st.slider(
                                "Importance",
                                0.0, 1.0,
                                float(props.get('importance', 0.5)),
                                0.1
                            )

                            new_happiness = st.slider(
                                "Happiness",
                                0.0, 1.0,
                                float(props.get('happiness', 0.5)),
                                0.1
                            )

                            new_domain = st.selectbox(
                                "Domain",
                                get_life_domains(),
                                index=get_life_domains().index(props.get('domain', 'life_milestone'))
                                if props.get('domain') in get_life_domains() else 0
                            )

                        update_submitted = st.form_submit_button("Update Event")

                        if update_submitted:
                            # Update node
                            edit_node.content['title'] = new_title
                            edit_node.content['description'] = new_description
                            edit_node.content['properties']['importance'] = new_importance
                            edit_node.content['properties']['happiness'] = new_happiness
                            edit_node.content['properties']['domain'] = new_domain

                            st.success(f"✅ Updated: {new_title}")
                            st.rerun()

                    # Delete button
                    if st.button(f"🗑️ Delete Event: {edit_node_id}", type="secondary"):
                        # Remove from universe
                        del universe.nodes[edit_node_id]

                        # Remove from parent's children
                        for node in universe.nodes.values():
                            if edit_node_id in node.children_ids:
                                node.children_ids.remove(edit_node_id)

                        st.success(f"✅ Deleted: {edit_node_id}")
                        st.rerun()

<<<<<<< HEAD
=======
            # Save & Publish at bottom of Life Timeline tab
            if st.session_state.universe is not None:
                render_save_and_publish(st.session_state.universe)

>>>>>>> feature/feature/hilbert-ui-enhancements
    # ========================================================================
    # TAB 1: CREATE NEW UNIVERSE
    # ========================================================================
    with tabs[1]:
        st.header("Create New Universe")

        if st.session_state.universe is not None:
            st.info("Universe already loaded. Create new or modify existing?")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                "Universe Name",
                value="New Universe",
                help="Give your universe a descriptive name. This will appear in visualizations and file exports."
            )
            description = st.text_area(
                "Description",
                value="A personal AI multiverse",
                help="Describe what this universe represents. This helps document the purpose and contents of your multiverse."
            )
            dimension = st.number_input(
                "RKHS Dimension",
                min_value=2,
                max_value=1000,
                value=8,
                help="Dimensionality of the Reproducing Kernel Hilbert Space. Higher dimensions allow more complex "
                     "relationships but require more memory. Common values: 8 (simple), 64-128 (moderate), 768 (like MPNet embeddings)."
            )

        with col2:
            kernel_type = st.selectbox(
                "Kernel Type",
                ["rbf", "linear", "polynomial", "cosine"],
                help="**RBF (Radial Basis Function)**: Gaussian kernel, good for smooth similarity decay. Most common choice.\n\n"
                     "**Linear**: Simple dot product, computationally efficient.\n\n"
                     "**Polynomial**: Polynomial similarity, captures complex interactions.\n\n"
                     "**Cosine**: Angle-based similarity, ignores magnitude."
            )

            if kernel_type == "rbf":
                gamma = st.slider(
                    "Gamma (RBF parameter)",
                    0.1, 10.0, 1.0,
                    help="Controls the width of the RBF kernel. Lower gamma (0.1-0.5) = broad influence, nodes remain similar over larger distances. "
                         "Higher gamma (2.0-10.0) = narrow influence, only very close nodes are considered similar. Default 1.0 works well for most cases."
                )
                kernel_params = {"gamma": gamma}
            else:
                kernel_params = {}

            n_nodes = st.number_input(
                "Initial Nodes",
                min_value=1,
                max_value=50000,
                value=100,
                help="Number of nodes to create initially. Start with 100 for testing. "
                     "You can add more nodes later using the Fork tab. Maximum 50,000 nodes."
            )

        if st.button("🌟 materialize Universe"):
            with st.spinner("materializing universe..."):
                universe = RKHSUniverse(
                    name=name,
                    description=description,
                    dimension=dimension,
                    kernel_type=kernel_type,
                    kernel_params=kernel_params,
                    nodes={},
                    edges=[],
                    metadata={"created": datetime.now().isoformat()}
                )

                # Generate initial nodes
                for i in range(n_nodes):
                    position = [np.random.randn() for _ in range(min(dimension, 3))]
                    kernel_features = [np.random.randn() for _ in range(dimension)]

                    node = RKHSNode(
                        id=f"node_{i:06d}",
                        position=position,
                        content={"title": f"Node {i}", "description": ""},
                        metadata={},
                        kernel_features=kernel_features,
                        timestamp=datetime.now().isoformat(),
                        parent_ids=[],
                        children_ids=[]
                    )
                    universe.nodes[node.id] = node

                st.session_state.universe = universe
                st.success(f"✅ materialized universe with {n_nodes} nodes")

        # Precompute Fork Paths for Entire Universe
        if st.session_state.universe is not None and LLM_AVAILABLE:
            st.divider()
            st.subheader("🔮 Precompute Alternate Paths for Universe")
            st.markdown("Generate AI-powered fork paths for all nodes in the universe")

            precompute_col1, precompute_col2 = st.columns(2)

            with precompute_col1:
                precompute_max_nodes = st.number_input(
                    "Max Source Nodes",
                    min_value=1,
                    max_value=100,
                    value=min(10, len(st.session_state.universe.nodes)),
                    help="Maximum number of nodes to use as fork starting points. Higher = more coverage but more expensive."
                )

                precompute_steps = st.number_input(
                    "Steps per Path",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="precompute_steps",
                    help="Number of intermediate steps in each generated path"
                )

            with precompute_col2:
                precompute_forks_per_node = st.number_input(
                    "Forks per Node",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="Number of alternate paths to generate from each source node"
                )

                precompute_model = st.selectbox(
                    "Model",
                    ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", "gpt-4o-mini"],
                    key="precompute_model",
                    help="LLM model for generation. Haiku 4.5 is fastest/cheapest, Sonnet 4.5 is highest quality."
                )

            # Calculate cost estimate
            universe = st.session_state.universe
            total_llm_calls = min(precompute_max_nodes, len(universe.nodes)) * precompute_forks_per_node
            estimated_cost = total_llm_calls * 0.01  # Rough estimate
            st.info(f"📊 Will generate ~{total_llm_calls} paths. Estimated cost: ~${estimated_cost:.2f}")

            if st.button("🚀 Precompute All Fork Paths", type="primary"):
                if not require_auth("precompute fork paths"):
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Select source nodes (prioritize nodes with few children)
                    node_scores = []
                    for nid, node in universe.nodes.items():
                        # Prefer nodes with content and few existing children
                        score = len(node.content.get('description', '')) - (len(node.children_ids) * 100)
                        node_scores.append((score, nid))

                    node_scores.sort(reverse=True)
                    source_nodes = [nid for _, nid in node_scores[:precompute_max_nodes]]

                    total_generated = 0
                    total_operations = len(source_nodes) * precompute_forks_per_node

                    for idx, source_nid in enumerate(source_nodes):
                        source_node = universe.nodes[source_nid]

                        # Find potential target nodes (nodes with some distance)
                        candidate_targets = [
                            nid for nid in universe.nodes.keys()
                            if nid != source_nid and nid not in source_node.parent_ids
                        ]

                        if not candidate_targets:
                            continue

                        # Generate multiple fork paths from this source
                        for fork_idx in range(precompute_forks_per_node):
                            operation_num = idx * precompute_forks_per_node + fork_idx + 1
                            progress_bar.progress(operation_num / total_operations)
                            status_text.text(f"Generating path {operation_num}/{total_operations} from {source_node.content.get('title', source_nid)[:30]}...")

                            # Pick random target
                            import random
                            target_nid = random.choice(candidate_targets)
                            target_node = universe.nodes[target_nid]

                            # Generate path
                            try:
                                universe_type = universe.metadata.get('type', 'unknown')
                                subject_name = universe.metadata.get('subject', None)

                                # Adaptive prompt based on universe type
                                if universe_type == 'life_timeline' and subject_name:
                                    context_intro = f"""Alternate timeline for {subject_name}'s life.
IMPORTANT: {subject_name} is the same person throughout."""
                                    event_type = "life events"
                                else:
                                    context_intro = f"""Alternate path through {universe.name}.
Universe type: {universe_type if universe_type != 'unknown' else 'conceptual space'}"""
                                    event_type = "intermediate states/concepts"

                                prompt = f"""{context_intro}

Starting: {source_node.content.get('title', source_nid)}
{source_node.content.get('description', '')}

Target: {target_node.content.get('title', target_nid)}
{target_node.content.get('description', '')}

Create {precompute_steps} {event_type} connecting these points. Each should flow naturally and maintain coherence.

Return ONLY JSON:
[{{"title": "...", "description": "...", "year": ...}}, ...]"""

                                response = litellm.completion(
                                    model=precompute_model,
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=1500,
                                    temperature=0.7 + (fork_idx * 0.15)
                                )

                                import json, re
                                content = response.choices[0].message.content
                                json_match = re.search(r'\[.*\]', content, re.DOTALL)

                                if not json_match:
                                    continue

                                events = json.loads(json_match.group())

                                # Create path nodes
                                prev_node_id = source_nid
                                prev_node = source_node

                                for step_idx, event in enumerate(events):
                                    new_id = f"precomp_f{idx}_p{fork_idx}_s{step_idx}_{datetime.now().timestamp()}"

                                    t = (step_idx + 1) / (precompute_steps + 1)
                                    new_position = [
                                        source_node.position[i] * (1 - t) + target_node.position[i] * t + np.random.randn() * 0.03
                                        for i in range(len(source_node.position))
                                    ]

                                    new_features = [
                                        source_node.kernel_features[i] * (1 - t) + target_node.kernel_features[i] * t + np.random.randn() * 0.03
                                        for i in range(len(source_node.kernel_features))
                                    ]

                                    new_node = RKHSNode(
                                        id=new_id,
                                        position=new_position,
                                        content={
                                            "title": event.get("title", f"Event {step_idx + 1}"),
                                            "description": event.get("description", ""),
                                            "year": event.get("year"),
                                            "precomputed": True,
                                            "llm_generated": True
                                        },
                                        metadata={
                                            "llm_model": precompute_model,
                                            "precompute_batch": datetime.now().isoformat()
                                        },
                                        kernel_features=new_features,
                                        timestamp=datetime.now().isoformat(),
                                        parent_ids=[prev_node_id],
                                        children_ids=[]
                                    )

                                    universe.nodes[new_id] = new_node
                                    prev_node.children_ids.append(new_id)

                                    # Create edge
                                    sim = compute_kernel_similarity(
                                        prev_node.kernel_features,
                                        new_features,
                                        universe.kernel_type,
                                        universe.kernel_params.get('gamma', 1.0)
                                    )

                                    edge = RKHSEdge(
                                        source_id=prev_node_id,
                                        target_id=new_id,
                                        weight=1.0,
                                        kernel_similarity=sim,
                                        transition_type="precomputed_fork",
                                        metadata={"precomputed": True}
                                    )
                                    universe.edges.append(edge)

                                    prev_node_id = new_id
                                    prev_node = new_node

                                # Connect to target
                                prev_node.children_ids.append(target_nid)
                                target_node.parent_ids.append(prev_node_id)

                                edge = RKHSEdge(
                                    source_id=prev_node_id,
                                    target_id=target_nid,
                                    weight=1.0,
                                    kernel_similarity=compute_kernel_similarity(
                                        prev_node.kernel_features,
                                        target_node.kernel_features,
                                        universe.kernel_type,
                                        universe.kernel_params.get('gamma', 1.0)
                                    ),
                                    transition_type="precomputed_fork",
                                    metadata={"precomputed": True}
                                )
                                universe.edges.append(edge)
                                total_generated += 1

                            except Exception as e:
                                st.warning(f"Error generating path {operation_num}: {e}")
                                continue

                    progress_bar.progress(1.0)
                    status_text.text("✅ Complete!")
                    st.success(f"🎉 Precomputed {total_generated} alternate paths across the universe!")
                    save_user_multiverse(universe)

                except Exception as e:
                    st.error(f"Precompute error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

<<<<<<< HEAD
=======
            # Save & Publish at bottom of Create tab
            if st.session_state.universe is not None:
                render_save_and_publish(st.session_state.universe)

>>>>>>> feature/feature/hilbert-ui-enhancements
    # ========================================================================
    # TAB 2: LOAD UNIVERSE
    # ========================================================================
    with tabs[2]:
        st.header("Load Universe")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Option 1: Upload File (< 200 MB)")
            uploaded_file = st.file_uploader(
                "Upload RKHS Universe (JSON)",
                type=['json'],
                help="Upload a previously saved RKHS multiverse from your computer. "
                     "The file must be in JSON format with .json extension and under 200 MB. "
                     "This will load all nodes, edges, and metadata from your saved universe."
            )

            if uploaded_file:
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                universe = load_rkhs_universe(temp_path)
                if universe:
                    st.session_state.universe = universe
                    apply_auto_filter_if_needed(universe)
<<<<<<< HEAD
                    st.success(f"✅ Loaded: {universe.name}")
=======
                    st.success(f"✅ Loaded: {universe.name} - {len(universe.nodes)} nodes loaded. Now go to any tab to explore!")
>>>>>>> feature/feature/hilbert-ui-enhancements
                    st.json({
                        "nodes": len(universe.nodes),
                        "edges": len(universe.edges),
                        "dimension": universe.dimension,
                        "kernel_type": universe.kernel_type
                    })
<<<<<<< HEAD
=======
                    # Force rerun to ensure state persists
                    st.rerun()
>>>>>>> feature/feature/hilbert-ui-enhancements

                    # Instant visualization preview
                    st.divider()
                    st.subheader("📊 Quick Preview")

                    # Determine best preview mode based on universe size
                    n_nodes = len(universe.nodes)
                    if n_nodes <= 100:
                        preview_nodes = list(universe.nodes.values())
                        preview_mode = "All Nodes"
                    elif n_nodes <= 500:
                        import random
                        preview_nodes = random.sample(list(universe.nodes.values()), min(200, n_nodes))
                        preview_mode = f"Sample (200/{n_nodes} nodes)"
                    else:
                        import random
                        preview_nodes = random.sample(list(universe.nodes.values()), 100)
                        preview_mode = f"Sample (100/{n_nodes} nodes)"

                    st.info(f"Showing: {preview_mode}")

                    # Create instant 3D preview
                    try:
                        # Convert list of nodes to set of IDs
                        preview_node_ids = {node.id for node in preview_nodes}
                        preview_fig = create_3d_network_viz(
                            universe,
                            selected_nodes=preview_node_ids,
                            layout_type="position"
                        )
                        st.plotly_chart(preview_fig, width='stretch')

                        st.success("👆 Preview loaded! Go to the 🔍 Explore tab for full interactive controls.")
                    except Exception as e:
                        st.warning(f"Preview unavailable: {e}")
                        st.info("Go to the 🔍 Explore tab to explore your universe.")

            # Option 2: Pre-loaded universes
            st.divider()

            st.subheader("Option 2: Open Pre-loaded Universe")

            # Dictionary of pre-configured universe files
            # Use environment-aware paths relative to project root
            hilberts_dir = project_root / "hilberts"
<<<<<<< HEAD
            viewer_dir = hilberts_dir / "multiverse_viewer"

            PRELOADED_UNIVERSES = {
                "Fred Z Timeline": str(viewer_dir / "fred_z_timeline.rkhs.json"),
                "Codexspace v1": str(hilberts_dir / "codexspace_v1.rkhs.json"),
                "Codexspace Sample": str(viewer_dir / "codexspace_sample.rkhs.json"),
=======
            viewer_examples_dir = hilberts_dir / "viewer" / "examples"

            PRELOADED_UNIVERSES = {
                "Life of Fred": str(hilberts_dir / "life_of_fred.rkhs.json"),
                "Life of Christ": str(hilberts_dir / "life_of_christ.rkhs.json"),
                "Antiques Roadshow": str(hilberts_dir / "antiques_roadshow.rkhs.json"),
                "Codexspace v1": str(hilberts_dir / "codexspace_v1.rkhs.json"),
                "Codexspace Sample": str(viewer_examples_dir / "codexspace_sample.rkhs.json"),
>>>>>>> feature/feature/hilbert-ui-enhancements
            }

            selected_universe = st.selectbox(
                "Select Universe",
                options=list(PRELOADED_UNIVERSES.keys()),
                index=0,  # Default to first option
                help="Choose from pre-configured universe files that are already stored on the system. "
<<<<<<< HEAD
                     "**Fred Z Timeline**: Personal life timeline with fork points (9 nodes).\n\n"
=======
                     "**Life of Fred**: Personal timeline 1961-2025 with career forks (17 nodes).\n\n"
                     "**Life of Christ**: Biblical timeline with theological forks (9 nodes).\n\n"
                     "**Antiques Roadshow**: Collection of memorable appraisals (10 nodes).\n\n"
>>>>>>> feature/feature/hilbert-ui-enhancements
                     "**Codexspace v1**: Full 28K+ books universe from the PG19 dataset.\n\n"
                     "**Codexspace Sample**: Smaller sample for quick testing and exploration."
            )

            if st.button("📂 Load Selected Universe"):
                file_path = PRELOADED_UNIVERSES[selected_universe]
                if Path(file_path).exists():
                    with st.spinner(f"Loading {selected_universe}..."):
                        universe = load_rkhs_universe(file_path)
                        if universe:
                            st.session_state.universe = universe
                            apply_auto_filter_if_needed(universe)
<<<<<<< HEAD
                            st.success(f"✅ Loaded: {universe.name}")
=======
                            st.success(f"✅ Loaded: {universe.name} - {len(universe.nodes)} nodes loaded. Now go to any tab to explore!")
>>>>>>> feature/feature/hilbert-ui-enhancements
                            st.json({
                                "nodes": len(universe.nodes),
                                "edges": len(universe.edges),
                                "dimension": universe.dimension,
                                "kernel_type": universe.kernel_type
                            })
<<<<<<< HEAD
=======
                            # Force rerun to ensure state persists
                            st.rerun()
>>>>>>> feature/feature/hilbert-ui-enhancements

                            # Instant visualization preview
                            st.divider()
                            st.subheader("📊 Quick Preview")

                            # Determine best preview mode based on universe size
                            n_nodes = len(universe.nodes)
                            if n_nodes <= 100:
                                preview_nodes = list(universe.nodes.values())
                                preview_mode = "All Nodes"
                            elif n_nodes <= 500:
                                import random
                                preview_nodes = random.sample(list(universe.nodes.values()), min(200, n_nodes))
                                preview_mode = f"Sample (200/{n_nodes} nodes)"
                            else:
                                import random
                                preview_nodes = random.sample(list(universe.nodes.values()), 100)
                                preview_mode = f"Sample (100/{n_nodes} nodes)"

                            st.info(f"Showing: {preview_mode}")

                            # Create instant 3D preview
                            try:
                                # Convert list of nodes to set of IDs
                                preview_node_ids = {node.id for node in preview_nodes}
                                preview_fig = create_3d_network_viz(
                                    universe,
                                    selected_nodes=preview_node_ids,
                                    layout_type="position"
                                )
                                st.plotly_chart(preview_fig, width='stretch')

                                st.success("👆 Preview loaded! Go to the 🔍 Explore tab for full interactive controls.")
                            except Exception as e:
                                st.warning(f"Preview unavailable: {e}")
                                st.info("Go to the 🔍 Explore tab to explore your universe.")
                else:
                    st.error(f"❌ File not found: {file_path}")

        with col2:
            st.subheader("Quick Start")

            n_sample_nodes = st.number_input(
                "Sample universe size",
                min_value=10,
                max_value=30000,
                value=100,
                step=10,
                help="Number of nodes to generate in the sample universe. "
                     "Start with 100 for quick testing. Larger values (1000+) are useful for "
                     "testing performance with bigger datasets. Maximum is 30,000 nodes."
            )

            if st.button("🎲 Create Sample Universe"):
                # Require authentication for creating universes
                if not require_auth("create new universes"):
                    st.stop()

                with st.spinner(f"Generating {n_sample_nodes} nodes..."):
                    universe = create_sample_universe(n_sample_nodes)
                    st.session_state.universe = universe
                    apply_auto_filter_if_needed(universe)
                    st.success(f"✅ Created sample universe with {n_sample_nodes} nodes")

                    # Auto-save user's multiverse if authenticated
                    save_user_multiverse(universe)

    # ========================================================================
    # TAB 8: MATHEMATICS
    # ========================================================================
    with tabs[8]:
        st.header("Mathematical Analysis")
        
        if st.session_state.universe is None:
            st.warning("⚠️ No universe loaded.")
        else:
            universe = st.session_state.universe
            
            st.subheader("RKHS Properties")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Universe Specification:**
                - **Dimension:** {universe.dimension}
                - **Kernel Type:** {universe.kernel_type}
                - **Parameters:** {universe.kernel_params}
                - **Nodes:** {len(universe.nodes)}
                - **Edges:** {len(universe.edges)}
                """)
                
                st.markdown("""
                **RKHS Formalization:**
                
                The multiverse is embedded in a Reproducing Kernel Hilbert Space ℋ where:
                
                1. Each state s ∈ ℋ with ||s|| < ∞
                2. Kernel function K: ℋ × ℋ → ℝ measures similarity
                3. Inner product ⟨s₁, s₂⟩_ℋ defines geometry
                4. Transitions preserve kernel structure
                """)
            
            with col2:
                # Compute statistics
                if len(universe.nodes) > 1:
                    # Sample for efficiency
                    sample_nodes = list(universe.nodes.values())[:1000]
                    
                    # Compute pairwise distances
                    distances = []
                    similarities = []
                    
                    for i, node1 in enumerate(sample_nodes[:100]):
                        for node2 in sample_nodes[i+1:i+20]:
                            dist = np.linalg.norm(
                                np.array(node1.kernel_features) - np.array(node2.kernel_features)
                            )
                            distances.append(dist)
                            
                            sim = compute_kernel_similarity(
                                node1.kernel_features,
                                node2.kernel_features,
                                universe.kernel_type,
                                universe.kernel_params.get('gamma', 1.0)
                            )
                            similarities.append(sim)
                    
                    if distances:
                        st.metric("Mean Distance", f"{np.mean(distances):.4f}")
                        st.metric("Mean Similarity", f"{np.mean(similarities):.4f}")
                        st.metric("Distance Std Dev", f"{np.std(distances):.4f}")
                        
                        # Distance distribution
                        fig = go.Figure(data=[
                            go.Histogram(x=distances, name="Distances", nbinsx=30),
                        ])
                        fig.update_layout(
                            title="Distribution of Pairwise Distances",
                            xaxis_title="Distance",
                            yaxis_title="Count",
                            height=300
                        )
                        st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})
            
            st.divider()
            
            st.subheader("Kernel Operations")
            
            col1, col2 = st.columns(2)

            with col1:
                node1_id = st.selectbox(
                    "Node 1",
                    list(universe.nodes.keys()),
                    key="math_node1",
                    help="Select the first node for kernel similarity computation."
                )
                node2_id = st.selectbox(
                    "Node 2",
                    list(universe.nodes.keys()),
                    key="math_node2",
                    help="Select the second node for kernel similarity computation. "
                         "The kernel K(node1, node2) measures how similar these two nodes are in RKHS space."
                )
            
            with col2:
                if st.button("Compute Kernel"):
                    node1 = universe.nodes[node1_id]
                    node2 = universe.nodes[node2_id]
                    
                    sim = compute_kernel_similarity(
                        node1.kernel_features,
                        node2.kernel_features,
                        universe.kernel_type,
                        universe.kernel_params.get('gamma', 1.0)
                    )
                    
                    dist = np.linalg.norm(
                        np.array(node1.kernel_features) - np.array(node2.kernel_features)
                    )
                    
                    st.metric("Kernel Similarity K(s₁, s₂)", f"{sim:.6f}")
                    st.metric("Euclidean Distance ||s₁ - s₂||", f"{dist:.6f}")
                    
                    st.latex(r"K(s_1, s_2) = e^{-\gamma \|s_1 - s_2\|^2}")
            
<<<<<<< HEAD
            # Save universe
            st.divider()
            st.subheader("💾 Save Universe")

            filename = st.text_input(
                "Filename",
                value=f"{universe.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                help="Name for the saved universe file. Will be saved as JSON format. "
                     "Include the .json extension. Use descriptive names to identify your universe later. "
                     "Default includes the universe name and current date."
            )
            
            if st.button("Save to File"):
                # Require authentication for saving
                if not require_auth("save universes"):
                    st.stop()

                output_path = f"/mnt/user-data/outputs/{filename}"
                save_rkhs_universe(universe, output_path)
                st.success(f"✅ Saved to {output_path}")
                
                # Provide download
                with open(output_path, 'r') as f:
                    st.download_button(
                        "📥 Download",
                        f.read(),
                        file_name=filename,
                        mime="application/json"
                    )
=======
            # Save & Publish at bottom of Maths tab
            render_save_and_publish(universe)
>>>>>>> feature/feature/hilbert-ui-enhancements

if __name__ == "__main__":
    main()
