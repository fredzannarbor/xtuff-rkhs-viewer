# Timeline Positioning Guide

## How to Make Timeline Flow from Bottom-Left to Upper-Right

### Current Positioning
The example timeline currently has positions that don't follow a strict temporal diagonal:

```json
"birth_1995": {
  "position": [0.0, 0.5, 0.5],  // X=0.0, Y=0.5
  "properties": {"year": 1995}
}
"elementary_school_2001": {
  "position": [0.6, 0.6, 0.5],  // X=0.6, Y=0.6
  "properties": {"year": 2001}
}
"middle_school_2007": {
  "position": [1.2, 0.5, 0.6],  // X=1.2, Y=0.5 (goes down!)
  "properties": {"year": 2007}
}
```

### Diagonal Layout Formula

To make the timeline flow diagonally from **bottom-left** (birth) to **upper-right** (latest event):

**Formula:**
```python
# For each event:
time_progress = (event_year - earliest_year) / (latest_year - earliest_year)

position = [
    time_progress * max_coord,      # X: increases with time (left → right)
    time_progress * max_coord,      # Y: increases with time (bottom → top)
    0.5                             # Z: constant (or use for importance/happiness)
]
```

### Example Transformation

For John's life timeline (1995-2025, assuming 30 years):

```json
// Birth (1995) - Bottom-Left
"birth_1995": {
  "position": [0.0, 0.0, 0.5],  // time_progress = 0.0
  "properties": {"year": 1995, "age": 0}
}

// Elementary School (2001) - Moving up diagonal
"elementary_school_2001": {
  "position": [0.2, 0.2, 0.5],  // time_progress = 6/30 = 0.2
  "properties": {"year": 2001, "age": 6}
}

// Middle School (2007) - Further up diagonal
"middle_school_2007": {
  "position": [0.4, 0.4, 0.5],  // time_progress = 12/30 = 0.4
  "properties": {"year": 2007, "age": 12}
}

// High School (2009)
"high_school_2009": {
  "position": [0.47, 0.47, 0.5],  // time_progress = 14/30 ≈ 0.47
  "properties": {"year": 2009, "age": 14}
}

// College (2013)
"college_2013": {
  "position": [0.6, 0.6, 0.5],  // time_progress = 18/30 = 0.6
  "properties": {"year": 2013, "age": 18}
}

// First Job (2017)
"first_job_2017": {
  "position": [0.73, 0.73, 0.5],  // time_progress = 22/30 ≈ 0.73
  "properties": {"year": 2017, "age": 22}
}

// Career Change (2022)
"career_change_2022": {
  "position": [0.9, 0.9, 0.5],  // time_progress = 27/30 = 0.9
  "properties": {"year": 2022, "age": 27}
}

// Present (2025) - Upper-Right
"present_2025": {
  "position": [1.0, 1.0, 0.5],  // time_progress = 30/30 = 1.0
  "properties": {"year": 2025, "age": 30}
}
```

### Advanced: Using Z-Axis for Meaning

You can use the Z-axis (3rd dimension) to represent another property:

**Option 1: Importance**
```python
position = [
    time_progress * max_coord,
    time_progress * max_coord,
    importance  # 0.0 = trivial, 1.0 = life-changing
]
```

**Option 2: Happiness Level**
```python
position = [
    time_progress * max_coord,
    time_progress * max_coord,
    happiness  # 0.0 = very sad, 1.0 = very happy
]
```

**Option 3: Life Domain Height**
```python
domain_heights = {
    "life_milestone": 0.9,
    "education": 0.7,
    "career": 0.6,
    "relationships": 0.5,
    "hobby": 0.3
}

position = [
    time_progress * max_coord,
    time_progress * max_coord,
    domain_heights[event_domain]
]
```

### Alternative: Branching Timelines

For alternate life paths (forks), you can offset Y slightly while keeping X as time:

```python
# Main timeline
main_event = [time_progress, time_progress, 0.5]

# Alternative path 1 (what if different college?)
alt1_event = [time_progress, time_progress + 0.1, 0.6]

# Alternative path 2 (what if different career?)
alt2_event = [time_progress, time_progress - 0.1, 0.4]
```

### Python Script to Recalculate Positions

```python
import json

def recalculate_timeline_positions(input_file, output_file, max_coord=3.0):
    """Recalculate all positions to flow bottom-left to upper-right"""

    with open(input_file, 'r') as f:
        timeline = json.load(f)

    # Extract years from all nodes
    years = []
    for node in timeline['nodes'].values():
        year = node['content']['properties'].get('year')
        if year:
            years.append(year)

    min_year = min(years)
    max_year = max(years)
    year_range = max_year - min_year

    # Update positions
    for node_id, node in timeline['nodes'].items():
        year = node['content']['properties'].get('year', min_year)
        time_progress = (year - min_year) / year_range if year_range > 0 else 0

        # Diagonal positioning
        x = time_progress * max_coord
        y = time_progress * max_coord

        # Z can be importance, happiness, or constant
        z = node['content']['properties'].get('importance', 0.5)

        node['position'] = [x, y, z]

    # Save updated timeline
    with open(output_file, 'w') as f:
        json.dump(timeline, f, indent=2)

    print(f"✅ Updated timeline positions")
    print(f"   Timeline spans {min_year} to {max_year}")
    print(f"   Diagonal from [0,0] to [{max_coord},{max_coord}]")

# Usage:
# recalculate_timeline_positions('example_life_timeline.json', 'example_life_timeline_diagonal.json')
```

### Visual Result

With diagonal positioning, your timeline will:
- Start at **bottom-left** (birth, early years)
- Flow **diagonally upward** to **upper-right** (present, future)
- Show clear chronological progression
- Allow Z-axis to represent emotional valence, importance, or domain

This creates an intuitive "climbing through life" visualization! 📈
