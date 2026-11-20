---
name: kumo-graph-visualization
description: Provide graph visuals in Kumo notebooks and SDK scripts, display or export relationships on request, and inspect Markdown schemas during model iteration.
---

# Graph visualization

Use `Graph.visualize()` to give users a view of the tables and relationships their model uses.
Reuse the graph for the current task; graph changes follow `kumoai-reference/references/graph_construction.md`.
A request to view or export an existing graph needs no model-training intake.

## Generated notebooks

After graph construction and `graph.validate()`, include a separate cell before snapshot creation or training:

```python
from IPython.display import display

if not hasattr(graph, 'visualize'):
    print('Graph preview needs a kumo-engine build including #134; upgrade when available. Continuing.')
else:
    try:
        display(graph.visualize())
    except Exception as error:
        print(f'Graph preview unavailable: {type(error).__name__}: {error}')
```

Introduce the cell as an interactive view where the user can explore tables, columns, and relationships.
It requires no snapshot and reads no source rows by default.
Continue the workflow normally; pause for visual review only if the user asks.

## Generated SDK scripts

Save the diagram next to the artifact, named after it (`churn_30d_train.graph.svg` for
`churn_30d_train.py`), before training, and report its path. `visualize` overwrites an existing
file, and here that is intended: the name derives from the script's own name, so the only file it
can replace is the diagram an earlier run of the same script wrote, and a rerun should redraw it
for the graph it trains on:

```python
from pathlib import Path

graph_path = Path(__file__).with_suffix('.graph.svg').resolve()   # this script's own diagram
if not hasattr(graph, 'visualize'):
    print('Graph preview needs a kumo-engine build including #134; upgrade when available. Continuing.')
else:
    try:
        _ = graph.visualize(graph_path)
        print(f'Graph diagram: {graph_path}')
    except Exception as error:
        print(f'Graph preview unavailable: {type(error).__name__}: {error}')
```

A missing visualization API, unsupported graph, or export failure should report the error and let modeling continue with the original graph.
The viewer supports up to 30 tables and does not support self-relations.

## Requested visuals and exports

Display the graph in a notebook or save the format the user requests: `.svg` for a diagram, `.pdf` for a diagram and complete schema index, or `.md` for a Mermaid diagram and schema tables.
`graph.visualize(EXPORT_PATH, show_cols=True)` also shows ordinary columns in the diagram.
`EXPORT_PATH` is the path the user named. When they named none, choose one that does not exist yet and report it; when the named file exists, say so and confirm before replacing it.
Use `layout='standard'` or `layout='compact'` to select an arrangement; the default is `'auto'`.
Export paths belong to the Python runtime; use `kumo-data-science/hosts/<host>/runtime.md` for display and file retrieval.

Keep samples off unless the user requests source values.
`show_samples=True` queries source data and includes sample values in saved notebook output; file exports omit them.

## Model iteration with Markdown

When considering which tables or relationships to change for a use case, export the graph for text inspection:

```python
import tempfile
from pathlib import Path

schema_path = Path(tempfile.mkdtemp(prefix='kumo-graph-')) / 'graph.md'   # a new directory: replaces nothing
_ = graph.visualize(schema_path)
print(f'Graph schema: {schema_path}')
```

Read the file at the printed path to inspect all declared columns, key and time roles, and relationships, including columns hidden in the diagram.
The assignment suppresses automatic notebook display.
Use this schema alongside model evaluation results to plan graph experiments; it does not measure predictive usefulness.
Markdown inspection is optional during iteration, not a prerequisite for generating or running a model.

Source: shipped package (`kumo.engine.graph.Graph.visualize`, `kumo.engine._visualization.SVG`)
