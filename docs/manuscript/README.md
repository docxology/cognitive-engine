# Parse Trees and Typed Bindings in Cognitive Engine

The [abstract](00_abstract.md), [methods](02_methodology.md),
[results](03_results.md), [configuration](config.yaml), and
[bibliography](references.bib) live here. The [offline fixture](../../tests/test_symbolic_offline.py)
contains the inputs and assertions. The [earlier scope audit](../archive/manuscript_scope_audit.md)
is retained for historical context.

## Build the manuscript

Run these commands from the repository root with Git, uv, Graphviz (`dot`),
Pandoc, and XeLaTeX installed. Use a committed revision to reproduce the
reported evidence. The isolated source clone and render outputs stay under the
temporary directory printed at the end.

```bash
source=$(pwd -P)
commit=$(git rev-parse HEAD)
sandbox=$(mktemp -d)
template_checkout="$sandbox/template"
git clone https://github.com/docxology/template.git "$template_checkout"
git -C "$template_checkout" checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
mkdir -p "$template_checkout/projects/working"
git clone "$source" "$template_checkout/projects/working/cognitive-engine"
cd "$template_checkout/projects/working/cognitive-engine"
git checkout --detach "$commit"
printf '\n/output/\n' >> .git/info/exclude

uv run --no-project --python 3.12 --with 'networkx==3.7' \
  python -m unittest discover -s tests -p test_symbolic_offline.py -v
uv run --no-project --python 3.12 --with 'networkx==3.7' python - <<'PYTHON'
import json
import runpy
import subprocess
from pathlib import Path

receipt = runpy.run_path("tests/test_symbolic_offline.py")["symbolic_fixture"]()
data = Path("output/data/symbolic_fixture.json")
figure = Path("output/figures/symbolic_fixture.png")
data.parent.mkdir(parents=True, exist_ok=True)
figure.parent.mkdir(parents=True, exist_ok=True)
data.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

lines = [
    "digraph fixture {",
    'graph [bgcolor="white", fontname="DejaVu Sans", rankdir=TB, dpi=160];',
    'node [shape=box, style=rounded, fontname="DejaVu Sans"];',
    'edge [fontname="DejaVu Sans"];',
    'subgraph cluster_tree { label="Returned parse tree"; color="#557AA1";',
    f'root [label={json.dumps(receipt["tree_root"])}];',
]
for index, sentence in enumerate(receipt["sentence_labels"]):
    lines.extend([f"s{index} [label={json.dumps(sentence)}];", f"root -> s{index};"])
    for offset, word in enumerate(receipt["word_labels"][2 * index:2 * index + 2]):
        lines.extend([f"w{index}_{offset} [label={json.dumps(word)}];",
                      f"s{index} -> w{index}_{offset};"])
lines.extend(["}", 'subgraph cluster_bindings { label="Explicit binding graph"; color="#4C856B";'])
for start, relation, finish in receipt["bindings"]:
    lines.append(f"{json.dumps(start)} -> {json.dumps(finish)} [label={json.dumps(relation)}];")
lines.extend(["}", "}"])
subprocess.run(["dot", "-Tpng", "-o", str(figure)],
               input="\n".join(lines), text=True, check=True)
PYTHON

cd "$template_checkout"
uv sync --locked
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/cognitive-engine
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/cognitive-engine
printf '%s\n' "$template_checkout/projects/working/cognitive-engine/output"
```

The `output/` tree contains the fixture JSON in `data/`, its PNG in `figures/`,
the PDF in `pdf/`, HTML in `web/`, and render and validation records in
`reports/`. PDF and HTML use the same source sections and calculated fixture
labels.
