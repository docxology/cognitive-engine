# Methods

The fixture loads the repository's fixed `core.py`, `binding.py`, and `templates.py` modules in isolation and constructs fresh instances per observation. The parser splits the literal input `Oak shelf. Clay jar` at the period-space boundary, then splits each sentence on whitespace. We count the returned root, sentence and word nodes, the entries in the parser's node index, and the directed edges in its graph.

The graph fixture creates three named roots. It places a directed `stores_in` binding from Intake to Shelf and a directed `catalogued_in` binding from Shelf to Archive. The experiment queries both relation sequences from Intake and records the endpoint names. The template check passes `Catalog` as the `title` argument to the root label `$title`. The record omits generated UUIDs and keeps stable labels and counts.
