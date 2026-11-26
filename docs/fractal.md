# Fractal symbolic structures

The Python symbolic code defines these entry points:

| Entry point | Implemented behavior |
| --- | --- |
| [FractalSystem and FractalNode](../cognitive_engine/fractal/core.py) | Create roots and nested nodes, index roots, and serialize the nested tree |
| [SymbolBinding and BindingMechanism](../cognitive_engine/fractal/binding.py) | Store directed relations between indexed nodes and follow a supplied sequence of relation names |
| [Template and TemplateLibrary](../cognitive_engine/fractal/templates.py) | Instantiate parameterized node definitions and organize reusable templates |

`FractalSystem.parse` splits on a period followed by a space, then splits
each nonempty sentence on whitespace. It creates a nested representation.
Child creation appends to a parent's list and leaves the system index
unchanged. A caller that needs graph lookup can start with explicitly
registered roots. The fixture uses roots for the binding example.

`BindingMechanism.reason_with_bindings` follows one directed relation
name per step. It returns the reachable endpoints or an empty result
when a step has no match. The `FractalSystem.reason` method produces a
canned reasoning trace for the prototype. The tested path traversal is
the narrower implementation with direct input/output checks.

The [offline checks](quickstart.md) show the current contracts and the
[manuscript](manuscript/README.md) records their evidence.
