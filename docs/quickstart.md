# Verify the symbolic modules locally

Use the [installation command](installation.md) to run the
[Python fixture](../tests/test_symbolic_offline.py). It checks three
operations on fixed examples:

- Parse short text into a root, sentence nodes, and word nodes. The
  nested result contains descendants that are absent from the graph index.
- Bind registered roots and follow an ordered sequence of relations.
  Reversing those relation names returns an empty result for the fixture.
- Instantiate a root template and verify the substituted name.

The fixture compares names, ordering, and counts. Node UUIDs are generated
at runtime, so the evidence summary omits them. The
[results section](manuscript/03_results.md) reports what the test observes.
The [build instructions](manuscript/README.md) regenerate the JSON evidence
and figure in a disposable Template checkout.

The check covers the Python symbolic implementation. Earlier neural
reports and the TypeScript fixture layer have separate outputs and methods.
