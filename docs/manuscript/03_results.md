# Results

The parser returns `TextRoot`, two sentence nodes and four word nodes. The seven nested nodes and the single indexed root are separate observations. The parser instance has zero graph edges. Figure 1 contrasts the two output structures using the fixture labels.

![Fixture tree and explicit bindings. The parser returns a nested hierarchy while the independently registered roots form a directed binding path.](../../output/figures/symbolic_fixture.png)

The two explicit edges yield Archive as the sole endpoint for the sequence `stores_in`, `catalogued_in` from Intake. The reverse order returns an empty list because the first requested relation has no matching outgoing binding at Intake. The template instantiates a registered root named Catalog. The tests assert all three observations.
