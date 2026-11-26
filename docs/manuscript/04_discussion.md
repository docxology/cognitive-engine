# Discussion

The distinction matters when locating input text and traversing registered entities. The parse result supplies the sentence and word hierarchy. Queries over typed bindings require explicit endpoint registrations and bindings. In this fixture, reversing the relation sequence changes the result even though the directed edges are the same.

These observations characterize a fixed example and its implementation. The parser uses simple delimiters, so behavior on abbreviations, escaped punctuation, and different segment boundaries needs its own tests. The binding experiment does not measure performance on long paths, branching graphs, or cyclic graphs. The root-only template does not exercise templates with nested descendants and links. The core's general `reason()` method returns a fixed trace in this source snapshot. Its behavior is distinct from the relation-by-relation algorithm exercised here.
