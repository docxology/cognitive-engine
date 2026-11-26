# Introduction

The Cognitive Engine repository describes a hybrid framework with a symbolic subsystem alongside memory and neural components [@friedman2026cognitive]. The present study concerns the standalone Python symbolic core. Its public source uses nested node objects for parsed text and a NetworkX directed graph for registrations and explicit bindings [@hagberg2008networkx]. Since both representations involve nodes, readers can easily assume that traversing the graph also walks the hierarchy produced by parsing.

We ask three questions about the current implementation. Which parsed objects appear in the returned tree and in the graph index? How does the relation sequence affect the endpoint returned by a binding query? What does a minimal root-only template instantiate?
