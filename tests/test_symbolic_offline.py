"""Local implementation-backed checks for the standalone symbolic modules."""

import runpy
import unittest
from pathlib import Path

FRACTAL = Path(__file__).resolve().parents[1] / "cognitive_engine" / "fractal"


def _standalone_symbolic_classes():
    # The package initializer imports the hybrid stack. This fixture loads
    # only the fixed symbolic source files that it exercises.
    core = runpy.run_path(str(FRACTAL / "core.py"))
    binding = runpy.run_path(str(FRACTAL / "binding.py"))
    templates = runpy.run_path(str(FRACTAL / "templates.py"))
    return core["FractalSystem"], binding["BindingMechanism"], templates["Template"]


def symbolic_fixture():
    system_type, binding_type, template_type = _standalone_symbolic_classes()
    parser = system_type()
    tree = parser.parse("Oak shelf. Clay jar")
    sentences = tree["children"]
    words = [node for sentence in sentences for node in sentence["children"]]

    graph = system_type()
    intake, shelf, archive = [graph.create_root(name) for name in ("Intake", "Shelf", "Archive")]
    binding = binding_type(graph)
    binding.create_binding(intake.id, shelf.id, "stores_in")
    binding.create_binding(shelf.id, archive.id, "catalogued_in")

    route = binding.reason_with_bindings(intake.id, ["stores_in", "catalogued_in"])
    reversed_route = binding.reason_with_bindings(intake.id, ["catalogued_in", "stores_in"])

    template = template_type("catalog", [{"id": "root", "name": "$title", "level": 0}])
    new_root_id = template.instantiate(graph, {"title": "Catalog"})["root"]
    return {
        "tree_root": tree["name"],
        "sentence_labels": [node["properties"]["text"] for node in sentences],
        "word_labels": [node["properties"]["text"] for node in words],
        "tree_nodes": 1 + len(sentences) + len(words),
        "indexed_parser_nodes": len(parser.nodes_by_id),
        "indexed_parser_edges": parser.graph.number_of_edges(),
        "bindings": [
            (graph.get_node(item.source_id).name, item.relation_type, graph.get_node(item.target_id).name)
            for item in binding.bindings.values()
        ],
        "binding_count": graph.graph.number_of_edges(),
        "matching_endpoints": [graph.get_node(node).name for node in route],
        "reversed_endpoints": reversed_route,
        "template_root": graph.get_node(new_root_id).name,
    }


class SymbolicOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = symbolic_fixture()

    def test_parse_retains_words_in_a_nested_tree(self):
        self.assertEqual(self.results["sentence_labels"], ["Oak shelf", "Clay jar"])
        self.assertEqual(self.results["word_labels"], ["Oak", "shelf", "Clay", "jar"])
        self.assertEqual(self.results["tree_nodes"], 7)
        self.assertEqual(self.results["indexed_parser_nodes"], 1)
        self.assertEqual(self.results["indexed_parser_edges"], 0)

    def test_bindings_follow_the_declared_relation_order(self):
        self.assertEqual(self.results["binding_count"], 2)
        self.assertEqual(self.results["matching_endpoints"], ["Archive"])
        self.assertEqual(self.results["reversed_endpoints"], [])

    def test_root_template_substitutes_its_parameter(self):
        self.assertEqual(self.results["template_root"], "Catalog")
