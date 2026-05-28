"""
TCM-SCI-Patent-Forge Smoke Tests
Basic import and instantiation tests to verify project integrity.
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfigModule(unittest.TestCase):
    """Test config.py module loads correctly."""

    def test_config_import(self):
        import config
        self.assertTrue(hasattr(config, 'FLAGSHIP_PROJECTS'))
        self.assertTrue(hasattr(config, 'VERIFICATION_METRICS'))
        self.assertTrue(hasattr(config, 'CONSTITUTION_TYPES'))

    def test_flagship_projects_count(self):
        import config
        self.assertEqual(len(config.FLAGSHIP_PROJECTS), 10)

    def test_five_elements(self):
        import config
        self.assertEqual(len(config.FIVE_ELEMENTS), 5)
        self.assertIn("木", config.FIVE_ELEMENTS)

    def test_pulse_types(self):
        import config
        self.assertIsInstance(config.PULSE_TYPES, list)
        self.assertGreater(len(config.PULSE_TYPES), 0)

    def test_verification_metrics_keys(self):
        import config
        for project in config.FLAGSHIP_PROJECTS:
            if project in config.VERIFICATION_METRICS:
                metrics = config.VERIFICATION_METRICS[project]
                self.assertIsInstance(metrics, dict)
                self.assertGreater(len(metrics), 0)


class TestProjectStructure(unittest.TestCase):
    """Test that all flagship projects have expected files."""

    def test_projects_directory_exists(self):
        projects_dir = PROJECT_ROOT / "projects"
        self.assertTrue(projects_dir.exists())

    def test_all_projects_have_demo(self):
        projects_dir = PROJECT_ROOT / "projects"
        for project in projects_dir.iterdir():
            if project.is_dir():
                demo_file = project / "demo.py"
                self.assertTrue(
                    demo_file.exists(),
                    f"{project.name} missing demo.py"
                )

    def test_directions_json_exists(self):
        directions_file = PROJECT_ROOT / "directions_100.json"
        self.assertTrue(directions_file.exists())

    def test_directions_json_valid(self):
        import json
        directions_file = PROJECT_ROOT / "directions_100.json"
        with open(directions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIsInstance(data, (list, dict))
        if isinstance(data, list):
            self.assertGreater(len(data), 0)


class TestPaperGenerator(unittest.TestCase):
    """Test paper_generator.py basic functionality."""

    def test_import(self):
        import paper_generator
        self.assertTrue(hasattr(paper_generator, 'generate_paper'))

    def test_load_direction(self):
        import paper_generator
        if hasattr(paper_generator, 'load_direction'):
            # Should not raise for valid direction
            try:
                result = paper_generator.load_direction("01")
                self.assertIsNotNone(result)
            except (FileNotFoundError, KeyError):
                pass  # Acceptable if direction file not found


class TestPatentGenerator(unittest.TestCase):
    """Test patent_generator.py basic functionality."""

    def test_import(self):
        import patent_generator
        self.assertTrue(hasattr(patent_generator, 'generate_patent'))


class TestDirectionsData(unittest.TestCase):
    """Test directions_100.json data integrity."""

    def setUp(self):
        import json
        path = PROJECT_ROOT / "directions_100.json"
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def test_is_collection(self):
        self.assertIsInstance(self.data, (list, dict))

    def test_has_entries(self):
        if isinstance(self.data, list):
            self.assertGreater(len(self.data), 0)
        elif isinstance(self.data, dict):
            self.assertGreater(len(self.data.keys()), 0)


class TestCLI(unittest.TestCase):
    """Test config.py CLI functionality."""

    def test_list_projects(self):
        import config
        # Should not raise
        config.list_projects()

    def test_show_project_status(self):
        import config
        # Should not raise for known project
        config.show_project_status("01-NeuroSymbolic-TCM")


if __name__ == "__main__":
    unittest.main()
