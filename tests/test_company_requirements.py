from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sps.company_requirements import load_requirement_groups


class CompanyRequirementsTests(unittest.TestCase):
    def test_missing_source_returns_unavailable_without_fabricated_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.json"

            self.assertIsNone(load_requirement_groups(path))

    def test_loads_requirements_from_source_json(self) -> None:
        payload = {
            "groups": [
                {
                    "group_id": "Group 1",
                    "title": "Source group",
                    "requirements": [
                        {
                            "requirement_id": "REQ-001",
                            "title": "Source title",
                            "description": "Source description",
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requirements.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            groups = load_requirement_groups(path)

        self.assertIsNotNone(groups)
        assert groups is not None
        self.assertEqual(groups[0].group_id, "Group 1")
        self.assertEqual(groups[0].requirements[0].requirement_id, "REQ-001")

    def test_invalid_or_duplicate_source_records_are_rejected(self) -> None:
        payload = {
            "groups": [
                {
                    "group_id": "Group 1",
                    "title": "Group",
                    "requirements": [],
                },
                {
                    "group_id": "Group 1",
                    "title": "Duplicate",
                    "requirements": [],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requirements.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate requirement group"):
                load_requirement_groups(path)


if __name__ == "__main__":
    unittest.main()
