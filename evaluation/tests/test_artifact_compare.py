from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from artifact.compare import _pixel_diff


class ArtifactComparisonTests(unittest.TestCase):
    def compare(self, left: Image.Image, right: Image.Image) -> tuple[float, str]:
        with tempfile.TemporaryDirectory() as directory:
            left_path = Path(directory) / "paper.png"
            right_path = Path(directory) / "regenerated.png"
            left.save(left_path)
            right.save(right_path)
            return _pixel_diff(left_path, right_path)

    def test_rejects_a_chart_squeezed_vertically(self) -> None:
        paper = Image.new("RGB", (500, 280), "white")
        draw = ImageDraw.Draw(paper)
        draw.rectangle((50, 40, 180, 240), fill="black")
        draw.rectangle((220, 100, 350, 240), fill="blue")
        squeezed = paper.resize((500, 201))

        difference, dimensions = self.compare(paper, squeezed)

        self.assertTrue(math.isinf(difference))
        self.assertIn("aspect mismatch", dimensions)

    def test_accepts_a_change_in_raster_resolution(self) -> None:
        paper = Image.new("RGB", (500, 280), "white")
        higher_resolution = paper.resize((1000, 560))

        difference, _ = self.compare(paper, higher_resolution)

        self.assertEqual(difference, 0.0)

    def test_reports_actual_pixel_changes(self) -> None:
        paper = Image.new("RGB", (100, 60), "white")
        changed = Image.new("RGB", (100, 60), "black")

        difference, _ = self.compare(paper, changed)

        self.assertEqual(difference, 100.0)
