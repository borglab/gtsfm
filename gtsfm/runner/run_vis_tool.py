"""Runner for the visualization tool only (back-end results are not desired or required).
React always runs the server on http://localhost:3000/ locally.

Author: Hayk Stepanyan
"""

import os

from pynpm import NPMPackage
from pathlib import Path

VIS_TOOL_PATH = Path(__file__).resolve().parent.parent.parent / "rtf_vis_tool"


def run_vis_tool():
    """Run the JS visualizer."""
    pkg = NPMPackage(os.path.join(VIS_TOOL_PATH, "package.json"))
    pkg.start()


if __name__ == "__main__":
    run_vis_tool()
