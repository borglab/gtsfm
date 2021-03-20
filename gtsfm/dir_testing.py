import os
from pathlib import Path

react_result_metrics_path = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics_2"
os.makedirs(react_result_metrics_path)
print('dir made in react src folder?')