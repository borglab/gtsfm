import argparse
from typing import List
import plotly.graph_objects as go

import numpy as np
from yaml import parse

from gtsfm.common.constraint import Constraint
from gtsam import Pose3, Rot3, Point3

def analyze_constraints(pose_constraints: List[Constraint], rotation=False):
    all_magnitudes = []
    for step_size in [1, 2, 3, 4, 5]:
        magnitudes = {}
        for constraint in pose_constraints:
            if abs(constraint.a - constraint.b) == step_size:
                if rotation:
                    error = np.rad2deg(np.linalg.norm(Rot3.Logmap(constraint.aTb.rotation())))
                else:
                    error = np.linalg.norm(constraint.aTb.translation())
                magnitudes[min(constraint.a, constraint.b)] = error
        all_magnitudes.append(dict(sorted(magnitudes.items())))
    property = "rotations (degrees)" if rotation else "translations (meters)"
    title = "between factor " + property
    fig = go.Figure()
    for i, magnitudes in enumerate(all_magnitudes):
        fig.add_trace(go.Scatter(x=list(magnitudes.keys()), y=list(magnitudes.values()),
                            mode='lines',
                            name=f"magnitudes @ {i+1} step"))
    fig.update_layout(title=title)
    fig.show()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation arguments")

    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        required=True,
        help="Path to fastlio_odom.txt poses",
    )

    parser.add_argument(
        "--rotations",
        type=bool,
        default=False,
        help="Path to fastlio_odom.txt poses",
    )

    args = parser.parse_args()
    pose_constraints = Constraint.read(args.constraints)
    analyze_constraints(pose_constraints, args.rotations)

