"""Utility functions for handling subgraphs in GTSFM.

Authors: Zongyue Liu
"""

from typing import Dict, List, Tuple, Any

import numpy as np


def group_results_by_subgraph(
    results: Dict[Tuple[int, int], Any],
    subgraphs: List[List[Tuple[int, int]]],
) -> List[Dict[Tuple[int, int], Any]]:
    """Group results by subgraph.
    
    Args:
        results: Dictionary mapping image pairs to their results.
        subgraphs: List of subgraphs, where each subgraph is a list of image pairs.
        
    Returns:
        List of dictionaries, where each dictionary contains the results for a subgraph.
    """
    subgraph_results = []
    
    for subgraph_pairs in subgraphs:
        # Create a dictionary containing only the results for this subgraph
        subgraph_result = {
            pair: results[pair] 
            for pair in subgraph_pairs 
            if pair in results
        }
        subgraph_results.append(subgraph_result)
    
    return subgraph_results
