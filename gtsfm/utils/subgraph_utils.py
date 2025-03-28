"""Utility functions for handling subgraphs in GTSFM.

Authors: Zongyue Liu
"""

from typing import Dict, List, Tuple, TypeVar

T = TypeVar('T')


def group_results_by_subgraph(
    results_dict: Dict[Tuple[int, int], T], 
    subgraphs: List[List[Tuple[int, int]]]
) -> List[Dict[Tuple[int, int], T]]:
    """Group results by subgraph.
    
    Args:
        results_dict: Dictionary mapping image pairs to results.
        subgraphs: List of subgraphs, where each subgraph is a list of image pairs.
        
    Returns:
        List of dictionaries, where each dictionary contains results for one subgraph.
    """
    subgraph_results = []
    
    for subgraph in subgraphs:
        subgraph_dict = {}
        for pair in subgraph:
            if pair in results_dict:
                subgraph_dict[pair] = results_dict[pair]
        subgraph_results.append(subgraph_dict)
        
    return subgraph_results

