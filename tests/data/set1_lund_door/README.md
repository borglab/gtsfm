Data with explicit extrinsics and intrinsics from exif.

Also,
- [similarity_matrix.txt](similarity_matrix.txt)
- [cluster_tree.pkl](cluster_tree.pkl)

obtained using NetVlad and Metis partitioning at commit 6096b649, using
```bash
./run --dataset_dir tests/data/set1_lund_door --config_name unified --graph_partitioner metis image_pairs_generator.retriever.max_frame_lookahead=2
```
and used for testing in [test_tree_dask.py](../../utils/test_tree_dask.py).