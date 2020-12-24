# gtsfm
End-to-end SFM pipeline based on GTSAM

## Installation

```python
pip install -e .
```

## Authors (alphabetically)

Ayush Baid, Frank Dellaert, Fan Jiang, Akshay Krishnan, John Lambert, Aishwarya Venkataramanan, Sushmita Warrier, Jing Wu, Xiaolong Wu

## Repository Structure

GTSFM is designed in an extremely modular way. Each module can be swapped out with a new one, as long as it implements the API of the module's abstract base class. The code is organized as follows:

- `gtsfm`
    - `averaging`
        - `rotation`
        - `translation`
    - `bundle`
    - `common`
    - `data_association`
    - `densify`
    - `frontend`
    - `loader`
    - `utils`
