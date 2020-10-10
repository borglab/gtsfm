"""Functions to provide serialization support for GTSAM types.

Authors: Ayush Baid
"""
from typing import Dict, List, Tuple

from distributed.protocol import dask_deserialize, dask_serialize
from gtsam import Rot3

""""
Serialization and deserialization function calls will be handled in the background by Dask,
and need not be called explicitly.
""""


@dask_serialize.register(Rot3)
def serialize_Rot3(rot3: Rot3) -> Tuple[Dict, List[bytes]]:
    """Serialize Rot3 instance, and return serialized data."""
    header = {}
    frames = [bytes(rot3.serialize(), 'utf-8')]

    return header, frames


@ dask_deserialize.register(Rot3)
def deserialize_Rot3(header: Dict, frames: List[bytes]) -> Rot3:
    """Deserialize bytes into Rot3 instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        Rot3: deserialized instance
    """
    serialized_str = frames[0].decode('utf-8')

    r = Rot3()
    r.deserialize(serialized_str)

    return r
