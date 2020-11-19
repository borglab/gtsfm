"""Functions to provide serialization support for GTSAM types.

Authors: Ayush Baid
"""
import pickle
from typing import Dict, List, Tuple

from distributed.protocol import dask_deserialize, dask_serialize
from gtsam import Point3, Rot3, Unit3

"""
Serialization and deserialization function calls will be handled in the background by Dask,
and need not be called explicitly.
"""


@dask_serialize.register(Point3)
def serialize_Point3(point3: Point3) -> Tuple[Dict, List[bytes]]:
    """Serialize Point3 instance.

    Args:
        point3: Point3 instance to serielize.

    Returns:
        Tuple[Dict, List[bytes]]: Serialized data.
    """
    header = {'serializer': 'pickle'}
    frames = [pickle.dumps(point3)]
    return header, frames


@dask_deserialize.register(Point3)
def deserialize_Point3(header: Dict, frames: List[bytes]) -> Point3:
    """Deserialize bytes into Point3 instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        deserialized instance
    """
    if len(frames) > 1:  # this may be cut up for network reasons
        frame = ''.join(frames)
    else:
        frame = frames[0]
    return Point3(pickle.loads(frame))


@dask_serialize.register(Rot3)
def serialize_Rot3(rot3: Rot3) -> Tuple[Dict, List[bytes]]:
    """Serialize Rot3 instance, and return serialized data."""
    header = {}
    frames = [bytes(rot3.serialize(), 'utf-8')]

    return header, frames


@dask_deserialize.register(Rot3)
def deserialize_Rot3(header: Dict, frames: List[bytes]) -> Rot3:
    """Deserialize bytes into Rot3 instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        deserialized instance
    """
    serialized_str = frames[0].decode('utf-8')

    r = Rot3()
    r.deserialize(serialized_str)

    return r


@dask_serialize.register(Unit3)
def serialize_Unit3(unit3: Unit3) -> Tuple[Dict, List[bytes]]:
    """Serialize Unit3 instance.

    Args:
        unit3: Unit3 instance to serielize.

    Returns:
        Tuple[Dict, List[bytes]]: Serialized data.
    """
    return serialize_Point3(unit3.point3())


@dask_deserialize.register(Unit3)
def deserialize_Unit3(header: Dict, frames: List[bytes]) -> Unit3:
    """Deserialize bytes into Unit3 instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        deserialized instance
    """
    point3 = deserialize_Point3(header, frames)

    return Unit3(point3)
