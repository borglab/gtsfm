"""Functions to provide serialization support for GTSAM types.

Authors: Ayush Baid
"""
import pickle
from typing import Dict, List, Tuple

from distributed.protocol import dask_deserialize, dask_serialize
from gtsam import (
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Point3,
    Pose3,
    Rot3,
    SfmData,
    Unit3,
)

from gtsfm.common.sfm_result import SfmResult

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
    header = {"serializer": "pickle"}
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
        frame = "".join(frames)
    else:
        frame = frames[0]
    return Point3(pickle.loads(frame))


@dask_serialize.register(Rot3)
def serialize_Rot3(rot3: Rot3) -> Tuple[Dict, List[bytes]]:
    """Serialize Rot3 instance, and return serialized data."""
    header = {"serializer": "custom"}
    frames = [bytes(rot3.serialize(), "utf-8")]

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
    if len(frames) > 1:  # this may be cut up for network reasons
        frame = "".join(frames)
    else:
        frame = frames[0]

    serialized_str = frame.decode("utf-8")

    r = Rot3()
    r.deserialize(serialized_str)

    return r


@dask_serialize.register(Pose3)
def serialize_Pose3(pose3: Pose3) -> Tuple[Dict, List[bytes]]:
    """Serialize Pose3 instance, and return serialized data."""
    header = {"serializer": "custom"}
    frames = [bytes(pose3.serialize(), "utf-8")]

    return header, frames


@dask_deserialize.register(Pose3)
def deserialize_Pose3(header: Dict, frames: List[bytes]) -> Pose3:
    """Deserialize bytes into Pose3 instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        deserialized instance
    """
    if len(frames) > 1:  # this may be cut up for network reasons
        frame = "".join(frames)
    else:
        frame = frames[0]

    serialized_str = frame.decode("utf-8")

    pose3 = Pose3()
    pose3.deserialize(serialized_str)

    return pose3


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


@dask_serialize.register(Cal3Bundler)
def serialize_Cal3Bundler(obj: Cal3Bundler) -> Tuple[Dict, List[bytes]]:
    """Serialize Cal3Bundler instance, and return serialized data."""
    header = {"serializer": "custom"}
    frames = [bytes(obj.serialize(), "utf-8")]

    return header, frames


@dask_deserialize.register(Cal3Bundler)
def deserialize_Cal3Bundler(header: Dict, frames: List[bytes]) -> Cal3Bundler:
    """Deserialize bytes into Rot3 instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        deserialized instance.
    """
    if len(frames) > 1:  # this may be cut up for network reasons
        frame = "".join(frames)
    else:
        frame = frames[0]

    serialized_str = frame.decode("utf-8")

    obj = Cal3Bundler()
    obj.deserialize(serialized_str)

    return obj


@dask_serialize.register(PinholeCameraCal3Bundler)
def serialize_PinholeCameraCal3Bundler(
    obj: PinholeCameraCal3Bundler,
) -> Tuple[Dict, List[bytes]]:
    """Serialize PinholeCameraCal3Bundler instance, and return serialized data."""
    header = {"serializer": "custom"}
    frames = [bytes(obj.serialize(), "utf-8")]

    return header, frames


@dask_deserialize.register(PinholeCameraCal3Bundler)
def deserialize_PinholeCameraCal3Bundler(
    header: Dict, frames: List[bytes]
) -> PinholeCameraCal3Bundler:
    """Deserialize bytes into PinholeCameraCal3Bundler instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        deserialized instance.
    """
    if len(frames) > 1:  # this may be cut up for network reasons
        frame = "".join(frames)
    else:
        frame = frames[0]

    serialized_str = frame.decode("utf-8")

    obj = PinholeCameraCal3Bundler()
    obj.deserialize(serialized_str)

    return obj


@dask_serialize.register(SfmData)
def serialize_SfmData(
    obj: SfmData,
) -> Tuple[Dict, List[bytes]]:
    """Serialize SfmResult instance, and return serialized data."""
    header = {"serializer": "custom"}

    frames = [bytes(obj.serialize(), "utf-8")]

    return header, frames


@dask_deserialize.register(SfmData)
def deserialize_SfmData(header: Dict, frames: List[bytes]) -> SfmData:
    """Deserialize bytes into SfmData instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        deserialized instance.
    """
    if len(frames) > 1:  # this may be cut up for network reasons
        frame = "".join(frames)
    else:
        frame = frames[0]

    serialized_str = frame.decode("utf-8")

    obj = SfmData()
    obj.deserialize(serialized_str)

    return obj


@dask_serialize.register(SfmResult)
def serialize_SfmResult(
    obj: SfmResult,
) -> Tuple[Dict, List[bytes]]:
    """Serialize SfmResult instance, and return serialized data."""
    header = {"serializer": "custom"}

    # serialize SfmData instance to string as pickle cannot handle it
    info_dict = {
        "total_reproj_error": obj.total_reproj_error,
        "sfm_data_string": obj.sfm_data.serialize(),
    }

    # apply custom serialization on cameras
    serialized_sfm_result = pickle.dumps(info_dict)

    return header, [serialized_sfm_result]


@dask_deserialize.register(SfmResult)
def deserialize_SfmResult(header: Dict, frames: List[bytes]) -> SfmResult:
    """Deserialize bytes into SfmResult instance.

    Args:
        header: Header of the serialized data.
        frames: list of bytes in the serialized data.

    Returns:
        deserialized instance.
    """
    if len(frames) > 1:  # this may be cut up for network reasons
        frame = "".join(frames)
    else:
        frame = frames[0]

    # deserialize the top leveldictionary
    info_dict = pickle.loads(frame)

    # deserialize SfmData
    sfm_data = SfmData()
    sfm_data.deserialize(info_dict["sfm_data_string"])

    return SfmResult(
        sfm_data,
        total_reproj_error=info_dict["total_reproj_error"],
    )