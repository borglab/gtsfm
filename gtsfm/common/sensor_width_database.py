"""Database for sensor-width.

This database if used when we construct camera intrinsics from exif data.

Authors: Ayush Baid
"""
from pathlib import Path

import pandas as pd

ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_SENSOR_DB_PATH = ASSETS_ROOT / "camera_details" / "sensor_database.csv"


class SensorWidthDatabase:
    """Database class for sensor-width, reading data from a csv file."""

    def __init__(self, csv_path: Path = DEFAULT_SENSOR_DB_PATH) -> None:
        """Initializes the database from a csv file."""

        self.df = pd.read_csv(csv_path)

        # convert string to lower-case
        self.df["CameraMaker"] = self.df["CameraMaker"].str.lower()
        self.df["CameraModel"] = self.df["CameraModel"].str.lower()

    def lookup(self, make: str, model: str) -> float:
        """Look-up the sensor width given the camera make and model.

        Args:
            make: Make of the camera.
            model: Model of the camera.

        Returns:
            Sensor width in mm.
        """

        # Preprocess query strings.
        lower_make = make.split()[0].lower()
        lower_make = lower_make.replace(" ", "").replace("-", "")
        lower_model = model.lower()
        lower_model = lower_model.replace(" ", "").replace("-", "").replace(lower_make, "")

        match_count = 0
        sensor_width = 0.0
        for _, row in self.df.iterrows():
            db_make = row["CameraMaker"]
            # Check camera make substring.
            if not (lower_make in db_make or db_make in lower_make):
                continue
            db_model = row["CameraModel"]
            db_model = db_model.replace(" ", "").replace("-", "").replace(db_make, "")
            # Check camera model substring.
            if not (lower_model in db_model or db_model in lower_model):
                continue
            sensor_width = row["SensorWidth(mm)"]
            # Return directly if found exact match.
            if lower_model == db_model:
                return sensor_width
            match_count += 1
            # Check if found multiple matches.
            if match_count > 1:
                break

        # Check if found unique match.
        if match_count == 0 or match_count > 1:
            raise LookupError(f"make='{make}' and model='{model}' not found in sensor database")

        if sensor_width <= 0.0:
            raise ValueError("Sensor width must be positive value.")

        return sensor_width
