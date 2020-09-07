"""Database for sensor-width.

This database if used when we construct camera intrinsics from exif data.

Authors: Ayush Baid
"""
import os

import pandas as pd


class SensorWidthDatabase():
    """Database class for sensor-width, reading data from a csv file."""

    def __init__(self, csv_path=os.path.join('assets', 'camera_details', 'sensor_database.csv')):

        self.df = pd.read_csv(csv_path)

        # convert string to lower-case
        self.df['CameraMaker'] = self.df['CameraMaker'].str.lower()
        self.df['CameraModel'] = self.df['CameraModel'].str.lower()

    def lookup(self, make: str, model: str) -> float:
        """Look-up the sensor width given the camera make and model.

        Args:
            make (str): make of the camera
            model (str): model of the camera

        Returns:
            float: sensor-width in mm
        """

        # preprocess query strings
        make = make.split()[0].lower()
        model = model.lower()

        selection_condition = (self.df['CameraMaker'] == make) & (
            self.df['CameraModel'] == model)

        return self.df.loc[selection_condition, 'SensorWidth(mm)'].values[0]
