"""
Visual Odometry module used to determine position and attitude from camera
images over time.
"""

import os.path

from sparam import load_params

from .pose import Pose
from .stereo_image_source import StereoImageSource


class VisOdo:
    """
    Visual Odometry Class, implements the Visual Odometry system for the rover.

        usage
        -----
            # Create new instance of the class
            vis_odo = VisOdo("path/to/params/file.hjson")

            # Create a new image source, in this case the dataset 'devon'
            stereo_image_source = StereoImageSource.from_dataset("devon")

            # Start the processing, passing in the image source
            vis_odo.start(stereo_image_source)

            # Every 5 seconds get the delta pose estimate, can be used to
            # propagate the overall pose of the vehicle
            while True:
                delta_pose = vis_odo.get_delta_pose_estimate()

                ...
    """

    def __init__(self):
        """
        Initialise the system by loading the necessary parameters.
        """

        # Get the path to this file, since relative imports are messed up and
        # are relative to the executing script's path, not the path of this
        # file
        loco_ctrl_dir = os.path.dirname(os.path.realpath(__file__))

        # Load the parameter file using sparam
        self.params = load_params.load_params_from_hjson(
            os.path.join(loco_ctrl_dir, '../params/vis_odo.hjson'))

        # Define internal data
        self.image_source = None
        self.last_frame = None


    def start(self, image_source):
        """
        Start the image processing loop.
        """
        # Assign the image source
        self.image_source = image_source

        return True

    def step(self, sim_time_s):
        """
        Step the visual odometry system. Returns a status report indicating the
        success of the operations.
        """

        # Get the latest frame from the image source
        self.last_frame = self.image_source.get_frame(sim_time_s)

        return {}

    def get_last_frame(self):
        """
        Get the last frame provided by the image source.
        """
        return self.last_frame

    def get_delta_pose_estimate(self):
        """
        Get the estimated delta pose (change in position/attitude) since the
        last frame
        """

        # TODO: Temp for debugging
        return Pose()
