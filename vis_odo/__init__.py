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

    def __init__(self, params_file_path):
        """
        Initialise the system by loading the necessary parameters.
        """
        pass

    def start(self, stereo_image_source_func):
        """
        Start the image processing loop.
        """
        pass

    def get_delta_pose_estimate(self):
        pass