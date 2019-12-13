"""
Run a test of the vis_odo system on the Devon Island dataset. To setup the
dataset see `datasets/readme.md`
"""

import threading
import time
from datetime import datetime

import numpy as np
import cv2 as cv
import faulthandler

from vis_odo import VisOdo, StereoImageSource, Pose
import image_window

IMGS_TO_SHOW = "MATCH_STRENGTH" # Either KEYPOINTS, RAW,

faulthandler.enable()

def main():
    """
    Run the main test on the Devon Island dataset
    """

    # Test name, dataset, and time log
    print("---- VISODO TEST ----")
    print("  Devon Island Dataset")
    print(f"  {datetime.now()}")

    # Setup the image source
    print(
        "Loading image source from the `params/image_source_devon.hjson`"
        " parameter file")
    image_source = StereoImageSource.from_dataset(
        "params/image_source_devon.hjson")

    print(f"Loaded timeline of {len(image_source.timeline)} frames")

    # Initialise the vis_odo system
    vis_odo = VisOdo()

    # Setup simulation timestep info
    sim_time_s = 0
    sim_time_step_s = 0.1

    # Initialise the pose estimate
    rover_pose_est = Pose()

    # Get the size of the images that will be processed, the window will
    # display both images side by side so we want the width to be 2 * width
    image_size = (image_source.image_size[1] * 2, image_source.image_size[0])

    # Start the image window
    image_window.start_image_window(
        "Visual Odometry: Devon Island Test", image_size)

    # Start the VisOdo processing
    vis_odo.start(image_source)

    # Run the main loop
    cyclic_act_running = True
    while cyclic_act_running:

        # Run the cyclic activity
        (cyclic_act_running, exit_msg) = cyclic_activity(
            sim_time_s, vis_odo, rover_pose_est)

        if cyclic_act_running:
            time.sleep(sim_time_step_s)
            sim_time_s += sim_time_step_s
        else:
            print(exit_msg)

    # Print exit message
    print("End of cyclic activity, simulation stopped")


def cyclic_activity(sim_time_s, vis_odo, rov_pose_est):
    """
    Run the cyclic activity of the test suite
    """

    # Update the sim_time for vis_odo
    vis_odo.set_sim_time(sim_time_s)

    # Step the vis_odo processing
    vis_odo_status_rpt = vis_odo.get_status_rpt()

    # Check for any errors in the status report
    if len(vis_odo_status_rpt) > 0:

        # Format the status report items and print
        for (name, data) in vis_odo_status_rpt.items():
            print(f"[{'CRT' if data['is_critical'] else 'WRN'}]"
                  f" {name}: {data['message']}")

        if any([data["is_critical"] for (name, data)
                in vis_odo_status_rpt.items()]):
            return (False, "A critical error occured in vis_odo processing!")

    # Get the delta frame step
    delta_pose_est = vis_odo.get_delta_pose_est()

    # If there's a new pose estimate transform the current pose by it and
    # inform the user
    if delta_pose_est:
        rov_pose_est.transform_by_pose(delta_pose_est)

        # Get the images used for the estimate and display them
        current_frame = vis_odo.get_current_frame()

        if current_frame:

            # Concat images
            if IMGS_TO_SHOW == "RAW":
                mosaic = np.hstack((current_frame.img_left,
                                    current_frame.img_right))
            elif IMGS_TO_SHOW == "KEYPOINTS":
                mosaic = np.hstack((
                    current_frame.imgs_processed["left_with_keypoints"],
                    current_frame.imgs_processed["right_with_keypoints"]))
            elif IMGS_TO_SHOW == "MATCH_STRENGTH":
                # If the first frame (no matches) then use the keypoints one
                if "left_match_strength" not in current_frame.imgs_processed:
                    mosaic = np.hstack((
                        current_frame.imgs_processed["left_with_keypoints"],
                        current_frame.imgs_processed["right_with_keypoints"]))
                else:
                    mosaic = np.hstack((
                        current_frame.imgs_processed["left_match_strength"],
                        current_frame.imgs_processed["right_match_strength"]))

            # Add timestamp in bottom right
            mosaic_shape = np.shape(mosaic)
            text_pos = (mosaic_shape[1] - 100, mosaic_shape[0] - 20)
            cv.putText(
                mosaic, f"t = {current_frame.sim_time_s:.2f} s", text_pos,
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1,
                lineType=cv.LINE_AA)

            # Update the image in the image window
            if image_window.img_window_ref:
                image_window.img_window_ref.update_image(mosaic)

        print(f"Pose estimate updated")

    # Finally return true with no message
    return (True, "")


if __name__ == "__main__":
    # Gives us nice auto-exit behaviour
    threading.Thread(target=main).start()
