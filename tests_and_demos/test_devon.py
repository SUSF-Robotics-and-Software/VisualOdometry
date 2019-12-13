"""
Run a test of the vis_odo system on the Devon Island dataset. To setup the
dataset see `datasets/readme.md`
"""

import threading
from datetime import datetime
import tkinter as tk
import time

import numpy as np
import cv2 as cv
from PIL import ImageTk, Image

from vis_odo import VisOdo, StereoImageSource, Pose

# Image window "new data" mutex and associated image store. Setting the image
# value to False will close the window. Setting to None will have no effect
mtx_image_window = threading.Lock()
mth_image_window_new_img = None

# Interval after which the image window should check for an update to the image
IMAGE_WINDOW_CHECK_INTERVAL_MS = 100

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

    print(f"{image_size}")

    # Start the image window
    image_window_thread = threading.Thread(
        target=image_window_start, args=(image_size))
    image_window_thread.start()

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

    # Close the window
    if mtx_image_window.acquire():
        mth_image_window_new_img = False
        mtx_image_window.release()

    # Print exit message
    print("End of cyclic activity, simulation stopped")


def cyclic_activity(sim_time_s, vis_odo, rov_pose_est):
    """
    Run the cyclic activity of the test suite
    """

    # Step the vis_odo processing
    vis_odo_status_rpt = vis_odo.step(sim_time_s)

    # Check for any errors in the status report
    if len(vis_odo_status_rpt) > 0:

        # Format the status report items and print
        for (name, data) in vis_odo_status_rpt.items():
            print(f"[{'CRT' if data.is_critical else 'WRN'}]"
                  f"{name}: {data.message}")

        if any([d.is_critical for d in vis_odo_status_rpt]):
            return (False, "A critical error occured in vis_odo processing!")

    # Get the delta frame step
    delta_pose_est = vis_odo.get_delta_pose_estimate()

    # If there's a new pose estimate transform the current pose by it and
    # inform the user
    if delta_pose_est:
        rov_pose_est.transform_by_pose(delta_pose_est)

        # Get the images used for the estimate and display them
        last_frame = vis_odo.get_last_frame()

        if last_frame:

            # Concat images
            mosaic = np.hstack((last_frame.img_left, last_frame.img_right))

            # Add timestamp in bottom right
            mosaic_shape = np.shape(mosaic)
            text_pos = (mosaic_shape[1] - 100, mosaic_shape[0] - 20)
            cv.putText(
                mosaic, f"t = {last_frame.sim_time_s:.2f} s", text_pos,
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1,
                lineType=cv.LINE_AA)

            # Update the image in the image window
            image_window_update(mosaic)

        print(f"[---] {sim_time_s:.2f}: Rover pose estimate updated")

    # Finally return true with no message
    return (True, "")


def image_window_start(*args, **kwargs):
    """
    Start the thread which displays the image passed into it by
    `image_window_update`.
    """

    print("Starting image display window")

    # Create the tkinter window which will display the images we're working
    # with
    img_window = tk.Tk()
    img_window.title("Visual Odometry: Devon Island Test")

    # Start with a black image
    blank_image = np.zeros((args[0], args[1], 3), np.uint8)

    # Create a canvas in the window which will hold our images
    img_canvas = tk.Canvas(
        img_window,
        width=args[0],
        height=args[1])
    image_on_canvas = img_canvas.create_image(
        0, 0, image=blank_image, anchor=tk.NW)

    img_window.after(
        0, image_window_check, img_window, img_canvas, image_on_canvas)
    img_window.mainloop()


def image_window_check(img_window, img_canvas, image_on_canvas):
    """
    Check to see if an updated image is available. To update the image call
    `image_window_update` with the OpenCV image (numpy ndarray)
    """
    global mth_image_window_new_img

    if mtx_image_window.acquire(False):

        # If there's a new image (if it's a numpy array)
        if type(mth_image_window_new_img).__module__ == np.__name__:

            # Update the tkinter window
            img = ImageTk.PhotoImage(
                image=Image.fromarray(mth_image_window_new_img))
            img_canvas.create_image(0, 0, image=img, anchor=tk.NW)
            image_on_canvas = img_canvas.itemconfig(image_on_canvas, image=img)

        # Or if we have an instruction to destroy the window
        elif mth_image_window_new_img == False:
            img_window.destroy()

        # Release the lock
        mtx_image_window.release()

    # Add another check
    img_window.after(
        IMAGE_WINDOW_CHECK_INTERVAL_MS, image_window_check,
        img_window, img_canvas, image_on_canvas)

def image_window_update(img):
    """
    Update the image in the image window
    """
    global mth_image_window_new_img

    # Acquire the lock, update the image, and release the lock
    if mtx_image_window.acquire():
        print(f"Updating image with one of size {img.shape}")
        mth_image_window_new_img = img
        mtx_image_window.release()

if __name__ == "__main__":
    threading.Thread(target=main).start()
