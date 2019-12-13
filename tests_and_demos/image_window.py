"""
Image window class used for most tests to display images as the test runs.
"""

from threading import Lock, Thread
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# After how many milliseconds should the image window check for a new image?
IMAGE_WINDOW_REFRESH_INTERVAL_MS = 100

# Hacky as hell, but fuck tkinter. MUST CALL start_image_window BEFORE USING!
img_window_ref = None

def ndarray_to_photoimage(array):
    """
    Helper function to convert a numpy ndarray to a photoimage for use in
    tkinter.
    """
    return ImageTk.PhotoImage(image=Image.fromarray(array))


def start_image_window(window_title, image_size):
    """
    Start the image window processing on another thread. This is a non-blocking
    function
    """

    # All of the TkInter setup needs to be done in a thread here
    Thread(target=_start, 
        args=(window_title, image_size[0], image_size[1])).start()


def _start(*args):
    """
    Start all the tk stuff in another thread
    """
    global img_window_ref

    root = tk.Tk()
    root.title(args[0])
    img_window_ref = ImageWindow((args[1], args[2]), root)
    root.mainloop()


class ImageWindow(tk.Frame):
    """
    Window used to display the images being processed by a test run
    """

    def __init__(self, image_size, master=None):
        tk.Frame.__init__(self, master)

        print(f"Creating image window of size {image_size}")

        # Multithreading support. Use ImageWindow.update_image(img) to post a
        # new image
        self.image_lock = Lock()
        self.pending_image = None

        # Create the canvas to hold the images
        self.canvas = tk.Canvas(
            master, width=image_size[0], height=image_size[1])
        self.canvas.pack()

        # Image store - Tkinter needs to hold a reference to the image in the
        # canvas, otherwise it will fail to show and crash the app
        self.images = []

        # First image shall be an empty black image
        self.images.append(ndarray_to_photoimage(
            np.zeros((image_size[1], image_size[0], 3), np.uint8)))

        # Put the first image on the canvas
        self.image_on_canvas = self.canvas.create_image(
            0, 0, image=self.images[0], anchor=tk.NW)

        # Add the check for update function to the event loop. Note we use
        # 0 delay here rather than the refresh interval so that a new image
        # is acquired immediately if available
        self.after(0, self.check_for_new_image)


    def check_for_new_image(self):
        """
        Check to see if a new image has been posted to the shared resource area
        """

        # Attempt to get the lock (non-blocking so we don't hang here)
        if self.image_lock.acquire(False):

            # Check if the new image is an ndarray
            if type(self.pending_image).__module__ == np.__name__:

                # Append the image to the array and update the displayed image
                self.images.append(ndarray_to_photoimage(self.pending_image))
                self.canvas.itemconfig(
                    self.image_on_canvas, image=self.images[-1])

                # Clear the pending image
                self.pending_image = None

            # Release the lock
            self.image_lock.release()

        # Add this function back on the event loop so we check periodically
        self.after(IMAGE_WINDOW_REFRESH_INTERVAL_MS, self.check_for_new_image)


    def update_image(self, image):
        """
        Update the image in the window
        """

        # Get the lock to access the image shared data
        if self.image_lock.acquire():

            # Update the image and release lock
            self.pending_image = image
            self.image_lock.release()
