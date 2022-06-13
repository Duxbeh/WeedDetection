import functools
import queue
import threading
import time

from vimba import *
import cv2
import numpy as np
import vimba


# writer is a global object to make it easier to properly close if after the writer thread is done.
# There are certainly better ways to do this!
writer = None


class ImageRecorder:
    def __init__(self, cam: vimba.Camera, frame_queue: queue.Queue):
        self._cam = cam
        self._queue = frame_queue

    def __call__(self, cam: vimba.Camera, frame: vimba.Frame):
        # Place the image data as an opencv image and the frame ID into the queue
        if frame.get_status() == vimba.FrameStatus.Complete:
            self._queue.put((frame.as_numpy_ndarray(), frame.get_id()))

        # Hand used frame back to Vimba, so it can store the next image in this memory
        cam.queue_frame(frame)

    def _setup_camera(self):
        # setting the exposure
        self._cam.ExposureAuto.set('Off')
        self._cam.ExposureMode.set('Timed')
        self._cam.ExposureTime.set('4458.37')

        # setting the color
        self._cam.BalanceWhiteAuto.set('Continuous')
        self._cam.Hue.set('0.00')
        self._cam.Saturation.set('1.00')

        # setting image format
        # range of height (8-2056), range of width (8-2464)
        self._cam.Height.set('2056')
        self._cam.Width.set('2464')

        # setting pixel format
        self._cam.set_pixel_format(PixelFormat.Bgr8)

        # setting bandwidth of the data that will be streaming, range from (16250000-450000000)
        self._cam.DeviceLinkThroughputLimit.set('450000000')

    def _setup_software_triggering(self):
        # Always set the selector first so that following features are applied correctly!
        self._cam.TriggerSelector.set('FrameStart')

        # optional in this example but good practice as it might be needed for hardware triggering
        self._cam.TriggerActivation.set('RisingEdge')

        # Make camera listen to Software trigger
        self._cam.TriggerSource.set('Software')
        self._cam.TriggerMode.set('On')

    def record_images(self, num_pics: int):
        # This method assumes software trigger is desired. Free run image acquisition would work
        # similarly to get higher fps from the camera
        with vimba.Vimba.get_instance():
            with self._cam:
                try:
                    self._setup_camera()
                    self._setup_software_triggering()
                    self._cam.start_streaming(handler=self)
                    for i in range(num_pics):
                        print(i)
                        self._cam.TriggerSoftware.run()
                        # Delay between images can be adjusted or removed entirely
                        time.sleep(0.1)
                finally:
                    self._cam.stop_streaming()


def write_image(frame_queue: queue.Queue):
    global writer

    while True:
        img, id = frame_queue.get()
        print(f"took image {id} from queue")
        if writer is None:
            height, width = img.shape[0:2]
            size = (width, height)
            print(size)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # Warning. Shape here is expected as (width, height) np shapes are usually (height, width)
            writer = cv2.VideoWriter('/Users/admin/Desktop/pic/test.avi',
                                     fourcc,
                                     10,
                                     size)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frame_queue.task_done()


def main():
    num_pics = 25
    with vimba.Vimba.get_instance() as vmb:
        cams = vmb.get_all_cameras()

        frame_queue = queue.Queue()
        recorder = ImageRecorder(cam=cams[0], frame_queue=frame_queue)
        # Start a thread that runs write_image(frame_queue). Marking it as daemon allows the python
        # program to exit even though that thread is still running. The thread will then be stopped
        # when the program exits
        threading.Thread(target=functools.partial(write_image, frame_queue), daemon=True).start()

        recorder.record_images(num_pics=num_pics)
        frame_queue.join()
        # release the writer instance to finalize writing to disk
        writer.release()


if __name__ == "__main__":
    main()