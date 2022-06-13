import functools
import queue
import threading
import time
import torch

from vimba import *
import cv2
import vimba

status = True


class Recordor:
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        self._cam = cam
        self._queue = frame_queue
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        global st
        # Place the image data as an opencv image and the frame ID into the queue
        if frame.get_status() == vimba.FrameStatus.Complete:
            self._queue.put((frame.as_numpy_ndarray(), frame.get_id()))

        ENTER_KEY_CODE = 13

        key = cv2.waitKey(1)
        if key == ENTER_KEY_CODE:
            self.shutdown_event.set()
            status = False
            return

        elif frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)

            msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
            cv2.imshow(msg.format(cam.get_name()), frame.as_opencv_image())
        # Hand used frame back to Vimba, so it can store the next image in this memory
        cam.queue_frame(frame)

    def _setup_software_triggering(self):
        # Always set the selector first so that following features are applied correctly!
        self._cam.TriggerSelector.set('FrameStart')

        # optional in this example but good practice as it might be needed for hardware triggering
        self._cam.TriggerActivation.set('RisingEdge')

        # Make camera listen to Software trigger
        self._cam.TriggerSource.set('Software')
        self._cam.TriggerMode.set('Off')

    def _setup_camera(self):
        # setting the exposure (affects acquisition frame rate)
        self._cam.ExposureAuto.set('Off')
        self._cam.ExposureMode.set('Timed')
        self._cam.ExposureTime.set('50448.000')

        # setting the color
        self._cam.BalanceWhiteAuto.set('Continuous')
        self._cam.Hue.set('0.00')
        self._cam.Saturation.set('1.00')
        self._cam.Gamma.set('0.70')

        # setting image resolution (affects acquisition frame rate)
        # range of height (8-2056), range of width (8-2464)
        self._cam.Height.set('2056')
        self._cam.Width.set('2464')

        # setting pixel format
        self._cam.set_pixel_format(PixelFormat.Bgr8)

        # setting bandwidth of the data that will be streaming, range from (16250000-450000000)
        # (affects acquisition frame rate)
        self._cam.DeviceLinkThroughputLimit.set('450000000')

    def record_images(self):
        global st
        # This method assumes software trigger is desired. Free run image acquisition would work
        # similarly to get higher fps from the camera

        with vimba.Vimba.get_instance():
            with self._cam:
                self._setup_camera()
                self._setup_software_triggering()
                self._cam.start_streaming(handler=self, buffer_count=10)
                self.shutdown_event.wait()
                while st:
                    self._cam.TriggerSoftware.run()
                    # Delay between images can be adjusted or removed entirely
                    time.sleep(8)
                    if not status:
                        break
                self._cam.stop_streaming()

def write_image(frame_queue: queue.Queue):
    while True:
        # Get an element from the queue.
        frame, id = frame_queue.get()
        cv2.imwrite(f'/Users/admin/Desktop/pic/image_{id}.jpg', frame)
        # let the queue know we are finished with this element so the main thread can figure out
        # when the queue is finished completely
        frame_queue.task_done()


def main():
    #num_pics = 10
    with vimba.Vimba.get_instance() as vmb:
        cams = vmb.get_all_cameras()

        frame_queue = queue.Queue()
        recorder = Recordor(cam=cams[0], frame_queue=frame_queue)
        # Start a thread that runs write_image(frame_queue). Marking it as daemon allows the python
        # program to exit even though that thread is still running. The thread will then be stopped
        # when the program exits
        threading.Thread(target=functools.partial(write_image, frame_queue), daemon=True).start()

        recorder.record_images()
        frame_queue.join()


if __name__ == "__main__":
    main()
