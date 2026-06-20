import time
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
from diffusion_policy.real_world.multi_realsense import MultiRealsense


class MultiCameraVisualizer:
    """
    Live multi-camera preview. OpenCV's Qt highgui requires imshow/waitKey on the
    main thread; a background Process or Thread triggers QObject::moveToThread spam.
    Call tick() from the main thread (e.g. each get_obs).
    """

    def __init__(
        self,
        realsense: MultiRealsense,
        row,
        col,
        window_name='Multi Cam Vis',
        vis_fps=60,
        fill_value=0,
        rgb_to_bgr=True,
    ):
        self.row = row
        self.col = col
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.fill_value = fill_value
        self.rgb_to_bgr = rgb_to_bgr
        self.realsense = realsense
        self._active = False
        self._last_tick = 0.0
        self._headless = False
        self._vis_data = None
        self._vis_img = None
        cv2.setNumThreads(1)
        threadpool_limits(1)

    def start(self, wait=False):
        self._active = True
        self._last_tick = 0.0

    def stop(self, wait=False):
        self._active = False
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def start_wait(self):
        pass

    def stop_wait(self):
        pass

    def tick(self):
        if not self._active or self._headless:
            return
        now = time.monotonic()
        if self._last_tick > 0 and (now - self._last_tick) < (1.0 / self.vis_fps):
            return
        self._last_tick = now

        channel_slice = slice(None)
        if self.rgb_to_bgr:
            channel_slice = slice(None, None, -1)

        try:
            self._vis_data = self.realsense.get_vis(out=self._vis_data)
            color = self._vis_data['color']
            N, H, W, C = color.shape
            assert C == 3
            oh = H * self.row
            ow = W * self.col
            if self._vis_img is None:
                self._vis_img = np.full(
                    (oh, ow, 3),
                    fill_value=self.fill_value,
                    dtype=np.uint8,
                )
            for row in range(self.row):
                for col in range(self.col):
                    idx = col + row * self.col
                    h_start = H * row
                    h_end = h_start + H
                    w_start = W * col
                    w_end = w_start + W
                    if idx < N:
                        self._vis_img[h_start:h_end, w_start:w_end] = (
                            color[idx, :, :, channel_slice]
                        )
            cv2.imshow(self.window_name, self._vis_img)
            cv2.waitKey(1)
        except cv2.error:
            print(
                "MultiCameraVisualizer: cv2.imshow not available "
                "(headless OpenCV?). Disabling visualization."
            )
            self._headless = True
