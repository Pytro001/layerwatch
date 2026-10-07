import numpy as np

from layerwatch import part


def test_solid_matches_known_points():
    # wall of the nozzle near the exit, test bars, empty centre of the chamber
    r = part.r_in(0.05) + 0.0012  # inner skin, inside the channel ring
    assert part.solid(np.array([r]), np.array([0.0]), 0.05)[0]
    assert part.solid(np.array([part.BAR]), np.array([part.BAR]), 0.1)[0]
    assert not part.solid(np.array([0.0]), np.array([0.0]), 0.2)[0]
    assert not part.solid(np.array([0.0]), np.array([0.0]), part.HPIPE + 0.01)[0]
    # test bars stop at their height
    assert not part.solid(np.array([part.BAR]), np.array([part.BAR]), part.BAR_H + 0.01)[0]


def test_slice_has_36_cooling_channels():
    import cv2 as cv
    m = part.slice_mask(0.1, 1024)
    # the channels are holes in the wall ring: count inner contours inside the wall
    cnts, hier = cv.findContours(m, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    holes = [c for c, h in zip(cnts, np.asarray(hier).reshape(-1, 4)) if h[3] >= 0 and cv.contourArea(c) < 400]
    assert len(holes) == 36


def test_slice_mask_is_symmetric():
    m = part.slice_mask(0.25)
    assert np.mean(m == m[:, ::-1]) > 0.995
