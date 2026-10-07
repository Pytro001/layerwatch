import cv2 as cv


def test_opencv_major_version_is_5():
    assert cv.__version__.startswith("5"), cv.__version__


def test_modules_used_by_the_pipeline_exist():
    for name in ("findTransformECC", "warpPerspective", "HoughLinesP", "matchShapes", "adaptiveThreshold",
                 "createBackgroundSubtractorMOG2", "SimpleBlobDetector_create", "Laplacian"):
        assert hasattr(cv, name), name
    assert hasattr(cv.dnn, "readNetFromONNX")
