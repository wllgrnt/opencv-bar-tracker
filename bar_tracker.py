import os
import cv2
import sys
import numpy as np
import tensorflow as tf

# Import utilites from research/object_detection
# requires that this be on the PYTHONPATH
from utils import label_map_util
from utils import visualization_utils as vis_util
from imutils.video import FPS
from collections import deque

def getInitialBoundingBox(frame):
    """
    given an image, find the bounding box for the barbell plate using the
    trained RCNN.

    This is then tracked using OpenCV KCF.
    """
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME,
                                'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Get the actual frame to analyse
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    ymin, xmin, ymax, xmax = np.squeeze(boxes)[0]
    print(np.squeeze(boxes)[0])
    # Gotta multiply by image width and height to get pixels
    im_height, im_width, _ = frame.shape
    print(frame.shape)
    xmin = int(xmin * im_width)
    xmax = int(xmax * im_width)
    ymin = int(ymin * im_height)
    ymax = int(ymax * im_height)
    return (xmin, xmax, ymin, ymax)


if __name__ == '__main__':

    # Set up tracker.
    tracker_types = [
        'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE',
        'CSRT'
    ]
    tracker_functions = [
        cv2.TrackerBoosting_create, cv2.TrackerMIL_create,
        cv2.TrackerKCF_create, cv2.TrackerTLD_create,
        cv2.TrackerMedianFlow_create, cv2.TrackerGOTURN_create,
        cv2.TrackerMOSSE_create, cv2.TrackerCSRT_create
    ]

    tracker_dict = dict(zip(tracker_types, tracker_functions))
    tracker_type = "KCF"
    tracker = tracker_dict[tracker_type]()
    # Read video
    try:
        VIDEO_NAME = sys.argv[1]
    except IndexError:
        print("please supply a path to a video file")
        sys.exit()

    video = cv2.VideoCapture(VIDEO_NAME)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    im_height, im_width, _ = frame.shape
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Lets write the annotated file out as an mp4
    height, width, layers = frame.shape
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter("out.mp4", fourcc, fps, (width,height), isColor=True)

    frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # # Initialize tracker with first frame and bounding box
    initBB = None
    fps = None
    xmin, xmax, ymin, ymax = getInitialBoundingBox(frame)
    initBB = (xmin, ymin, xmax-xmin, ymax-ymin)
    tracker.init(frame, initBB)
    fps = FPS().start()
    points = deque(maxlen=frameCount)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        ok, bbox = tracker.update(frame)

        if ok:
            # Tracking success
            (x, y, w, h) = [int(v) for v in bbox]
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            path_color = (0,255,0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), path_color, 2)
            # Draw centroid
            center_point_x = int(x+ 0.5*w)
            center_point_y = int(y + 0.5*h)
            center = (center_point_x,center_point_y)
            cv2.circle(frame, center, 2, path_color, -1)
            points.appendleft(center)
            for i in range(1, len(points)):
                if points[i-1] is None or points[i] is None:
                    continue
                cv2.line(frame, points[i-1], points[i], path_color,2)


        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        fps.update()
        fps.stop()
        info = [
            ("Tracker", tracker_type),
            ("Success", "Yes" if ok else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, im_height - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        video_out.write(frame)
        key = cv2.waitKey(1) & 0xFF

    # Clean up
    video.release()
    cv2.destroyAllWindows()
