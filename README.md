# opencv-bar-tracker

The ideal aim is automatically detect the barbell and annotate the video with the bar's path - this is useful for checking form in olympic weightlifting and powerlifting (though this is mostly an exercise in learning OpenCV)

High-level steps required:
- Draw a bounding box on the barbell 
- Track the centroid of the bounding box smoothly (Kalman filter to ensure smoothness?)
- Return an annotated video

(Ideally once trained this could be incorporated into a website for general use)
