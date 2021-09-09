# emotion and action_unit recognition
** Code is partly forked/copied from the official code of [emonet](https://github.com/face-analysis/emonet)**
## Prerequisites
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [OpenCV](https://opencv.org/) (only needed by the test script): `$pip3 install opencv-python`
* [ibug.face_detection](https://github.com/hhj1897/face_detection) (only needed by the test script). See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).
* [ibug.face_alignment](https://github.com/hhj1897/face_alignment). See this repository for details: [https://github.com/hhj1897/face_alignment](https://github.com/hhj1897/face_alignment)

## How to Test
* To test on live video: `python emotion_recognition_test.py [-i webcam_index]`
* To test on a video file: `python emotion_recognition_test.py [-i input_file] [-o output_file]`
* 
## References
\[1\] Toisoul, Antoine, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, and Maja Pantic. "[Estimation of continuous valence and arousal levels from faces in naturalistic conditions.](https://rdcu.be/cdnWi)" _Nature Machine Intelligence_ 3, no. 1 (2021): 42-50.
