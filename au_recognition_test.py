import os
import cv2
import time
import torch
from argparse import ArgumentParser
from ibug.face_alignment import FANPredictor
from ibug.face_alignment.utils import plot_landmarks
from ibug.emotion_recognition import AUPredictor
from ibug.face_detection import RetinaFacePredictor, S3FDPredictor


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video path or webcam index (default=0)', default=0)
    parser.add_argument('--output', '-o', help='Output file path', default=None)
    parser.add_argument('--fourcc', '-f', help='FourCC of the output video (default=mp4v)',
                        type=str, default='mp4v')
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--no-display', '-n', help='No display if processing a video file',
                        action='store_true', default=False)

    parser.add_argument('--detection-threshold', '-dt', type=float, default=0.8,
                        help='Confidence threshold for face detection (default=0.8)')
    parser.add_argument('--detection-method', '-dm', default='retinaface',
                        help='Face detection method, can be either RatinaFace or S3FD (default=RatinaFace)')
    parser.add_argument('--detection-weights', '-dw', default=None,
                        help='Weights to be loaded for face detection, ' +
                             'can be either resnet50 or mobilenet0.25 when using RetinaFace')
    parser.add_argument('--detection-device', '-dd', default='cuda:0',
                        help='Device to be used for face detection (default=cuda:0)')

    parser.add_argument('--alignment-threshold', '-at', type=float, default=0.2,
                        help='Score threshold used when visualising detected landmarks (default=0.2)'),
    parser.add_argument('--alignment-method', '-am', default='fan',
                        help='Face alignment method, must be set to FAN')
    parser.add_argument('--alignment-weights', '-aw', default=None,
                        help='Weights to be loaded for face alignment, can be either 2DFAN2 or 2DFAN4')
    parser.add_argument('--alignment-device', '-ad', default='cuda:0',
                        help='Device to be used for face alignment (default=cuda:0)')

    parser.add_argument('--emotion-method', '-em', default='emonet',
                        help='Emotion recognition method, must be set to EmoNet')
    parser.add_argument('--emotion-weights', '-ew', default=None,
                        help='Weights to be loaded for emotion recognition, can be either EmNet248 or EmoNet245')
    parser.add_argument('--emotion-device', '-ed', default='cuda:0',
                        help='Device to be used for emotion recognition (default=cuda:0)')
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    vid = None
    out_vid = None
    has_window = False
    try:
        # Create the face detector
        args.detection_method = args.detection_method.lower()
        if args.detection_method == 'retinaface':
            face_detector = RetinaFacePredictor(threshold=args.detection_threshold, device=args.detection_device,
                                                model=(RetinaFacePredictor.get_model(args.detection_weights)
                                                       if args.detection_weights else None))
            print('Face detector created using RetinaFace.')
        elif args.detection_method == 's3fd':
            face_detector = S3FDPredictor(threshold=args.detection_threshold, device=args.detection_device,
                                          model=(S3FDPredictor.get_model(args.detection_weights)
                                                 if args.detection_weights else None))
            print('Face detector created using S3FD.')
        else:
            raise ValueError('detector-method must be set to either RetinaFace or S3FD')

        # Create the landmark detector
        args.alignment_method = args.alignment_method.lower()
        if args.alignment_method == 'fan':
            landmark_detector = FANPredictor(device=args.alignment_device,
                                             model=(FANPredictor.get_model(args.alignment_weights)
                                                    if args.alignment_weights else None))
            print('Landmark detector created using FAN.')
        else:
            raise ValueError('alignment-method must be set to FAN')

        # Create the emotion recogniser
        args.emotion_method = args.emotion_method.lower()
        if args.emotion_method == 'emonet':
            au_recogniser = AUPredictor(device=args.emotion_device,
                                            model=(AUPredictor.get_model(args.emotion_weights)
                                                        if args.emotion_weights else None))
            print('Emotion recogniser created using EmoNet.')
        else:
            raise ValueError('action-unit-method must be set to EmoNet')

        # Open the input video
        using_webcam = not os.path.exists(args.input)
        vid = cv2.VideoCapture(int(args.input) if using_webcam else args.input)
        assert vid.isOpened()
        if using_webcam:
            print(f'Webcam #{int(args.input)} opened.')
        else:
            print(f'Input video "{args.input}" opened.')

        # Open the output video (if a path is given)
        if args.output is not None:
            out_vid = cv2.VideoWriter(args.output, fps=vid.get(cv2.CAP_PROP_FPS),
                                      frameSize=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      fourcc=cv2.VideoWriter_fourcc(*args.fourcc))
            assert out_vid.isOpened()

        # Process the frames
        frame_number = 0
        if len(au_recogniser.config.au_labels) == 12:
            emotion_colours = ((0, 255, 0), (255, 0, 0), (0, 0, 255),
                               (0, 255, 255), (255, 255, 0), (255, 0, 255),
                               (0, 128, 255), (255, 0, 128), (128, 255, 0),
                               (192, 192, 192), (0, 255, 128), (128,0,255))
        else:
            raise ValueError('au number much be set to 12')
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        print('Processing started, press \'Q\' to quit.')
        while True:
            # Get a new frame
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Detect faces
                start_time = time.time()
                faces = face_detector(frame, rgb=False)
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Face alignment
                start_time = current_time
                landmarks, scores, fan_features = landmark_detector(frame, faces, rgb=False, return_features=True)
                current_time = time.time()
                elapsed_time2 = current_time - start_time

                # Emotion recognition
                start_time = current_time
                aus = au_recogniser(fan_features)
                current_time = time.time()
                elapsed_time3 = current_time - start_time

                # Textural output
                print(f'Frame #{frame_number} processed in {elapsed_time * 1000.0:.04f} + ' +
                      f'{elapsed_time2 * 1000.0:.04f} + {elapsed_time3 * 1000.0:.04f} ms: ' +
                      f'{len(faces)} faces analysed.')

                # Rendering
                for idx, (face, lm, sc) in enumerate(zip(faces, landmarks, scores)):
                    bbox = face[:4].astype(int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
                    plot_landmarks(frame, lm, sc, threshold=args.alignment_threshold)
                    if len(face) > 5:
                        plot_landmarks(frame, face[5:].reshape((-1, 2)), pts_radius=3)
                    emo = au_recogniser.config.au_labels[aus].title()

                    # I do not know how to visualize the results


                    # val, ar = emotions['valence'][idx], emotions['arousal'][idx]
                    # text_content = f'{emo} ({val: .01f}, {ar: .01f})'
                    # cv2.putText(frame, text_content, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_DUPLEX,
                    #             0.5, emotion_colours[emotions['emotion'][idx]], lineType=cv2.LINE_AA)

                # Write the frame to output video (if recording)
                if out_vid is not None:
                    out_vid.write(frame)

                # Display the frame
                if using_webcam or not args.no_display:
                    has_window = True
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1) % 2 ** 16
                    if key == ord('q') or key == ord('Q'):
                        print('\'Q\' pressed, we are done here.')
                        break
                frame_number += 1
    finally:
        if has_window:
            cv2.destroyAllWindows()
        if out_vid is not None:
            out_vid.release()
        if vid is not None:
            vid.release()
        print('All done.')


if __name__ == '__main__':
    main()
