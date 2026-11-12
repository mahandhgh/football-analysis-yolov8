import argparse
from pathlib import Path
import cv2
from config import Config
from tracking.tracker import Tracker
from team_assignment.team_assigner import TeamAssigner
from transformation.transformer import Transformer
from visualization.renderer import Renderer


def main():
    config = Config()
    parser = argparse.ArgumentParser(description='Football tracking and pitch projection')
    parser.add_argument('--input', default=config.INPUT_VIDEO_PATH)
    parser.add_argument('--output', default=config.OUTPUT_VIDEO_PATH)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--batch-size', type=int, default=25)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('Frame limits must be positive')
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error('Input and output must differ')
    for model in (config.DETECTOR_MODEL_PATH, config.POSE_MODEL_PATH):
        if not Path(model).is_file():
            parser.error(f'Missing model: {model}')
    cap = cv2.VideoCapture(str(args.input))
    writer = None
    count = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not 0 < fps < 1000:
            raise ValueError(f'Cannot read video: {args.input}')
        tracker = Tracker(config.DETECTOR_MODEL_PATH, config, fps=fps)
        transformer = Transformer(config.POSE_MODEL_PATH, config)
        assigner = TeamAssigner()
        renderer = Renderer(config)
        while args.max_frames is None or count < args.max_frames:
            frames = []
            size = args.batch_size if args.max_frames is None else min(args.batch_size, args.max_frames-count)
            for _ in range(size):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            tracks = tracker.trackobjects(frames)
            keypoints = transformer.detect_keypoint(frames)
            homographies = [transformer.compute_homography(k) for k in keypoints]
            assigner.assigncolors(frames, tracks['players'], transformer, homographies=homographies)
            annotated = renderer.draw(frames, tracks, assigner, transformer, keypoints, homographies=homographies)
            if writer is None:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Cannot write output')
            for frame in annotated:
                writer.write(frame)
            count += len(frames)
            print(f'Processed {count} frames', flush=True)
        if not count:
            raise ValueError('Video contains no readable frames')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print(f'Saved {count} frames at {fps:g} fps: {args.output}')

if __name__ == '__main__':
    main()
