"""Build the README animation from an annotated video, without AI generation."""
import argparse
from pathlib import Path
import cv2
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--output', default=str(Path(__file__).with_name('demo.gif')))
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            raise ValueError('Cannot read video')
        step = max(1, round(fps / 8))
        for i in range(round(fps * 6)):
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                frame = cv2.resize(frame, (640, round(frame.shape[0]*640/frame.shape[1])))
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise ValueError('Empty video')
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.output, save_all=True, append_images=frames[1:],
                       duration=round(1000*step/fps), loop=0, optimize=True)
    finally:
        cap.release()

if __name__ == '__main__':
    main()
