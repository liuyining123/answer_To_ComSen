"""题五：使用 OpenCV Tracker 追踪 example3/example4 中的小球。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2


BBox = Tuple[int, int, int, int]

def create_csrt_tracker():
    """兼容 OpenCV 主命名空间和 legacy 命名空间。"""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError(
        "当前 OpenCV 未提供 CSRT Tracker，请安装 opencv-contrib-python。"
    )


def parse_bbox(value: str) -> BBox:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("目标框格式应为 x,y,w,h")
    try:
        bbox = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("目标框必须全部为整数") from exc
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("目标框的宽和高必须大于 0")
    return x, y, w, h


def detect_bbox(frame) -> BBox:
    """利用颜色分割和轮廓圆度自动定位首帧中的小球。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 60, 40), (15, 255, 255))
    red |= cv2.inRange(hsv, (165, 60, 40), (179, 255, 255))

    blue_difference = frame[:, :, 0].astype("int16") - frame[:, :, 2].astype(
        "int16"
    )
    low_saturation_ball = (
        (blue_difference > 8) & (hsv[:, :, 1] < 130) & (hsv[:, :, 2] > 70)
    ).astype("uint8") * 255

    candidates = []
    for mask in (red, low_saturation_ball):
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4 * 3.141592653589793 * area / (perimeter * perimeter)
                if perimeter
                else 0
            )
            x, y, w, h = cv2.boundingRect(contour)
            ratio = min(w, h) / max(w, h)
            candidates.append((area, circularity * ratio, (x, y, w, h)))

    if not candidates:
        raise RuntimeError(
            "无法自动识别首帧小球，请使用 --bbox x,y,w,h 指定目标框。"
        )

    _, _, detected = max(candidates, key=lambda item: (item[0], item[1]))
    x, y, w, h = detected
    padding = max(4, int(max(w, h) * 0.08))
    return x - padding, y - padding, w + 2 * padding, h + 2 * padding


def select_bbox(frame, input_path: Path, bbox: Optional[BBox], interactive: bool) -> BBox:
    if bbox is not None:
        return bbox
    if not interactive:
        detected = detect_bbox(frame)
        print(f"{input_path.name}: 自动识别首帧目标框 {detected}")
        return detected
    window = f"Select ball in {input_path.name}; press ENTER"
    selected = cv2.selectROI(window, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window)
    x, y, w, h = [int(value) for value in selected]
    if w <= 0 or h <= 0:
        raise ValueError(f"{input_path.name} 未选择有效的小球区域")
    return x, y, w, h


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def track_video(
    input_path: Path,
    output_path: Path,
    bbox: Optional[BBox],
    display: bool,
    interactive: bool,
) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频：{input_path}")

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"无法读取视频首帧：{input_path}")

    height, width = frame.shape[:2]
    initial_bbox = clamp_bbox(
        select_bbox(frame, input_path, bbox, interactive), width, height
    )
    tracker = create_csrt_tracker()
    tracker.init(frame, initial_bbox)

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    frame_index = 0
    lost_frames = 0
    try:
        while True:
            if frame_index > 0:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

            success, tracked = tracker.update(frame)
            if success:
                current = clamp_bbox(tuple(int(v) for v in tracked), width, height)
                x, y, w, h = current
                center = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
                label = f"CSRT tracking | frame={frame_index}"
                color = (0, 255, 0)
            else:
                lost_frames += 1
                label = f"CSRT lost | frame={frame_index}"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                label,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            writer.write(frame)

            if display:
                cv2.imshow("Ball tracking (q: quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    print(
        f"{input_path.name}: {frame_index} frames, "
        f"lost={lost_frames}, output={output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        default=[Path("example3.mp4"), Path("example4.mp4")],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("p5_output"))
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        help="所有视频使用同一个目标框，格式为 x,y,w,h；不提供则逐个手动框选",
    )
    parser.add_argument("--interactive", action="store_true", help="改用鼠标框选")
    args = parser.parse_args()
    display = args.interactive
    if display and not os.environ.get("DISPLAY"):
        parser.error("--interactive 需要可用的图形显示环境")

    for video in args.videos:
        output = args.output_dir / f"{video.stem}_CSRT.mp4"
        track_video(video, output, args.bbox, display, args.interactive)


if __name__ == "__main__":
    main()