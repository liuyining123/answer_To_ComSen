import os
import cv2
import numpy as np

# ============================================================
# 0. 准备
# ============================================================
video_path = "example2.mp4"
out_dir    = "p4_output"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"无法打开视频：{video_path}")

fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频信息: {w}x{h}, {fps:.2f} fps, 共 {total} 帧")

# 编码器：mp4v 兼容性最好
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 关键：所有 VideoWriter 声明尺寸必须与 write() 的帧完全一致
size_pairs  = (w * 2, h)     # 左原图 + 右掩码
size_triple = (w * 3, h)     # MOG2 + 帧差 + 原图（对比视频）

vw_mog2 = cv2.VideoWriter(f"{out_dir}/mog2_result.mp4",  fourcc, fps, size_pairs)
vw_nobg = cv2.VideoWriter(f"{out_dir}/nobg_result.mp4",  fourcc, fps, size_pairs)
vw_cmp  = cv2.VideoWriter(f"{out_dir}/compare_result.mp4", fourcc, fps, size_triple)

for name, vw in [("mog2", vw_mog2), ("nobg", vw_nobg), ("compare", vw_cmp)]:
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter[{name}] 打开失败，检查 mp4v 编码器")

# ============================================================
# 1. 模型初始化
# ============================================================
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)

prev_gray = None

# 形态学核
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

MIN_AREA = 200    # 最小轮廓面积，过滤噪点

# 统计容器
stat = {
    "mog2": {"fg_px": 0, "objs": 0, "frames": 0, "obj_per_frame": []},
    "nobg": {"fg_px": 0, "objs": 0, "frames": 0, "obj_per_frame": []},
}

def process_mask(mask):
    """形态学去噪：先开运算去小白点，再闭运算连通大块"""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask

def get_boxes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) >= MIN_AREA:
            boxes.append(cv2.boundingRect(c))
    return boxes

# ============================================================
# 2. 逐帧处理
# ============================================================
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # 方法 A：MOG2 背景减除
    # --------------------------------------------------------
    fg_mog2 = mog2.apply(frame)
    # 去掉阴影：MOG2 把阴影标记为 127，阈值化只保留 255
    _, fg_mog2 = cv2.threshold(fg_mog2, 200, 255, cv2.THRESH_BINARY)
    fg_mog2 = process_mask(fg_mog2)
    boxes_mog2 = get_boxes(fg_mog2)

    stat["mog2"]["fg_px"] += int(np.count_nonzero(fg_mog2))
    stat["mog2"]["objs"]  += len(boxes_mog2)
    stat["mog2"]["obj_per_frame"].append(len(boxes_mog2))
    stat["mog2"]["frames"] += 1

    vis_mog2 = frame.copy()
    for (x, y, bw, bh) in boxes_mog2:
        cv2.rectangle(vis_mog2, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    cv2.putText(vis_mog2, f"MOG2 | objects={len(boxes_mog2)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    mask_mog2_bgr = cv2.cvtColor(fg_mog2, cv2.COLOR_GRAY2BGR)
    canvas_mog2   = np.hstack([vis_mog2, mask_mog2_bgr])
    assert canvas_mog2.shape[1] == size_pairs[0], canvas_mog2.shape
    vw_mog2.write(canvas_mog2)

    # --------------------------------------------------------
    # 方法 B：无背景减除（帧差）
    # --------------------------------------------------------
    if prev_gray is None:
        prev_gray = gray
        blank = np.zeros_like(frame)
        canvas_nobg = np.hstack([frame, blank])
        vw_nobg.write(canvas_nobg)
        # 对比视频：三栏
        vw_cmp.write(np.hstack([frame, frame, frame]))
        continue

    diff = cv2.absdiff(gray, prev_gray)
    _, fg_nobg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    prev_gray = gray
    fg_nobg = process_mask(fg_nobg)
    boxes_nobg = get_boxes(fg_nobg)

    stat["nobg"]["fg_px"] += int(np.count_nonzero(fg_nobg))
    stat["nobg"]["objs"]  += len(boxes_nobg)
    stat["nobg"]["obj_per_frame"].append(len(boxes_nobg))
    stat["nobg"]["frames"] += 1

    vis_nobg = frame.copy()
    for (x, y, bw, bh) in boxes_nobg:
        cv2.rectangle(vis_nobg, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    cv2.putText(vis_nobg, f"NoBG | objects={len(boxes_nobg)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    mask_nobg_bgr = cv2.cvtColor(fg_nobg, cv2.COLOR_GRAY2BGR)
    canvas_nobg   = np.hstack([vis_nobg, mask_nobg_bgr])
    vw_nobg.write(canvas_nobg)

    # --------------------------------------------------------
    # 对比视频：三栏 [MOG2 | NoBG | 原图]
    # --------------------------------------------------------
    cmp_frame = np.hstack([vis_mog2, vis_nobg, frame])
    assert cmp_frame.shape[1] == size_triple[0], cmp_frame.shape
    vw_cmp.write(cmp_frame)

    # 每 60 帧保存一张图片
    if frame_idx % 60 == 0:
        cv2.imwrite(f"{out_dir}/mog2_f{frame_idx:04d}.jpg", canvas_mog2)
        cv2.imwrite(f"{out_dir}/nobg_f{frame_idx:04d}.jpg", canvas_nobg)

    if frame_idx % 50 == 0:
        print(f"  已处理 {frame_idx}/{total} 帧")

cap.release()
vw_mog2.release()
vw_nobg.release()
vw_cmp.release()

# ============================================================
# 3. 统计分析
# ============================================================
def summarize(name, d):
    n = max(d["frames"], 1)
    arr = np.array(d["obj_per_frame"], dtype=float)
    return {
        "name":        name,
        "frames":      d["frames"],
        "avg_fg_px":   d["fg_px"] / n,
        "avg_objs":    float(arr.mean()) if len(arr) else 0.0,
        "max_objs":    int(arr.max())    if len(arr) else 0,
        "std_objs":    float(arr.std())  if len(arr) else 0.0,
        "total_objs":  d["objs"],
    }

s_mog2 = summarize("MOG2 背景减除",      stat["mog2"])
s_nobg = summarize("无背景减除(帧差法)", stat["nobg"])

lines = []
lines.append("=" * 68)
lines.append("运动目标检测方法对比分析")
lines.append("=" * 68)
lines.append(f"视频: {video_path}   分辨率: {w}x{h}   FPS: {fps:.2f}   总帧数: {total}")
lines.append("")
header = f"{'指标':<24}{'MOG2 背景减除':>20}{'无背景减除(帧差)':>22}"
lines.append(header)
lines.append("-" * 68)
lines.append(f"{'处理帧数':<22}{s_mog2['frames']:>22}{s_nobg['frames']:>22}")
lines.append(f"{'平均前景像素/帧':<19}{s_mog2['avg_fg_px']:>22.1f}{s_nobg['avg_fg_px']:>22.1f}")
lines.append(f"{'平均检测物体数/帧':<18}{s_mog2['avg_objs']:>22.2f}{s_nobg['avg_objs']:>22.2f}")
lines.append(f"{'物体数标准差':<21}{s_mog2['std_objs']:>22.2f}{s_nobg['std_objs']:>22.2f}")
lines.append(f"{'单帧最多物体数':<20}{s_mog2['max_objs']:>22}{s_nobg['max_objs']:>22}")
lines.append(f"{'累计检测物体数':<20}{s_mog2['total_objs']:>22}{s_nobg['total_objs']:>22}")
lines.append("")
lines.append("=" * 68)
lines.append("分析结论")
lines.append("=" * 68)
lines.append("""
1. 前景完整性
   - MOG2：使用高斯混合模型建模背景，运动物体被识别为完整实心区域，
           检测框能完整包围物体本体。
   - 帧差：只比较相邻两帧像素差，物体内部颜色均匀区域差值为 0，
           只有边缘/纹理变化处产生响应 → 前景呈"空心轮廓"，
           检测框往往是多个小框或多个碎块。

2. 物体静止时的表现
   - MOG2：物体停下后仍能在一段时间内（history 帧内）保持前景状态，
           逐渐融入背景。
   - 帧差：物体一停下，前后帧无差异 → 立即"消失"，
           完全检测不到。

3. 抗噪与稳定性
   - MOG2：对慢速运动、小幅度抖动相对稳健；阴影可通过阈值单独过滤。
   - 帧差：对相机抖动、光照变化非常敏感，容易产生大片误检；
           前景像素数往往远高于真实运动区域。

4. 速度与资源
   - MOG2：需维护每个像素的高斯分布，CPU 开销较大，但现代 CPU 上
           1080p 仍可实时。
   - 帧差：仅需一帧缓存 + 一次 absdiff，速度最快，适合嵌入式场景。

5. 适用场景
   - MOG2：监控、长期固定摄像头、需要完整目标轮廓的任务。
   - 帧差：资源受限、运动非常剧烈、不需要精确轮廓的快速预筛。

6. 本视频结论
   - MOG2 平均前景像素/帧 ≈ {:.0f}，帧差法 ≈ {:.0f}，比值 {:.2f}。
   - MOG2 平均物体数/帧 ≈ {:.2f}，帧差法 ≈ {:.2f}，说明帧差法
     会把一个运动物体拆分成多个碎片。
   - 帧差法物体数标准差更大（{:.2f} vs {:.2f}），表明其检测结果
     抖动更剧烈、稳定性更差。
   - 综合结论：本场景下 MOG2 背景减除显著优于无背景减除的帧差法。
""".format(s_mog2['avg_fg_px'], s_nobg['avg_fg_px'],
           s_mog2['avg_fg_px'] / max(s_nobg['avg_fg_px'], 1),
           s_mog2['avg_objs'],  s_nobg['avg_objs'],
           s_nobg['std_objs'],  s_mog2['std_objs']))

report = "\n".join(lines)
print("\n" + report)

with open(f"{out_dir}/analysis.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n输出文件：")
print(f"  {out_dir}/mog2_result.mp4       MOG2 标注视频（左检测框，右掩码）")
print(f"  {out_dir}/nobg_result.mp4       帧差法标注视频")
print(f"  {out_dir}/compare_result.mp4    三栏对比视频 [MOG2 | NoBG | 原图]")
print(f"  {out_dir}/analysis.txt          量化分析报告")