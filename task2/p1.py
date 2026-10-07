import os
import cv2
import numpy as np
from PIL import Image

# ========== 0. 配置 ==========
img_path = "example1.jpg"          # 替换为你的图片路径
out_dir  = "p1_output"
os.makedirs(out_dir, exist_ok=True)

# ========== 1. 用 OpenCV 读取图片 ==========
img_bgr = cv2.imread(img_path)          # OpenCV 默认 BGR
if img_bgr is None:
    raise FileNotFoundError(f"无法读取图片：{img_path}")

print(f"图片尺寸 (高, 宽, 通道数): {img_bgr.shape}")

# ========== 2. 用 Pillow 打开并保存 RGB 原图 ==========
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
pil_img = Image.fromarray(img_rgb)
print(f"Pillow 读取成功: 模式={pil_img.mode}, 尺寸={pil_img.size}")
pil_img.save(os.path.join(out_dir, "original_rgb.png"))

# ========== 3. 用 OpenCV 分离三个通道 ==========
b, g, r = cv2.split(img_bgr)            # 顺序：B、G、R

cv2.imwrite(os.path.join(out_dir, "channel_blue.png"),  b)
cv2.imwrite(os.path.join(out_dir, "channel_green.png"), g)
cv2.imwrite(os.path.join(out_dir, "channel_red.png"),   r)

# ========== 4. 用 Pillow 分别打开并保存三个通道 ==========
# Pillow 单通道图统一用 "L" 模式（灰度）保存
Image.fromarray(b, mode="L").save(os.path.join(out_dir, "pil_blue.png"))
Image.fromarray(g, mode="L").save(os.path.join(out_dir, "pil_green.png"))
Image.fromarray(r, mode="L").save(os.path.join(out_dir, "pil_red.png"))

# ========== 5. 生成"彩色通道视图"（更直观） ==========
zeros = np.zeros_like(b)
red_view   = cv2.merge([zeros, zeros, r])   # 只保留 R
green_view = cv2.merge([zeros, g, zeros])   # 只保留 G
blue_view  = cv2.merge([b, zeros, zeros])   # 只保留 B

cv2.imwrite(os.path.join(out_dir, "view_red.png"),   red_view)
cv2.imwrite(os.path.join(out_dir, "view_green.png"), green_view)
cv2.imwrite(os.path.join(out_dir, "view_blue.png"),  blue_view)

print(f"全部结果已保存到 {out_dir}/ 目录")