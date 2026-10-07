import os
import cv2
import numpy as np

# ========== 0. 准备 ==========
img_path = "example1.jpg"           # 换成你的图片路径
out_dir  = "p2_output"
os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(img_path)      # BGR
if img is None:
    raise FileNotFoundError(f"无法读取图片：{img_path}")

h, w = img.shape[:2]
print(f"原图尺寸 (高, 宽): ({h}, {w})")

# ========== 1. 缩放 (Resize) ==========
# 方式 A：按比例缩放
scale = 0.5
resized_ratio = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_LINEAR)

# 方式 B：按目标尺寸缩放（固定宽高）
resized_fixed = cv2.resize(img, (300, 200),   # 注意：是 (宽, 高)
                           interpolation=cv2.INTER_AREA)

# 方式 C：保持宽高比缩放到指定宽度
target_w = 400
ratio = target_w / w
resized_keep = cv2.resize(img, (target_w, int(h * ratio)),
                          interpolation=cv2.INTER_LINEAR)

cv2.imwrite(f"{out_dir}/01_resize_ratio.jpg", resized_ratio)
cv2.imwrite(f"{out_dir}/01_resize_fixed.jpg", resized_fixed)
cv2.imwrite(f"{out_dir}/01_resize_keep_ratio.jpg", resized_keep)
print("✔ 缩放完成")

# ========== 2. 旋转 (Rotate) ==========
# 方式 A：以图像中心为轴旋转任意角度（保持完整图像）
angle = 45
center = (w // 2, h // 2)
# 计算旋转后不裁剪图像所需的新尺寸
M = cv2.getRotationMatrix2D(center, angle, 1.0)
cos, sin = abs(M[0, 0]), abs(M[0, 1])
new_w = int(h * sin + w * cos)
new_h = int(h * cos + w * sin)
M[0, 2] += (new_w / 2) - center[0]
M[1, 2] += (new_h / 2) - center[1]
rotated_any = cv2.warpAffine(img, M, (new_w, new_h))

# 方式 B：90° / 180° / 270° 快速旋转
rot_90  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rot_180 = cv2.rotate(img, cv2.ROTATE_180)
rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imwrite(f"{out_dir}/02_rotate_45.jpg",       rotated_any)
cv2.imwrite(f"{out_dir}/02_rotate_90.jpg",       rot_90)
cv2.imwrite(f"{out_dir}/02_rotate_180.jpg",      rot_180)
cv2.imwrite(f"{out_dir}/02_rotate_270.jpg",      rot_270)
print("✔ 旋转完成")

# ========== 3. 平移 (Translate) ==========
# 平移矩阵：[[1, 0, tx], [0, 1, ty]]
tx, ty = 100, 50          # 右移 100 像素，下移 50 像素
M_translate = np.float32([[1, 0, tx],
                          [0, 1, ty]])
translated = cv2.warpAffine(img, M_translate, (w, h))

# 平移时用边界填充（避免黑边）
translated_border = cv2.warpAffine(img, M_translate, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)

cv2.imwrite(f"{out_dir}/03_translate_black.jpg",  translated)
cv2.imwrite(f"{out_dir}/03_translate_border.jpg", translated_border)
print("✔ 平移完成")

# ========== 4. 翻转 (Flip) ==========
flip_h = cv2.flip(img, 1)    # 1  = 水平翻转（左右）
flip_v = cv2.flip(img, 0)    # 0  = 垂直翻转（上下）
flip_b = cv2.flip(img, -1)   # -1 = 水平 + 垂直（旋转 180°）

cv2.imwrite(f"{out_dir}/04_flip_horizontal.jpg", flip_h)
cv2.imwrite(f"{out_dir}/04_flip_vertical.jpg",   flip_v)
cv2.imwrite(f"{out_dir}/04_flip_both.jpg",       flip_b)
print("✔ 翻转完成")

print(f"\n全部结果已保存到 {out_dir}/")