import os
import cv2
import numpy as np

# ========== 0. 准备 ==========
img_path = "example1.jpg"
out_dir  = "p3_output"
os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"无法读取图片：{img_path}")

print(f"原图尺寸 (高, 宽, 通道): {img.shape}")

# ========== 1. 转灰度 ==========
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"{out_dir}/01_gray.jpg", gray)

# ========== 2. 二值化 ==========
# 方式 A：全局阈值（大津法，自动选阈值）
_, binary_otsu = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 方式 B：自适应阈值（适合光照不均）
binary_adapt = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

cv2.imwrite(f"{out_dir}/02_binary_otsu.jpg",  binary_otsu)
cv2.imwrite(f"{out_dir}/02_binary_adapt.jpg", binary_adapt)

# ========== 3. 去噪（形态学开运算，去掉小噪点） ==========
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imwrite(f"{out_dir}/03_binary_clean.jpg", binary)

# ========== 4. 检测轮廓 ==========
# RETR_EXTERNAL：只取最外层轮廓
# RETR_LIST    ：取所有轮廓（无层级）
# RETR_TREE    ：取所有轮廓并建立树状层级
# CHAIN_APPROX_SIMPLE：只保留拐点，节省内存
contours, hierarchy = cv2.findContours(binary,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

print(f"\n检测到轮廓数量: {len(contours)}")

# ========== 5. 保存所有轮廓（含小的/噪点） ==========
img_all = img.copy()
cv2.drawContours(img_all, contours, -1, (0, 0, 255), 2)   # 红色，线宽 2
cv2.imwrite(f"{out_dir}/04_contours_all.jpg", img_all)

# ========== 6. 过滤掉太小的轮廓，只保留有效轮廓 ==========
min_area = 100          # 面积阈值，按实际图片调整
valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
print(f"过滤后有效轮廓数量: {len(valid_contours)}")

img_valid = img.copy()
cv2.drawContours(img_valid, valid_contours, -1, (0, 255, 0), 2)   # 绿色
cv2.imwrite(f"{out_dir}/05_contours_valid.jpg", img_valid)

# ========== 7. 逐个轮廓统计面积、周长 ==========
total_area      = 0.0
total_perimeter = 0.0

print("\n===== 各轮廓统计 =====")
print(f"{'序号':<6}{'面积(px²)':<16}{'周长(px)':<16}{'外接矩形(x,y,w,h)'}")
for i, c in enumerate(valid_contours):
    area = cv2.contourArea(c)              # 面积
    peri = cv2.arcLength(c, True)          # 周长，True 表示闭合轮廓
    x, y, w, h = cv2.boundingRect(c)       # 外接矩形

    total_area      += area
    total_perimeter += peri

    print(f"{i:<6}{area:<16.2f}{peri:<16.2f}({x}, {y}, {w}, {h})")

print("\n===== 汇总 =====")
print(f"轮廓总数        : {len(valid_contours)}")
print(f"轮廓总面积      : {total_area:.2f} 像素²")
print(f"轮廓总周长      : {total_perimeter:.2f} 像素")

# ========== 8. 标注每个轮廓的编号 + 面积 ==========
img_label = img.copy()
for i, c in enumerate(valid_contours):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    # 画外接矩形
    cv2.rectangle(img_label, (x, y), (x + w, y + h), (255, 0, 0), 1)
    # 写编号和面积
    text = f"#{i}: {int(area)}"
    cv2.putText(img_label, text, (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite(f"{out_dir}/06_contours_labeled.jpg", img_label)

# ========== 9. 额外：计算最大轮廓的外接圆、拟合椭圆 ==========
if valid_contours:
    largest = max(valid_contours, key=cv2.contourArea)
    img_largest = img.copy()

    # 最小外接矩形（带旋转）
    rect = cv2.minAreaRect(largest)
    box  = cv2.boxPoints(rect)
    box  = np.intp(box)
    cv2.drawContours(img_largest, [box], 0, (0, 0, 255), 2)

    # 最小外接圆
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cv2.circle(img_largest, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)

    # 多边形拟合（轮廓近似）
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)
    cv2.drawContours(img_largest, [approx], 0, (255, 0, 0), 2)

    cv2.imwrite(f"{out_dir}/07_largest_analysis.jpg", img_largest)

    print(f"\n最大轮廓面积    : {cv2.contourArea(largest):.2f} 像素²")
    print(f"最大轮廓周长    : {cv2.arcLength(largest, True):.2f} 像素")
    print(f"最小外接圆      : 圆心=({cx:.0f},{cy:.0f}), 半径={radius:.1f}")
    print(f"多边形拟合顶点数: {len(approx)}")

print(f"\n全部结果已保存到 {out_dir}/")