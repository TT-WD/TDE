import json
import os
import sys
from contextlib import contextmanager

import cv2
import numpy as np


@contextmanager
def change_dir(path):
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

# 模型全局变量
hrnet_model = None
zoe_model = None
hrnet_inference_one_img= None
zoe_inference_one_img= None
def init_models():
    global hrnet_model, zoe_model, hrnet_inference_one_img, zoe_inference_one_img

    # 加载 HRNet 模型
    hrnet_path = os.path.join(os.path.dirname(__file__), "CBAM-HRNet-Detection")
    sys.path.insert(0, hrnet_path)
    with change_dir(hrnet_path):
        from tools.inference_img import init_config, get_point_model
        from tools.inference_img import inference_one_img
        hrnet_inference_one_img= inference_one_img
        init_config()
        hrnet_model = get_point_model("./pth_dir/model_best.pth")

    # 加载 ZoeDepth 模型
    zoe_path = os.path.join(os.path.dirname(__file__), "zoe-tooth-distance")
    sys.path.insert(0, zoe_path)
    with change_dir(zoe_path):
        from inference_img import get_dis_model
        from inference_img import inference_one_img
        zoe_inference_one_img= inference_one_img
        zoe_model = get_dis_model()



def run_hrnet(image:np.ndarray):
    """调用 HRNet 检测关键点"""
    input_img=image.copy()
    input_img=cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    #with change_dir(os.path.join(os.path.dirname(__file__), "HRNet-Facial-Landmark-Detection")):
    global hrnet_inference_one_img
    keypoints = hrnet_inference_one_img(hrnet_model, input_img)
    return keypoints

def run_zoe(image:np.ndarray, point_loca:np.ndarray):
    """调用 ZoeDepth 计算距离并画示意图"""
    global zoe_inference_one_img
    input_img = image.copy()
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    tooth_distance = zoe_inference_one_img(zoe_model, input_img, point_loca)
    return tooth_distance

def draw_keypoints(image_bgr, keypoints):
    img = image_bgr.copy()
    h, w = img.shape[:2]

    # 动态参数
    scale = max(h, w) / 512
    radius = int(3 * scale)
    thickness = int(2 * scale)

    for i, pt in enumerate(keypoints):
        x, y = map(int, pt)
        cv2.circle(img, (x, y), radius, (0, 255, 0), -1)
        if i % 2 == 1:  # 每两个点连线
            x1, y1 = map(int, keypoints[i - 1])
            cv2.line(img, (x1, y1), (x, y), (0, 0, 255), thickness)
    return img

def draw_distances(image_bgr, keypoints, distances,font_color):
    img = image_bgr.copy()
    h, w = img.shape[:2]
    font_scale = max(h, w) / 1500
    font_thickness = int(1.5 * font_scale)
    if keypoints.shape[1] == 4:
        keypoints=keypoints.reshape(-1,2)
    for i in range(0, len(keypoints), 2):
        pt1 = tuple(map(int, keypoints[i]))
        pt2 = tuple(map(int, keypoints[i + 1]))
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        text = f"{distances[i//2]:.2f}mm"
        cv2.putText(img, text, mid, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
    return img

# ================= Gradio 主逻辑 =================
import gradio as gr
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Tooth landmark detection & Distance Estimate")

        with gr.Row():
            input_img = gr.Image(label="upload_img", type="numpy")
            keypoint_img = gr.Image(label="landmark detection", type="numpy")
            distance_img = gr.Image(label="distance estimate", type="numpy")

        btn_detect = gr.Button("1️⃣ Detect landmark")
        btn_distance = gr.Button("2️⃣ Estimate distance")

        # 用于存储关键点数据
        keypoints_state = gr.State()

        # 第一步：检测关键点
        def step1(image):
            input_img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            keypoints = run_hrnet(input_img)
            img_with_points = draw_keypoints(input_img, keypoints)
            img_with_points = cv2.cvtColor(img_with_points, cv2.COLOR_BGR2RGB)
            keypoints_json=json.dumps(keypoints.tolist())
            return img_with_points, keypoints_json

        btn_detect.click(step1, inputs=input_img, outputs=[keypoint_img,keypoints_state])

        # 第二步：计算距离
        def step2(origin_image, keypoints_json):
            if keypoints_json is None:
                return None
            keypoints = np.array(json.loads(keypoints_json),dtype=np.int32)
            keypoints=keypoints.reshape(-1, 4)
            input_img = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
            distances = run_zoe(input_img,keypoints)
            img_with_point= draw_keypoints(input_img,keypoints.reshape(-1,2))
            img_with_dist = draw_distances(img_with_point, keypoints, distances,font_color=(0,0,0))
            img_with_dist = cv2.cvtColor(img_with_dist, cv2.COLOR_BGR2RGB)
            return img_with_dist

        btn_distance.click(step2, inputs=[input_img,keypoints_state], outputs=distance_img)

    demo.launch()#share=True)

if __name__ == '__main__':
    init_models()
    main()