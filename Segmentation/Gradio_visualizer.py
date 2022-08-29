import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
import numpy as np
import time
import os

from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import gradio as gr
from torchvision import models
import torch
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

import threading

predictor = None
model = None
cam_algorithm = None
cfg = None
result = {"seg" : None,"attenction" : None}
def inital():
    #Grad cam init
    global model
    model = models.resnet50()
    ckpt_path = "/data1/solo/downstream/object_detection/pretrain_weight/byol+loss_f-lr0.3-beta_cosine_V2_0.9-resnet50-imagenet-1000ep-mask-cropping0.3-mask-pooling-1tykkdcy-ep=999.ckpt"
    state = torch.load(ckpt_path, map_location={'cuda:7': 'cuda:0'})["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            raise Exception(
                "You are using an older checkpoint."
                "Either use a new one, or convert it by replacing"
                "all 'encoder' occurances in state_dict with 'backbone'"
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    model.load_state_dict(state, strict=False)
    global cam_algorithm
    cam_algorithm = GradCAM

    # seg cam init
    def setup():
        """
        Create configs and perform basic setups.
        """
        args = default_argument_parser().parse_args()
        args.config_file = "configs/coco_R_50_C4_2x.yaml"
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        return cfg

    global cfg
    cfg = setup()
    global predictor
    predictor = DefaultPredictor(cfg)
    return predictor, cam_algorithm

def grad_cam_attenction(image):
    s = time.time()
    global model
    global cam_algorithm
    target_layers = [model.layer4]
    targets = None
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=True) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)
                            # aug_smooth=args.aug_smooth,
                            # eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        #cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    # gb = gb_model(input_tensor, target_category=None)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)
    print("cam_image time : ", time.time() - s)
    result["attenction"] = cam_image
    return cam_image


def seg_func(image):
    s = time.time()
    global predictor
    global cfg
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    outputs = outputs["instances"].to("cpu")
    print(type(outputs))
    outputs_boxes = outputs.pred_boxes
    outputs_scores = outputs.scores
    outputs_mask = outputs.pred_masks

    # print(outputs_boxes)
    # print(outputs_scores)
    # print(outputs_mask)
    out = v.draw_instance_predictions(outputs)
    # out = v.draw_box(outputs["instances"].pred_boxes.to("cpu"))
    out_img = out.get_image()[:, :, ::-1]
    print("seg_func time : ", time.time() - s)
    result["seg"] = out_img
    #return out_img


def get_result(image):
    seg_thread = threading.Thread(target=seg_func, name='seg_func', args=[image])
    seg_thread.start()
    attenction_thread = threading.Thread(target=grad_cam_attenction, name='grad_cam_attenction', args=[image])
    attenction_thread.start()
    seg_thread.join()
    attenction_thread.join()
    return result["seg"], result["attenction"]


# gr.Interface(fn=seg_func, inputs="image", outputs="image", css=css_def ).launch(server_name="0.0.0.0",server_port=2022, share=True)
# blk_cntr = gr.Blocks()

css_format = '''
#img_id {height: 80% !important;}
#txt_id {width: 100% !important;}
#btn_id {width: 20% !important; height: 15% !important;}
#sld_id {width: 40% !important;} 
#heatmap_size {width: 40px;}
'''
inital()

with gr.Blocks(css=css_format) as demo:
    with gr.Column():
        # Main markdown title
        gr.Markdown(f"## *影像分割(Segmentation)*")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### **輸入影像(Input Image)**")
                im_lab = "上傳影像請盡量小於300Kb (Upload image should not greater then 300Kb)"
                image_input = gr.Image(label=im_lab, elem_id="img_id")
                submit_btn = gr.Button("執行 (Run)", elem_id="btn_id")

            with gr.Column():
                gr.Markdown("#### *選取範例 (Example gallery)* : 請任意選取一個 (pick up one image to run the example)")
                ims_uri = [f"examples_image/{x}" for x in os.listdir("examples_image")]
                ims_uri = [[ex, 3] for ex in ims_uri]
                examples = gr.Examples(examples=ims_uri, inputs=[image_input], fn=get_result)

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### *分割及偵測結果*(**Result image**)")
                out1 = gr.Image(elem_id="img_id", show_label=False)
            with gr.Column():
                gr.Markdown("#### *影像熱力圖*(**Heatmap**)")
                with gr.Row():
                    out2 = gr.Image(elem_id="img_id", show_label=False)
                    gr.Image(elem_id = "heatmap_size", show_label=False, value=os.path.join(os.path.dirname(__file__), "heatmap.png"))
            # image_output = [gr.Image(show_label=False, elem_id="img_id") for _ in range(2)]

    # act component & process request..
    submit_btn.click(fn=get_result, inputs=[image_input], outputs=[out1,out2])

demo.launch(server_name="0.0.0.0",server_port=2022)

# for x in os.listdir("examples_image"):
#     im = cv2.imread(f"examples_image/{x}")
#     im = cv2.resize(im, (96,96), interpolation=cv2.INTER_AREA)
#     cv2.imwrite(f"examples_image_small/{x}",im)