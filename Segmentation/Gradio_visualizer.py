import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
import numpy as np
import time
import os

from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels, GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
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

from argparse import ArgumentParser
import sys
#
#
# predictor = None
# model = None
# cam_algorithm = None
# cfg = None
# result = {"seg" : None,"attenction" : None}
#
#
# def inital():
#     #Grad cam init
#     global model
#     model = models.resnet50()
#     ckpt_path = "/data1/solo/downstream/object_detection/pretrain_weight/byol+loss_f-lr0.3-beta_cosine_V2_0.9-resnet50-imagenet-1000ep-mask-cropping0.3-mask-pooling-1tykkdcy-ep=999.ckpt"
#     state = torch.load(ckpt_path, map_location={'cuda:7': 'cuda:0'})["state_dict"]
#     for k in list(state.keys()):
#         if "encoder" in k:
#             raise Exception(
#                 "You are using an older checkpoint."
#                 "Either use a new one, or convert it by replacing"
#                 "all 'encoder' occurances in state_dict with 'backbone'"
#             )
#         if "backbone" in k:
#             state[k.replace("backbone.", "")] = state[k]
#         del state[k]
#     model.load_state_dict(state, strict=False)
#     global cam_algorithm
#     cam_algorithm = GradCAM
#
#     # seg cam init
#     def setup():
#         """
#         Create configs and perform basic setups.
#         """
#         args = default_argument_parser().parse_args()
#         args.config_file = "configs/coco_R_50_C4_2x.yaml"
#         cfg = get_cfg()
#         cfg.merge_from_file(args.config_file)
#         cfg.merge_from_list(args.opts)
#         cfg.freeze()
#         default_setup(cfg, args)
#         return cfg
#
#     global cfg
#     cfg = setup()
#     global predictor
#     predictor = DefaultPredictor(cfg)
#     return predictor, cam_algorithm
#
# def grad_cam_attenction(image):
#     s = time.time()
#     global model
#     global cam_algorithm
#     target_layers = [model.layer4]
#     targets = None
#     rgb_img = np.float32(image) / 255
#     input_tensor = preprocess_image(rgb_img,
#                                     mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#     with cam_algorithm(model=model,
#                        target_layers=target_layers,
#                        use_cuda=True) as cam:
#         # AblationCAM and ScoreCAM have batched implementations.
#         # You can override the internal batch size for faster computation.
#         cam.batch_size = 32
#         grayscale_cam = cam(input_tensor=input_tensor,
#                             targets=targets)
#                             # aug_smooth=args.aug_smooth,
#                             # eigen_smooth=args.eigen_smooth)
#
#         # Here grayscale_cam has only one image in the batch
#         grayscale_cam = grayscale_cam[0, :]
#
#         cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#
#         # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
#         #cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
#
#     # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
#     # gb = gb_model(input_tensor, target_category=None)
#
#     # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
#     # cam_gb = deprocess_image(cam_mask * gb)
#     # gb = deprocess_image(gb)
#     print("cam_image time : ", time.time() - s)
#     result["attenction"] = cam_image
#     return cam_image
#
#
# def seg_func(image):
#     s = time.time()
#     global predictor
#     global cfg
#     outputs = predictor(image)
#     v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#     outputs = outputs["instances"].to("cpu")
#     print(type(outputs))
#     outputs_boxes = outputs.pred_boxes
#     outputs_scores = outputs.scores
#     outputs_mask = outputs.pred_masks
#
#     # print(outputs_boxes)
#     # print(outputs_scores)
#     # print(outputs_mask)
#     out = v.draw_instance_predictions(outputs)
#     # out = v.draw_box(outputs["instances"].pred_boxes.to("cpu"))
#     out_img = out.get_image()[:, :, ::-1]
#     print("seg_func time : ", time.time() - s)
#     result["seg"] = out_img
#     #return out_img
#
#
# def get_result(image):
#     seg_thread = threading.Thread(target=seg_func, name='seg_func', args=[image])
#     seg_thread.start()
#     attenction_thread = threading.Thread(target=grad_cam_attenction, name='grad_cam_attenction', args=[image])
#     attenction_thread.start()
#     seg_thread.join()
#     attenction_thread.join()
#     return result["seg"], result["attenction"]
#
#
# # gr.Interface(fn=seg_func, inputs="image", outputs="image", css=css_def ).launch(server_name="0.0.0.0",server_port=2022, share=True)
# # blk_cntr = gr.Blocks()
#
# css_format = '''
# #img_id {height: 80% !important;}
# #txt_id {width: 100% !important;}
# #btn_id {width: 20% !important; height: 15% !important;}
# #sld_id {width: 40% !important;}
# #heatmap_size {width: 40px;}
# '''
# inital()
#
# with gr.Blocks(css=css_format) as demo:
#     with gr.Column():
#         # Main markdown title
#         gr.Markdown(f"## *影像分割(Segmentation)*")
#
#         with gr.Row():
#             with gr.Column():
#                 gr.Markdown("### **輸入影像(Input Image)**")
#                 im_lab = "上傳影像請盡量小於300Kb (Upload image should not greater then 300Kb)"
#                 image_input = gr.Image(label=im_lab, elem_id="img_id")
#                 submit_btn = gr.Button("執行 (Run)", elem_id="btn_id")
#
#             with gr.Column():
#                 gr.Markdown("#### *選取範例 (Example gallery)* : 請任意選取一個 (pick up one image to run the example)")
#                 ims_uri = [f"examples_image/{x}" for x in os.listdir("examples_image")]
#                 ims_uri = [[ex, 3] for ex in ims_uri]
#                 examples = gr.Examples(examples=ims_uri, inputs=[image_input], fn=get_result)
#
#         with gr.Row():
#             with gr.Column():
#                 gr.Markdown("#### *分割及偵測結果*(**Result image**)")
#                 out1 = gr.Image(elem_id="img_id", show_label=False)
#             with gr.Column():
#                 gr.Markdown("#### *影像熱力圖*(**Heatmap**)")
#                 with gr.Row():
#                     out2 = gr.Image(elem_id="img_id", show_label=False)
#                     gr.Image(elem_id = "heatmap_size", show_label=False, value=os.path.join(os.path.dirname(__file__), "heatmap.png"))
#             # image_output = [gr.Image(show_label=False, elem_id="img_id") for _ in range(2)]
#
#     # act component & process request..
#     submit_btn.click(fn=get_result, inputs=[image_input], outputs=[out1,out2])
#
# demo.launch(server_name="0.0.0.0", server_port=2022)
_SMALL_OBJECT_AREA_THRESH = 1000
import random
def random_color(is_light = False):
    return ((random.randint(0 ,127) + int(is_light) * 128)/255,
            (random.randint(0,127) + int(is_light) * 128)/255,
            (random.randint(0,127) + int(is_light) * 128)/255)

class Visualizer_c(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, thresh = 60):
        super().__init__(img_rgb, metadata=metadata, scale=scale, instance_mode=instance_mode)
        self.thresh = thresh

    def overlay_instances(
            self,
            *,
            boxes=None,
            labels=None,
            masks=None,
            keypoints=None,
            assigned_colors=None,
            alpha=0.5,
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:
                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color() for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            if self.thresh <= float(labels[i].split(" ")[-1][:-1]):
                color = assigned_colors[i]
                if boxes is not None:
                    self.draw_box(boxes[i], edge_color=color)

                if masks is not None:
                    for segment in masks[i].polygons:
                        self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

                if labels is not None:
                    # first get a box
                    if boxes is not None:
                        x0, y0, x1, y1 = boxes[i]
                        text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                        horiz_align = "left"
                    elif masks is not None:
                        # skip small mask without polygon
                        if len(masks[i].polygons) == 0:
                            continue

                        x0, y0, x1, y1 = masks[i].bbox()

                        # draw text in the center (defined by median) when box is not drawn
                        # median is less sensitive to outliers.
                        text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                        horiz_align = "center"
                    else:
                        continue  # drawing the box confidence for keypoints isn't very useful.
                    # for small objects, draw text at the side to avoid occlusion
                    instance_area = (y1 - y0) * (x1 - x0)
                    if (
                            instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                            or y1 - y0 < 40 * self.output.scale
                    ):
                        if y1 >= self.output.height - 5:
                            text_pos = (x1, y0)
                        else:
                            text_pos = (x0, y1)

                    height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                    lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                    font_size = (
                            np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                            * 0.5
                            * self._default_font_size
                    )
                    self.draw_text(
                        labels[i],
                        text_pos,
                        color=lighter_color,
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                    )

class Segmentation_Demo:
    def __init__(self, ckpt_path ,segment = True):
        # Grad cam init(pretrain)
        self.model = models.resnet50()
        #ckpt_path = "/data1/solo/downstream/object_detection/pretrain_weight/byol+loss_f-lr0.3-beta_cosine_V2_0.9-resnet50-imagenet-1000ep-mask-cropping0.3-mask-pooling-1tykkdcy-ep=999.ckpt"
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
        self.model.load_state_dict(state, strict=False)
        #global cam_algorithm
        self.cam_algorithm = GradCAMPlusPlus

        # seg cam init
        if segment:
            def setup():
                """
                Create configs and perform basic setups.
                """
                args = default_argument_parser().parse_args()
                args.config_file = "/data1/solo/downstream/object_detection/configs/coco_R_50_C4_2x.yaml"
                cfg = get_cfg()
                cfg.merge_from_file(args.config_file)
                cfg.merge_from_list(args.opts)
                cfg.freeze()
                default_setup(cfg, args)
                return cfg

            #global cfg
            self.cfg = setup()
            #global predictor
            self.predictor = DefaultPredictor(self.cfg)
            self.result = {"seg": None, "attenction": None}
        else:
            self.result = {"attenction": None}


        # Grad cam init(fin)
        # self.model = build_model(self.cfg)
        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(self.cfg.MODEL.WEIGHTS)


        # return predictor, cam_algorithm

        #gradio setting

        # .gradio-container {
        #     background-image:url("file/demo_background.png");
        #     background-repeat: no-repeat;
        #     background-attachment: fixed;
        #     background-position: left bottom;
        #     background-size: 100% 100%;
        #     }
        self.css_format = '''
        #img_id {height: 80% !important;}
        #img_id_out {width: 100% !important;}
        #txt_id {width: 100% !important;}
        #btn_id {width: 20% !important; height: 15% !important;}
        #sld_id {width: 40% !important;}
        #heatmap_size {width: 40px !important; height: 300px !important;}
        '''
        self.example_dir = "/data1/solo/downstream/object_detection/examples_image"
        self.segment = segment

    def grad_cam_attenction(self, image):
        s = time.time()
        target_layers = [self.model.layer4]
        targets = None
        rgb_img = np.float32(image) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        with self.cam_algorithm(model=self.model,
                           target_layers=target_layers,
                           use_cuda=True) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets)

            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        print("cam_image time : ", time.time() - s)
        self.result["attenction"] = cam_image
        return cam_image

    def seg_func(self, image, T):
        s = time.time()
        outputs = self.predictor(image)
        Metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        v = Visualizer_c(image[:, :, ::-1], Metadata, scale=1.2, thresh=T)
        v._default_font_size = 15.0
        outputs = outputs["instances"].to("cpu")
        out = v.draw_instance_predictions(outputs)
        out_img = out.get_image()[:, :, ::-1]
        print("seg_func time : ", time.time() - s)
        self.result["seg"] = out_img

    def get_result(self, image, T):
        if self.segment:
            seg_thread = threading.Thread(target=self.seg_func, name='seg_func', args=[image,T])
            seg_thread.start()
        attenction_thread = threading.Thread(target=self.grad_cam_attenction, name='grad_cam_attenction', args=[image])
        attenction_thread.start()
        if self.segment:
            seg_thread.join()
            attenction_thread.join()
        else:
            attenction_thread.join()
            self.result["seg"] = self.result["attenction"]

        return self.result["seg"], self.result["attenction"]

    def get_gradio_blocks(self):
        with gr.Blocks(css=self.css_format) as demo:
            with gr.Row():
                with gr.Column():
                    # Main markdown title
                    gr.Markdown("## *影像分割(Image Segmentation)*", elem_id="font_")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### **輸入影像(Input Image)**", elem_id="font_")
                            im_lab = "上傳影像請盡量小於300Kb (Upload image should not greater then 300Kb)"
                            image_input = gr.Image(label=im_lab, elem_id="img_id")
                            submit_btn = gr.Button("執行 (Run)", elem_id="btn_id")

                        with gr.Column():
                            gr.Markdown("#### *選取範例 (Example gallery)* : 請任意選取一個 (pick up one image to run the example)", elem_id="font_")
                            ims_uri = [f"{self.example_dir}/{x}" for x in os.listdir(self.example_dir)]
                            #breakpoint()
                            thresh = gr.Slider(10, 100, value=70, label="閥值(%)", step=1)
                            examples = gr.Examples(examples=ims_uri, inputs = image_input)


                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### *分割及偵測結果*(**Result image**)" if self.segment else "#### *AI模型關注熱力圖*(**Heatmap**)", elem_id="font_")
                            out1 = gr.Image(elem_id="img_id", show_label=False)

                        with gr.Column():
                            gr.Markdown("#### *AI模型關注熱力圖*(**Heatmap**)", elem_id="font_")
                            with gr.Row():
                                out2 = gr.Image(elem_id="img_id", show_label=False)
                                gr.Image(elem_id="heatmap_size", shape=(15, None), show_label=False,
                                         value=os.path.join(os.path.dirname(__file__), "heatmap.png"))


                        # image_output = [gr.Image(show_label=False, elem_id="img_id") for _ in range(2)]

            # act component & process request..
            submit_btn.click(fn=self.get_result, inputs=[image_input, thresh], outputs=[out1, out2])
        return demo

if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument("-p", "--port", help="port setting", type=int)
    # args = parser.parse_args()
    # port = args.port
    print("port :",sys.argv[1])
    poart = int(sys.argv[1])
    sys.argv = [sys.argv[0]]
    gr.close_all()
    segmentation_Demo = Segmentation_Demo("/data1/solo/downstream/object_detection/pretrain_weight/byol+loss_f-lr0.3-beta_cosine_V2_0.9-resnet50-imagenet-1000ep-mask-cropping0.3-mask-pooling-1tykkdcy-ep=999.ckpt",False)
    demo = segmentation_Demo.get_gradio_blocks()
    demo = demo.queue(concurrency_count=20)
    demo.launch(server_name="0.0.0.0", server_port=poart, share = True)

    # css_code = 'body{background-image:url("https://picsum.photos/seed/picsum/200/300");}'
    # gr.Interface(lambda x: x, "textbox", "textbox", css=css_code).launch(server_name="0.0.0.0", server_port=2022)

# for x in os.listdir("examples_image"):
#     im = cv2.imread(f"examples_image/{x}")
#     im = cv2.resize(im, (96,96), interpolation=cv2.INTER_AREA)
#     cv2.imwrite(f"examples_image_small/{x}",im)
