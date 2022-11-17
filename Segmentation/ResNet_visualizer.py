import numpy as np
import time
import os
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

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

import threading

import sys

class Feature_Visualizer:
    def __init__(self, weight_dic):
        # Grad cam init(pretrain)
        # if type(weight_dic) is list:
            #self.weight_mapping = {"MOCOV3-100epoch":0, "DINO-100epoch":1, "MVMA-100epoch":2, "HARL-1000epoch":3, "HARL-100epoch":4, "BYOL-100epoch":5, "BYOL-300epoch":6, "HARL-300epoch":7}
        self.weight_dic = weight_dic
        self.models = { name : self.load_weight(models.resnet50(), path) for name , path in self.weight_dic.items()}

        #self.models = [self.load_weight(models.resnet50(),ckpt_path[i]) for i in range(len(ckpt_path))]
        # self.model = self.models[0]
        # else:
        #     self.models = [self.load_weight(models.resnet50(),ckpt_path)]
        #     self.model = self.models[0]
        #ckpt_path = "/data1/solo/downstream/object_detection/pretrain_weight/byol+loss_f-lr0.3-beta_cosine_V2_0.9-resnet50-imagenet-1000ep-mask-cropping0.3-mask-pooling-1tykkdcy-ep=999.ckpt"


        #global cam_algorithm
        self.cam_algorithm = GradCAMPlusPlus

        self.result = {"weight1": None, "weight2": None}

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

    def load_weight(self,model,path):
        checkpoint = torch.load(path, map_location={'cuda:7': 'cuda:0'})

        if len(list(checkpoint.keys())) > 100:
            checkpoint = {"state_dict":checkpoint}

        key = "model" if "model" in checkpoint.keys() else "student" if "student" in checkpoint.keys() else "state_dict"

        state = checkpoint[key]
        state = {k.replace("module.encoder.model.", ""): v for k, v in state.items()}
        state = {k.replace("module.base_encoder.", ""): v for k, v in state.items()}
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        return model

    def grad_cam_attenction(self, image, model, weight_id):
        s = time.time()
        target_layers = [model.layer4]
        targets = None
        rgb_img = np.float32(image) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        with self.cam_algorithm(model = model,
                           target_layers=target_layers,
                           use_cuda=True) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets)

            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        print("cam_image time : ", time.time() - s)
        self.result[weight_id] = cam_image
        return cam_image

    def get_result(self, image, weight1, weight2):
        attenction_thread1 = threading.Thread(target=self.grad_cam_attenction, name='grad_cam_attenction', args=[image, self.models[weight1], "weight1"])
        attenction_thread1.start()
        attenction_thread2 = threading.Thread(target=self.grad_cam_attenction, name='grad_cam_attenction', args=[image, self.models[weight2], "weight2"])
        attenction_thread2.start()
        attenction_thread1.join()
        attenction_thread2.join()
        return self.result["weight1"], self.result["weight2"]

    def get_gradio_blocks(self):
        with gr.Blocks(css=self.css_format) as demo:
            with gr.Row():
                with gr.Column():
                    # Main markdown title
                    gr.Markdown("## *模型可是化 (Model feature visualizer)*", elem_id="font_")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### **輸入影像(Input Image)**", elem_id="font_")
                            im_lab = "上傳影像請盡量小於300Kb (Upload image should not greater then 300Kb)"
                            image_input = gr.Image(label=im_lab, elem_id="img_id")

                        with gr.Column():
                            gr.Markdown("#### *選取範例 (Example gallery)* : 請任意選取一個 (pick up one image to run the example)", elem_id="font_")
                            ims_uri = [f"{self.example_dir}/{x}" for x in os.listdir(self.example_dir)]
                            #breakpoint()
                            #thresh = gr.Slider(10, 100, value=70, label="閥值(%)", step=1)
                            examples = gr.Examples(examples=ims_uri, inputs = image_input)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### *AI模型關注熱力圖*(**Heatmap**)", elem_id="font_")
                            weight1 = gr.Dropdown(list(self.weight_dic.keys()), label="選擇權重 (weight)")
                            out1 = gr.Image(elem_id="img_id", show_label=False)

                        with gr.Column():
                            gr.Markdown("#### *AI模型關注熱力圖*(**Heatmap**)", elem_id="font_")
                            weight2 = gr.Dropdown(list(self.weight_dic.keys()), label="選擇權重 (weight)")
                            # with gr.Row():
                            out2 = gr.Image(elem_id="img_id", show_label=False)

            # act component & process request..
            submit_btn = gr.Button("執行 (Run)", elem_id="btn_id")

            submit_btn.click(fn=self.get_result, inputs=[image_input, weight1, weight2], outputs=[out1, out2])
        return demo

if __name__ == '__main__':
    print("port :",sys.argv[1])
    poart = int(sys.argv[1])
    sys.argv = [sys.argv[0]]
    gr.close_all()
    #"/data1/solo/downstream/object_detection/pretrain_weight/byol+loss_f-lr0.3-beta_cosine_V2_0.9-resnet50-imagenet-1000ep-mask-cropping0.3-mask-pooling-1tykkdcy-ep=999.ckpt"
    feature_visualizer = Feature_Visualizer(
        {
            "MOCOV3-100epoch": "/data1/DINO/MoCoV3_ResNet-50-100ep.pth",
            "DINO-100epoch": "/data1/DINO/dino_rn50_official_ResNet_100epochs.pth",
            "MVMA-100epoch": "/data1/DINO/ResNet50-100ep.ckpt",
            "HARL-1000epoch": "/data1/solo/downstream/object_detection/pretrain_weight/byol+loss_f-lr0.3-beta_cosine_V2_0.9-resnet50-imagenet-1000ep-mask-cropping0.3-mask-pooling-1tykkdcy-ep=999.ckpt",
            "HARL-100epoch": "/data1/MPLCL_ckpt/mncrl/19c2fhvn/byol+loss_f-lr0.5-beta_cosine_V2_0.7-resnet50-imagenet-100ep-mask-cropping0.3-mask-pooling-DRFI-19c2fhvn-ep=99.ckpt",
            "BYOL-100epoch": "/data1/solo/downstream/object_detection/pretrain_weight/byol-imagenet-100ep-2x21pcjd-ep=99.ckpt",
            "BYOL-300epoch": "/data1/solo_ckpt/byol/byol-resnet50-imagenet-300ep-3vikusj6-ep.ckpt",
            "HARL-300epoch": "/data1/solo_ckpt/MNCRL-resnet50-imagenet-300ep-1f2tdknm-ep=290.ckpt"
         })
    demo = feature_visualizer.get_gradio_blocks()
    demo = demo.queue(concurrency_count=20)
    demo.launch(server_name="0.0.0.0", server_port=poart, share = True)