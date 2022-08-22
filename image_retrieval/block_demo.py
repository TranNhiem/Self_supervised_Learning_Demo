import clip
import os
import re
import cv2
import string
from PIL import Image
import numpy as np
import random
from collections import OrderedDict
import torch
import torchvision
import gradio as gr
from demo_script.inner_interface import select_backbone, retriv_by_phase, retriv_by_img
import pickle
from glob import glob
import time

__all__ = ["ImgRetrieval_By_Img", "ImgRetrieval_By_Txt"]

def msk_output(topk_im, k_val, msk_first=False):
    if msk_first:
        topk_im = topk_im[1:]

    fill_with_msk = topk_im[k_val:]
    topk_im[k_val:] = np.ones_like(fill_with_msk)
    return topk_im


class ImgRetrieval_By_Img:
    def __init__(self, model, preprocess, key_embs, clear_images):
        self.model = model
        self.preprocess = preprocess
        self.key_embs = key_embs
        self.clear_images = clear_images

    def img_retri_by_image(self, query_img=None, k_val=None):
        query_img = self.preprocess( Image.fromarray(query_img) )
        topk_im = retriv_by_img(query_img.unsqueeze(0), self.key_embs, self.clear_images, self.model, self.preprocess, top_k=6)

        topk_im = np.clip(topk_im, -1, 1)
        return msk_output(topk_im, k_val, msk_first=True)

    def pkg_demo_app(self, blk_cntr, bkn_tag, db_name):
            
        with blk_cntr:
            with gr.Column():
                # Main markdown title
                gr.Markdown(f"## **影像**對影像檢索 (Image retrival **By Image**) 🖼️")
                
                with gr.Row():
                    
                    with gr.Column():
                        gr.Markdown("### **查詢影像 (Query Image)**")
                        image_input = gr.Image(label="Upload image should not greater then 300Kb", elem_id="img_id")
                        submit_btn = gr.Button("Run", elem_id="btn_id")

                    with gr.Column():
                        gr.Markdown("#### **Example gallery** : pick up one image to run the example")
                        ims_uri = glob(f"/workspace/data/{db_name}_ex/*.jpg")
                        ims_uri = [ [ex, 3] for ex in ims_uri] 
                        examples = gr.Examples(examples=ims_uri, inputs=[image_input], fn=self.img_retri_by_image)

                        k_val = gr.Slider(0, 5, step=1, value=3, interactive=True, 
                                        label="🔝 Top k value", elem_id="sld_id")

                gr.Markdown("### **Top-k個搜尋結果** (**Top-k matched images**)")
                with gr.Row():
                    image_output = [ gr.Image(show_label=False, elem_id="img_id") for _ in range(5) ]
                    
            # act component & process request..
            submit_btn.click(self.img_retri_by_image, inputs=[image_input, k_val], outputs=image_output)



class ImgRetrieval_By_Txt:
    def __init__(self, model, preprocess, key_embs, clear_images):
        self.model = model
        self.preprocess = preprocess
        self.key_embs = key_embs
        self.clear_images = clear_images

    def img_retri_by_text(self, query_text=None, k_val=None):
        qurey_phases = query_text.split('.')[0] # only process first phase..
        topk_im = retriv_by_phase(qurey_phases, self.key_embs.cuda(), self.clear_images, self.model)

        topk_im = np.clip(topk_im, -1, 1)
        return msk_output(topk_im, k_val)

    def pkg_demo_app(self, blk_cntr, bkn_tag, db_name):
        lab_lst =['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        demo_doc = \
        '''
        ### Note : the dataset only contains the following kind of images : 
        ### **airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck**
        ##### Please confirm the content of **Query text** related with the above categories.
        ##### Otherwise, the model only return the similar images according to the query text.
        ---
        '''

        with blk_cntr:
            with gr.Column():
                # markdown title
                gr.Markdown(f"## **文本**對影像檢索 (Image retrival **By Text**) 📜")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(demo_doc.format(lab_lst))
                        gr.Markdown("---")
                        gr.Markdown("#### **查詢語句 (Query Text)**")
                        text_input = gr.Textbox(show_label=False, elem_id="txt_id")
                        # act component & process request..
                        submit_btn = gr.Button("執行 (Run)", elem_id="btn_id")
                        gr.Markdown("---")
                        gr.Markdown("<br><br><br><br><br>")
                        
                    with gr.Column():
                        gr.Markdown("#### **選取範例** (**Example gallery**) : pick up one image to run the example")
                        # show example
                        txt_exms = ['A blue bird stays on the top of the tree.', 'How to define "cute" !?', 
                                    'What kind of animal like to eat a banana?', 'Image something flying in the sky!!',
                                    'Image something blue that can fly!', 'This car is expensive.', 'How to define "fast" !?',
                                    'A tiny red monkey.', 'Could the monkey climb on the tree?', 'A red car is parked in front of a house.',
                                    'A cargo ship crosses the ocean.', 'Enjoy an equestrian performance.', 'A lazy dog lies on the ground.',
                                    'The Boeing 747.', 'A car is parked near the other one.']  
                        txt_exms = [ [ex, 3] for ex in txt_exms] 
                        examples = gr.Examples(examples=txt_exms, inputs=[text_input], fn=self.img_retri_by_text)
                        
                        k_val = gr.Slider(0, 5, step=1, value=3, interactive=True, 
                                    label="🔝 Top k value", elem_id="sld_id")

                gr.Markdown("#### **Top-k個搜尋結果** (**Top-k result**)")
                with gr.Row():
                    image_output = [ gr.Image(show_label=False, elem_id="img_id") for _ in range(5) ]

            submit_btn.click(self.img_retri_by_text, inputs=[text_input, k_val], outputs=image_output)


class Im_Txt_Retrieval:
    def __init__(self, model, preprocess, key_embs, clear_images):
        self.model = model
        self.preprocess = preprocess
        self.key_embs = key_embs
        self.clear_images = clear_images
        self.folder_id = None


    def img_retri_by_text(self, query_text=None, k_val=None):
        qurey_phases = query_text.split('.')[0] # only process first phase..
        topk_im = retriv_by_phase(qurey_phases, self.key_embs.cuda(), self.clear_images, self.model)

        topk_im = np.clip(topk_im, -1, 1)
        return topk_im[:k_val]
        

    def img_retri_by_image(self, query_img=None, k_val=None):
        query_img = self.preprocess( Image.fromarray(query_img) )
        topk_im = retriv_by_img(query_img.unsqueeze(0), self.key_embs, self.clear_images, self.model, self.preprocess, top_k=k_val+1)

        topk_im = np.clip(topk_im, -1, 1)
        return  topk_im[1:]


    def pkg_demo_app(self, blk_cntr, bkn_tag, db_name):
        lab_lst =['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        demo_doc = \
        '''
        ### Note : the dataset only contains the following kind of images : 
        > ### airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
        > Please confirm the content of **Query text** related with the above categories.
        > Otherwise, the model only return the similar images according to the query text.
        ---
        '''

        with blk_cntr:
            # markdown title
            gr.Markdown(f"## **Image retrival** 🔥🖼️")
        
            # tab layout
            with gr.Tabs():

                with gr.TabItem("image-retrival By Text 📜"):
                    gr.Markdown(demo_doc.format(lab_lst))
                    with gr.Column():
                        gr.Markdown("#### **Query Text**")
                        text_input = gr.Textbox(show_label=False, elem_id="txt_id")
                        
                        gr.Markdown("#### **Top-k matched images**")
                        image_output = gr.Gallery(show_label=False, elem_id="gal_id")
                        k_val = gr.Slider(0, 5, step=1, value=3, interactive=True, 
                                            label="🔝 Top k value", elem_id="sld_id")

                    # show example
                    txt_exms = ['A blue bird stays on the top of the tree.', 'How to define "cute" !?', 
                                'What kind of animal like to eat a banana?', 'Image something flying in the sky!!',
                                'Image something blue that can fly!', 'This car is expensive.', 'How to define "fast" !?',
                                'A tiny red monkey.', 'Could the monkey climb on the tree?', 'A red car is parked in front of a house.',
                                'A cargo ship crosses the ocean.', 'Enjoy an equestrian performance.', 'A lazy dog lies on the ground.',
                                'The Boeing 747.', 'A car is parked near the other one.']  
                    txt_exms = [ [ex, 3] for ex in txt_exms] 
                    examples = gr.Examples(examples=txt_exms, inputs=[text_input, k_val], outputs=image_output, fn=self.img_retri_by_text)
                    
                    # act component & process request..
                    submit_btn = gr.Button("Submit")
                    submit_btn.click(self.img_retri_by_text, inputs=[text_input, k_val], outputs=image_output)

                with gr.TabItem("image-retrival By Image 🖼️"):
                    with gr.Column():
                        gr.Markdown("#### **Query Image**")
                        image_input = gr.Image(label="Upload image should not greater then 300Kb", elem_id="gal_id")
                        
                        gr.Markdown("#### **Top-k matched images**")
                        image_output = gr.Gallery(show_label=False, elem_id="gal_id")
                        k_val = gr.Slider(0, 5, step=1, value=3, interactive=True, 
                                            label="🔝 Top k value", elem_id="sld_id")

                    # show example
                    ims_uri = glob(f"/workspace/data/{db_name}_ex/*.jpg")
                    ims_uri = [ [ex, 3] for ex in ims_uri] 
                    examples = gr.Examples(examples=ims_uri, inputs=[image_input], outputs=image_output, fn=self.img_retri_by_image)

                    # act component & process request..
                    submit_btn = gr.Button("Submit")
                    submit_btn.click(self.img_retri_by_image, inputs=[image_input, k_val], outputs=image_output)

                
if __name__ == "__main__":
    # ViT-B/32, ViT-B/16, ViT-L/14(best), ViT-L/14@336px on stl_ds
    bkn_tag = "ViT-L/14"   
    pkl_tag = "".join(re.findall(r'[A-Za-z0-9]', bkn_tag))
    db_name = "stl"

    ## pre-load model to speed-up 
    model, preprocess = select_backbone(backbone=bkn_tag)

    # pre-load clear_ims
    with open(f'/db/{db_name}_clnIms.pickle', 'rb') as f:
        clear_images = pickle.load(f)
    # pre-load embs
    with open(f'/db/{db_name}_emb_{pkl_tag}.pickle', 'rb') as f:
        key_embs = pickle.load(f)

    # pre-define css to adjust each component
    css_def = '''
    #gal_id {height: 30rem !important; width: 40% !important;}
    #txt_id {width: 40% !important;}
    #sld_id {width: 20% !important;}
    '''
    blk_demo_1 = gr.Blocks(css=css_def)
    
    # package demo app into block layout
    demo = Im_Txt_Retrival(model, preprocess, key_embs, clear_images)
    demo.pkg_demo_app(blk_demo_1, bkn_tag, db_name)
    #blk_demo_1.launch(share=True, enable_queue=True)

    # maybe apply gr.TabbedInterface to intergrate our demo app!!
    demo = gr.TabbedInterface([blk_demo_1, blk_demo_1, blk_demo_1], 
                               ["Multimodal Image-Retrival", "Attention Visualization (segmentation)", "Patch-level Matching"],
                               css=css_def
                               )
    demo.launch(share=True, enable_queue=True)