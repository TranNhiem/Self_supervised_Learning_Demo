import gradio as gr
import re 
import os

# demo 1, 2
from block_demo import ImgRetrieval_By_Img, ImgRetrieval_By_Txt
from demo_script.inner_interface import select_backbone
import pickle
# demo 3
from Gradio_visualizer import inital, grad_cam_attenction, seg_func, get_result
# demo 4
...

if __name__ == "__main__":
    # demo 1, 2
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

    # package demo app into block layout
    blk_demo_1 = gr.Blocks()
    demo_1 = ImgRetrieval_By_Img(model, preprocess, key_embs, clear_images)
    demo_1.pkg_demo_app(blk_demo_1, bkn_tag, db_name)

    blk_demo_2 = gr.Blocks()
    demo_2 = ImgRetrieval_By_Txt(model, preprocess, key_embs, clear_images)
    demo_2.pkg_demo_app(blk_demo_2, bkn_tag, db_name)
    

    # demo 3.
    inital()
    blk_demo_3 = gr.Blocks()
    with blk_demo_3 as demo:
        gr.Markdown("Click **Run** to see the segmentation and attention result.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(elem_id="input_img")
                btn = gr.Button("Run",elem_id="run_button")
                gr.Examples(
                    examples=[ f"data/seg_ex/{x}" for x in os.listdir("data/seg_ex/")],
                    inputs=inp)
            with gr.Column():
                out1 = gr.Image(elem_id="output_img1")
                out2 = gr.Image(elem_id="output_img2")

        btn.click(fn=get_result, inputs=inp, outputs=[out1,out2])

    # demo 4.
    ...

    # pre-define css to adjust each component
    css_def = '''
    /* Demo 1, 2 CSS style sheet */
    
    #img_id {height: 80% !important;}
    #txt_id {width: 100% !important;}
    #btn_id {width: 20% !important; height: 15% !important;}
    #sld_id {width: 40% !important;}

    /* Demo 3. CSS style sheet*/
    #output_img1 {width: 300px;  
                height: 300px;}
    #output_img2 {width: 300px;  
                height: 300px;}
    '''
    # maybe apply gr.TabbedInterface to intergrate our demo app!!
    tab_demo = gr.TabbedInterface([blk_demo_1, blk_demo_2, blk_demo_3], 
                               ["Image-Retrieval By Image", "Image-Retrieval By Txt", "Segmentation"],
                               css=css_def
                )
    # build on local server, suggested by harry !!
    tab_demo.launch(server_name="0.0.0.0",  server_port=5555,
                    share=False, enable_queue=True)   # enable_queue=True