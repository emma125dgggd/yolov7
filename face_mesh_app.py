
import base64

import streamlit as st
#import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
import glob
from PIL import Image
import os
import sys
import uuid
#from streamlit_camera_component import st_camera_component

import threading
from typing import Union
#import av
import numpy as np
import streamlit as st
from PIL import Image
#import cv2
import os
#from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
#import streamlit_webrtc as st_webrtc

#import pdfkit
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from datetime import date
import streamlit as st
from streamlit.components.v1 import iframe
#os.add_dll_directory(r"C:\Program Files\GTK3-Runtime Win64\bin")
#sys.path.append('.\yolo\yolov7')

env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
#template = env.get_template("report.html")
import argparse
import time
from pathlib import Path
#import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)






classes_to_filter = None  # You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

opt = {

    "weights": "yolov7.pt",
    # Path to weights file default weights are for nano model
    "yaml": "data/coco.yaml",
    "img-size": 640,  # default image size
    "conf-thres": 0.25,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter  # list of classes to filter or None

}
count =0
#file saver
def save_uploaded_file(uploadedfile):
    with open(os.path.join("Uploads",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
   # return st.success("Saved file :{} in Uploads".format(uploadedfile.name))


def save_uploaded_file1(uploadedfile,i):
    name = uploadedfile.name
    with open(os.path.join("Captured","pic {}.jpg".format(i)),"wb") as f:
        f.write(uploadedfile.getbuffer())



DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

st.title('Cleeve AI')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Menu',
["About",'Run on Image','Run on Video']
)

if app_mode =='About':
    st.markdown('Cleeve AI is an Assistive Medical Diagnostic Software that speeds up the turn around time of Lab Diagnosis ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.video('result_compressed.mp4')

    st.markdown('''
          # Login \n 
             [Signin]() of you don't have an account \n
           Follow Us On:
            - [YouTube]()
            - [LinkedIn]()
            - [Facebook]()
            - [Discord]()
        
             
            ''')



elif app_mode =='Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)



    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    # max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    #video = cv2.VideoCapture(video_path)

    # Video information
    fps = vid.get(cv2.CAP_PROP_FPS)
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    #codec = cv2.VideoWriter_fourcc('V','P','0','9')
    #out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    # Initialzing object for writing video output
    output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))




    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0

    

    st.markdown("<hr/>", unsafe_allow_html=True)

    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes']:
                classes.append(opt['classes'].index(class_name))

        for j in range(nframes):
            i +=1
            #ret, frame = vid.read()


            ret, img0 = vid.read()
            if not ret:
                continue
            if ret:
                img = letterbox(img0, imgsz, stride=stride)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=False)[0]

                pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
                t2 = time_synchronized()
                for i, det in enumerate(pred):
                    s = ''
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

            #currTime = time.time()
            #fps = 1 / (currTime - prevTime)
            #prevTime = currTime
            if record:
                #st.checkbox("Recording", value=True)
                output.write(img0)
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(img0,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
           
            stframe.image(frame,channels = 'BGR',use_column_width=True)

    st.text('Video Processed')

    
    st.video(output.mp4)

    vid.release()
    output. release()

elif app_mode =='Run on Image':


    #st.sidebar.markdown('---')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
   
    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'],
                                               accept_multiple_files=True)
    print(img_file_buffer)

    if img_file_buffer is None:
        img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'],
                                                   accept_multiple_files=True, key=1)
        print(img_file_buffer)

    #st.markdown('---')

  

    if img_file_buffer is not None:
        i=0
        for UploadedFile in img_file_buffer:
            i+=1
            print(UploadedFile)
            print(i)

            file_details={"filename": UploadedFile.name, "file_type":UploadedFile.type}
            #st.write(file_details)
            #st.write(type(img_file_buffer))

            save_uploaded_file(UploadedFile)
            paths = []
            image ="uploads/{}".format(UploadedFile.name)
            paths =paths.append(image)


            st.sidebar.text('Original Image')


            st.sidebar.image(image)

            face_count = 0
            # Dashboard
            def counter():
                with torch.no_grad():
                    weights, imgsz = opt['weights'], opt['img-size']
                    set_logging()
                    device = select_device(opt['device'])
                    half = device.type != 'cpu'
                    model = attempt_load(weights, map_location='cpu')  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check img_size
                    if half:
                        model.half()

                    names = model.module.names if hasattr(model, 'module') else model.names
                    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

                    img0 = cv2.imread(image)
                    img = letterbox(img0, imgsz, stride=stride)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=False)[0]

                    # Apply NMS
                    classes = None
                    if opt['classes']:
                        classes = []
                        for class_name in opt['classes']:
                            classes.append(opt['classes'].index(class_name))

                    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
                    t2 = time_synchronized()
                    for i, det in enumerate(pred):
                        s = ''
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            print('Plasmodium Falciparum Parasite Detected ' + "{}".format(int(n)))

                            crp_cnt=0

                            for *xyxy, conf, cls in reversed(det):
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                                #crop
                                # crop an image based on coordinates
                                object_coordinates = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                                cropobj = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

                                # save crop part
                                crop_file_path = os.path.join("crop", str(crp_cnt) + ".jpg")
                                cv2.imwrite(crop_file_path, cropobj)
                                crp_cnt = crp_cnt + 1


                    st.subheader('Output Image')
                    st.image(img0, use_column_width=True, channels="BGR")
                    st.success('Cell Count Completed \n')

                    st.success("Performing Parasited Cell Detection")


            def parasited():
                with torch.no_grad():
                    weights, imgsz = opt['weights'], opt['img-size']
                    set_logging()
                    device = select_device(opt['device'])
                    half = device.type != 'cpu'
                    model = attempt_load(weights, map_location='cpu')  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check img_size
                    if half:
                        model.half()

                    names = model.module.names if hasattr(model, 'module') else model.names
                    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

                    img0 = cv2.imread(image)
                    img = letterbox(img0, imgsz, stride=stride)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=False)[0]

                    # Apply NMS
                    classes = None
                    if opt['classes']:
                        classes = []
                        for class_name in opt['classes']:
                            classes.append(opt['classes'].index(class_name))

                    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
                    t2 = time_synchronized()
                    for i, det in enumerate(pred):
                        s = ''
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            print('Plasmodium Falciparum Parasite Detected ' + "{}".format(int(n)))

                            for *xyxy, conf, cls in reversed(det):
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                    #kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)


                    st.image(img0, use_column_width=True, channels="BGR")
                    st.success('Parasited Cell Detection Complete')
                    st.success("Performing Parasite Staging")



            def staging():
                with torch.no_grad():
                    weights, imgsz = opt['weights'], opt['img-size']
                    set_logging()
                    device = select_device(opt['device'])
                    half = device.type != 'cpu'
                    model = attempt_load(weights, map_location='cpu')  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check img_size
                    if half:
                        model.half()

                    names = model.module.names if hasattr(model, 'module') else model.names
                    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

                    img0 = cv2.imread(image)
                    img = letterbox(img0, imgsz, stride=stride)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=False)[0]

                    # Apply NMS
                    classes = None
                    if opt['classes']:
                        classes = []
                        for class_name in opt['classes']:
                            classes.append(opt['classes'].index(class_name))

                    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
                    t2 = time_synchronized()
                    for i, det in enumerate(pred):
                        s = ''
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            print('Plasmodium Falciparum Parasite Detected ' + "{}".format(int(n)))

                            for *xyxy, conf, cls in reversed(det):
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                    #kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)


                    st.image(img0, use_column_width=True, channels="BGR")
                    st.success('Staging Completed')
                    st.success("Performing Speciation")


            def species():
                with torch.no_grad():
                    weights, imgsz = opt['weights'], opt['img-size']
                    set_logging()
                    device = select_device(opt['device'])
                    half = device.type != 'cpu'
                    model = attempt_load(weights, map_location='cpu')  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check img_size
                    if half:
                        model.half()

                    names = model.module.names if hasattr(model, 'module') else model.names
                    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

                    img0 = cv2.imread(image)
                    img = letterbox(img0, imgsz, stride=stride)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=False)[0]

                    # Apply NMS
                    classes = None
                    if opt['classes']:
                        classes = []
                        for class_name in opt['classes']:
                            classes.append(opt['classes'].index(class_name))

                    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
                    t2 = time_synchronized()
                    for i, det in enumerate(pred):
                        s = ''
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            print('Plasmodium Falciparum Parasite Detected ' + "{}".format(int(n)))

                            for *xyxy, conf, cls in reversed(det):
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                    #kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)

                    st.subheader('Output Image')
                    st.image(img0, use_column_width=True, channels="BGR")
                    st.success('Speciation Completed')
                    #st.success("Performing Parasited Cell Detection")

            counter()
            parasited()
            staging()
            species()
            env = Environment(loader=FileSystemLoader('templates'))
            template = env.get_template('report.html')
            html = template.render(
            )
            print(html)

            st.success("Your Report was generated!")
            with open('pdf.html', "w") as f:
                f.write(html)
            # st.write("")
            from weasyprint import HTML, CSS

            css = CSS(string=''' 
                        @page {size :A4: margin: 1cm}
                        ''')
            data = HTML('pdf.html').write_pdf("Lab Report.pdf", )

            pdf_display = None

            # help(st.download_button)
            with open(os.path.join("Lab Report.pdf"), "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                # PDFbyte = f.read()

                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

            with open("Lab Report.pdf", "rb") as pdf_file:
                PDFbyte = pdf_file.read()

            st.download_button(label="Download PDF", key='6',
                               data=PDFbyte,
                               file_name="Lab Report.pdf",
                               mime='application/octet-stream')



else:
        demo_image = DEMO_IMAGE
        image = demo_image
        st.sidebar.text('Original Image')

        st.sidebar.image(image)


        def counter2():
            with torch.no_grad():
                weights, imgsz = opt['weights'], opt['img-size']
                set_logging()
                device = select_device(opt['device'])
                half = device.type != 'cpu'
                model = attempt_load(weights, map_location='cpu')  # load FP32 model
                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(imgsz, s=stride)  # check img_size
                if half:
                    model.half()

                names = model.module.names if hasattr(model, 'module') else model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

                img0 = cv2.imread(image)
                img = letterbox(img0, imgsz, stride=stride)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=False)[0]

                # Apply NMS
                classes = None
                if opt['classes']:
                    classes = []
                    for class_name in opt['classes']:
                        classes.append(opt['classes'].index(class_name))

                pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
                t2 = time_synchronized()
                for i, det in enumerate(pred):
                    s = ''
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        print('Plasmodium Falciparum Parasite Detected ' + "{}".format(int(n)))

                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>",
                                unsafe_allow_html=True)
                st.success('Cell Count Completed'"Performing Parasited Cell Detection")
                st.success("Performing Parasited Cell Detection")
                st.subheader('Output Image')
                st.image(img0, use_column_width=True, channels="BGR")


        counter2()





