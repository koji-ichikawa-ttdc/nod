#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# YOLOXでビデオ映像から人物の映っているフレームだけ抽出してみた
# https://qiita.com/masashi-ai/items/66684c3fd953091c819d#26-extract_cls%E9%96%A2%E6%95%B0
# python nearby_od.py video -n yolox-l -c yolox_l.pth --path video.avi --save_result --extract person --frame_rate 5 --device gpu

# 指定した車両をトラッキング
# https://qiita.com/toyohisa/items/d4a8361c4f16a7766065
# python nearby_od.py video -n yolox-nano -c /near_object/yolox_nano.pth --path /near_object/20231031_140959_F_Nor.AVI --save_result --object_width 10

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np
import json

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def extract_cls(output, extract_class):
    if extract_class is not None:
        if output is not None:
            # クラス名はinfres[6]で確認
            extracted = [infres.cpu().detach().numpy() for infres in output
                            # if infres[6] == COCO_CLASSES.index(extract_class)]
                            if infres[6] in [COCO_CLASSES.index(cls) for cls in extract_class]]
            if extracted != []:
                return torch.tensor(np.array(extracted))
            else:
                return None
    else:
        return output

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument("--extract", nargs='*', default=["person", "bicycle", "car", "motorcycle", "bus", "truck"], help="extract a class")
    parser.add_argument("--frame_rate", type=int, default=0, help="frame rate")
    parser.add_argument("--object_width", type=int, default=200, help="detect object width")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


# def image_demo(predictor, vis_folder, path, current_time, save_result):
def image_demo(predictor, vis_folder, path, current_time, args):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        outputs[0] = extract_cls(outputs[0], args.extract)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        # if save_result:
        if args.save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )


    frame_rate = int(fps) if args.frame_rate == 0 else args.frame_rate
    frame_num = -1
    stored_outputs = []
    file_save = False
    while True:
        ret_val, frame = cap.read()
        frame_num += 1
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            outputs[0] = extract_cls(outputs[0], args.extract)
            # 検出物体なしのときは空のリストを追加、またbboxは元画像のサイズに変換して保存
            if outputs[0]==None:
                stored_outputs.append([])
            else:
                # logger.info(f"outputs[0][0]: {outputs[0][0]}")
                output_np = outputs[0].cpu().detach().numpy().copy()
                bbox = (output_np[:,0:4]/img_info["ratio"]).astype(int)
                stored_outputs.append(np.hstack((bbox,output_np[:,4:])).tolist())

                near_object = False
                for i in range(len(bbox)):
                    object_width = bbox[i][2] - bbox[i][0]
                    if object_width > args.object_width:
                        near_object = True
                        file_save = True
                        logger.info(f"object_width:{object_width},{np.hstack((bbox,output_np[:,4:])).tolist()}")
    

            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)

            if args.save_result:
                if args.extract is None:
                    vid_writer.write(result_frame)
                elif (outputs[0] is not None) and (args.extract is not None):
                    if near_object:
                        minutes = frame_num//(frame_rate*60)
                        seconds = (frame_num - minutes*(frame_rate*60)) // frame_rate
                        frame = cv2.putText(result_frame, "Time: {:d} min {:d} sec".format(minutes, seconds),
                                    (60, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 127), 2)
                        frame = cv2.putText(result_frame, "Frame Num: {:3d}".format(frame_num),
                                    (60, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 127), 2)
                        vid_writer.write(result_frame)

            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    # stored_outputsをjsonファイルに保存
    if file_save:
        stored_outputs_filename = f"{save_path}_outputs.json"
        with open(stored_outputs_filename,"wt") as f:
            json.dump(stored_outputs,f)
            logger.info(f"saved to {stored_outputs_filename}")

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        # image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
        image_demo(predictor, vis_folder, args.path, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    start_time = time.time()

    main(exp, args)

    end_time = time.time()

    processing_time = end_time - start_time
    logger.info(f"time:{int(processing_time)} s")
          