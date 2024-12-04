import argparse
import os.path
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from torch.nn import functional as F
import torch


class Detic():
    def __init__(self, modelpath, detection_width=800, confThreshold=0.8):
        # net = cv2.dnn.readNet(modelpath)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(modelpath, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.max_size = detection_width
        self.confThreshold = confThreshold
        self.class_names = list(map(lambda x: x.strip(), open('imagenet_21k_class_names.txt').readlines()))
        self.assigned_colors = np.random.randint(0,high=256, size=(len(self.class_names), 3)).tolist()

    def preprocess(self, srcimg):
        im_h, im_w, _ = srcimg.shape
        dstimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        if im_h < im_w:
            scale = self.max_size / im_h
            oh, ow = self.max_size, scale * im_w
        else:
            scale = self.max_size / im_w
            oh, ow = scale * im_h, self.max_size

        max_hw = max(oh, ow)
        if max_hw > self.max_size:
            scale = self.max_size / max_hw
            oh *= scale
            ow *= scale
        ow = int(ow + 0.5)
        oh = int(oh + 0.5)
        dstimg = cv2.resize(dstimg, (ow, oh))
        return dstimg

    def get_mask(self, pred_masks, pred_boxes, output_height, output_width, mask_thres=0.5, device="cuda:0"):
        if isinstance(pred_masks, np.ndarray):
            pred_masks = torch.from_numpy(pred_masks)
        if isinstance(pred_boxes, np.ndarray):
            pred_boxes = torch.from_numpy(pred_boxes)

        pred_masks = pred_masks.to(device)
        pred_boxes = pred_boxes.to(device)
        device = pred_masks.device

        image_shape = (output_height, output_width)
        assert pred_masks.shape[-1] == pred_masks.shape[-2], "Only square mask predictions are supported"
        N = len(pred_masks)
        if N == 0:
            return pred_masks.new_empty((0,) + image_shape, dtype=torch.uint8).cpu().detach().numpy()
        assert len(pred_boxes) == N, pred_boxes.shape

        # num_chunks = N
        if device.type == "cpu" or torch.jit.is_scripting():
            # CPU is most efficient when they are pasted one by one with skip_empty=True
            # so that it performs minimal number of operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks, but may have memory issue
            # int(img_h) because shape may be tensors in tracing
            BYTES_PER_FLOAT = 4
            GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit
            num_chunks = int(np.ceil(N * int(output_height) * int(output_width) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (
                    num_chunks <= N
            ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        img_masks = torch.zeros(
            N, output_height, output_width, device=device, dtype=torch.bool if mask_thres >= 0 else torch.uint8
        )

        for inds in chunks:
            masks_chunk, spatial_inds = self._do_paste_mask(pred_masks[inds, None, :, :], pred_boxes[inds], output_height, output_width, skip_empty=False)

            if mask_thres >= 0:
                masks_chunk = (masks_chunk >= mask_thres).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
                img_masks[inds] = masks_chunk
            else:
                img_masks[(inds,) + spatial_inds] = masks_chunk

        img_masks = img_masks.cpu().detach().numpy()
        return img_masks

    def _do_paste_mask(self, masks, boxes, img_h, img_w, skip_empty=True):
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        device = masks.device

        if skip_empty and not torch.jit.is_scripting():
            x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
                dtype=torch.int32
            )
            x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
            y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
        else:
            x0_int, y0_int = 0, 0
            x1_int, y1_int = img_w, img_h
        x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

        N = masks.shape[0]

        img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
        img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        # img_x, img_y have shapes (N, w), (N, h)

        gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
        gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
        grid = torch.stack([gx, gy], dim=3)

        if not torch.jit.is_scripting():
            if not masks.dtype.is_floating_point:
                masks = masks.float()
        img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

        if skip_empty and not torch.jit.is_scripting():
            return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
        else:
            return img_masks[:, 0], ()

    def nms(self, boxes, scores, threshold):
        # 按得分降序排序
        indices = np.argsort(scores)[::-1]
        selected_indices = []

        while len(indices) > 0:
            # 选择当前得分最高的框
            current_index = indices[0]
            selected_indices.append(current_index)

            # 计算当前框与后续框的重叠情况
            current_box = boxes[current_index]
            x1 = np.maximum(current_box[0], boxes[indices, 0])
            y1 = np.maximum(current_box[1], boxes[indices, 1])
            x2 = np.minimum(current_box[2], boxes[indices, 2])
            y2 = np.minimum(current_box[3], boxes[indices, 3])

            # 计算重叠区域
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)
            overlap_area = w * h

            # 计算当前框与其他框的面积
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            other_areas = (boxes[indices, 2] - boxes[indices, 0]) * (boxes[indices, 3] - boxes[indices, 1])

            # 计算 IOU
            iou = overlap_area / (current_area + other_areas - overlap_area)

            # 选择 IOU 小于阈值的框
            indices = indices[np.where(iou <= threshold)[0]]

        return selected_indices

    def nms_with_classes(self, boxes, scores, classes, threshold):
        selected_indices = []

        unique_classes = np.unique(classes)

        for cls in unique_classes:
            cls_indices = np.where(classes == cls)[0]
            cls_boxes = boxes[cls_indices]
            cls_scores = scores[cls_indices]

            # 调用之前的 NMS 实现
            selected_cls_indices = self.nms(cls_boxes, cls_scores, threshold)
            selected_indices.extend(cls_indices[selected_cls_indices])

        return selected_indices

    def post_processing(self, pred_boxes, scores, pred_classes, pred_masks, im_hw, pred_hw):
        scale_x, scale_y = (im_hw[1] / pred_hw[1], im_hw[0] / pred_hw[0])

        pred_boxes[:, 0::2] *= scale_x
        pred_boxes[:, 1::2] *= scale_y
        pred_boxes[:, [0, 2]] = np.clip(pred_boxes[:, [0, 2]], 0, im_hw[1])
        pred_boxes[:, [1, 3]] = np.clip(pred_boxes[:, [1, 3]], 0, im_hw[0])

        threshold = 0
        widths = pred_boxes[:, 2] - pred_boxes[:, 0]
        heights = pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (widths > threshold) & (heights > threshold)

        pred_boxes = pred_boxes[keep]
        scores = scores[keep]
        pred_classes = pred_classes[keep]
        pred_masks = pred_masks[keep]

        nms_kep = self.nms(pred_boxes, scores, threshold=0.5)
        # nms_kep = self.nms_with_classes(pred_boxes, scores, pred_classes, threshold=0.5)
        pred_boxes = pred_boxes[nms_kep]
        scores = scores[nms_kep]
        pred_classes = pred_classes[nms_kep]
        pred_masks = pred_masks[nms_kep]

        pred_masks = self.get_mask(pred_masks[:, 0, :, :], pred_boxes, output_height=im_hw[0], output_width=im_hw[1], mask_thres=0.5)

        pred = {
            'pred_boxes': pred_boxes,
            'scores': scores,
            'pred_classes': pred_classes,
            'pred_masks': pred_masks,
        }
        return pred

    def draw_predictions(self, img, predictions, f_mask=None):
        height, width = img.shape[:2]
        default_font_size = int(max(np.sqrt(height * width) // 90, 10))
        boxes = predictions["pred_boxes"].astype(np.int64)
        scores = predictions["scores"]
        classes_id = predictions["pred_classes"].tolist()
        masks = predictions["pred_masks"].astype(np.uint8)
        num_instances = len(boxes)
        print('detect', num_instances, 'instances')
        for i in range(num_instances):
            if scores[i] < self.confThreshold:
                continue

            if f_mask is not None:
                if not f_mask[i]:
                    continue
            #
            x0, y0, x1, y1 = boxes[i]
            b_h = y1 - y0
            b_w = x1 - x0
            # # filter big object
            # big_scale = 0.6
            # if b_h > big_scale * height or b_w > big_scale * width:
            #     continue
            #
            # # filter small object
            # small_scale = 0.1
            # if b_h < small_scale * height or b_w < small_scale * width:
            #     continue

            color = self.assigned_colors[classes_id[i]]
            cv2.rectangle(img, (x0, y0), (x1, y1), color=color,thickness=default_font_size // 4)
            text = "{} {:.0f}%".format(self.class_names[classes_id[i]], round(scores[i],2) * 100)
            cv2.putText(img, text, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1, lineType=cv2.LINE_AA)
            # plt.imshow(masks[i])
            # plt.show()
            plt.show()
        return img

    def detect(self, srcimg):
        im_h, im_w = srcimg.shape[:2]
        dstimg = self.preprocess(srcimg)
        pred_hw = dstimg.shape[:2]
        input_image = np.expand_dims(dstimg.transpose(2, 0, 1), axis=0).astype(np.float32)

        # Inference
        pred_boxes, scores, pred_classes, pred_masks = self.session.run(None, {self.input_name: input_image})
        preds = self.post_processing(pred_boxes, scores, pred_classes, pred_masks, (im_h, im_w), pred_hw)
        return preds


from infer_complete_objs_in_scence_online import get_crop_and_expand_rect_by_mask, pad_and_resize_img, resize_to_128_with_K
def save_splatter_input_img(s_root, color_image, preds):
    obj_boxes = preds["pred_boxes"]
    obj_masks = preds["pred_masks"]

    # everything_results = fast_sam_model(color_image[:, :, ::-1], device="cuda:0", retina_masks=True, imgsz=1024, conf=0.6, iou=0.01,)
    # prompt_process = FastSAMPrompt(color_image[:, :, ::-1], everything_results, device="cuda:0")
    # s_color_path = os.path.join(s_root, str(count) + "_color.jpg")
    # s_depth_path = os.path.join(s_root, str(count) + "_depth.npy")
    #
    # cv2.imwrite(s_color_path, color_image)
    # np.save(s_depth_path, depth)

    if len(obj_boxes) > 0:
        # print("obj_boxes:", len(obj_boxes))
        # anns = prompt_process.box_prompt(bboxes=obj_boxes)
        # print("anns:", len(anns))
        # prompt_process.plot(annotations=anns, output_path='./tmp/dog.jpg', )

        complete_ojbs = []
        complete_ojbs_color = []
        complete_ojbs_sematic = []
        for obj_id, object_mask in enumerate(obj_masks):
            object_mask = object_mask > 0
            if np.sum(object_mask) < 20:
                continue

            obj_mask = object_mask

            radius = 10
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            obj_mask = cv2.morphologyEx(obj_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            obj_mask = obj_mask > 0

            crop_x, crop_y, crop_w, crop_h = get_crop_and_expand_rect_by_mask(obj_mask)
            if crop_x == None:
                continue

            # mask有可能是分开的，选择最大的mask部分
            max_part_mask = np.zeros_like(obj_mask)
            max_part_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = 1
            obj_mask = np.bitwise_and(obj_mask, max_part_mask)

            img_crop = color_image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :].copy()

            mask_crop = obj_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            img_crop[~mask_crop] = (255, 255, 255)

            img_crop_pad, crop_pad_h, crop_pad_w = pad_and_resize_img(img_crop, foreground_ratio=0.65)
            img_crop_pad, crop_cam_k, crop_pad_scale = resize_to_128_with_K(img_crop_pad, fov=49.0)

            s_path = os.path.join(s_root, str(obj_id) + ".jpg")
            cv2.imwrite(s_path, img_crop_pad)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='desk.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.3, type=float, help='class confidence')
    parser.add_argument("--modelpath", type=str, default='weights/Detic_C2_R50_640_4x_in21k.onnx', help="onnxmodel path")
    args = parser.parse_args()

    args.modelpath = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/local_files/Detic_C2_R50_640_4x_in21k.onnx"
    # args.imgpath = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/colors/20_color.jpg"
    args.imgpath = "/home/pxn-lyj/Egolee/data/test/透明物体/0_raw_color_shot4.png"

    mynet = Detic(args.modelpath, confThreshold=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)
    preds = mynet.detect(srcimg)

    s_root = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/local_files/tmp"
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)


    save_splatter_input_img(s_root, srcimg, preds)

    srcimg = mynet.draw_predictions(srcimg, preds)
    plt.imshow(srcimg[:, :, ::-1])
    plt.show()

    # cv2.imwrite('result.jpg', srcimg)
    # winName = 'Deep learning Detic in ONNXRuntime'
    # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    # cv2.imshow(winName, srcimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
