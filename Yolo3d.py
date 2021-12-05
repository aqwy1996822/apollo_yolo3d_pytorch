import os
import time

import cv2
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from yolo3d.config import CameraConfig
from D2toD3.camera_converter import GeometryCameraConverter
import glob
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_anchors = 16
num_classes = 4
anchors = np.array([
    [4.9434993, 1.516986],
    [2.1259836, 1.6779645],
    [19.452609, 17.815241],
    [3.1458852, 2.4994355],
    [15.0302664, 2.3736405],
    [1.2374577, 2.8255595],
    [5.5330938, 3.605915],
    [2.4232311, 0.8086055],
    [0.3672315, 0.6450615],
    [1.3549788, 1.2046775],
    [0.9085392, 0.726555],
    [0.772209, 2.031382],
    [4.0958478, 9.108235],
    [0.5070438, 1.26041],
    [10.0207692, 6.877788],
    [1.9708173, 4.677844]
])

thresh = 0.5
obj_thresh = 0.8
nms_thresh = 0.45


def draw_axis(im):
    cv2.line(im, (0, 980), (1000, 980), (0, 0, 255), 2)
    cv2.line(im, (500, 0), (500, 1000), (0, 0, 255), 2)


def show_fov(im, objs):
    for obj in objs:
        z = 980 - 10 * int(obj.center[2])
        x = 700 + 10 * int(obj.center[0])

        c1 = (int(x - obj.width / 2 * 10), int(z - obj.height / 2 * 10))
        c2 = (int(x + obj.width / 2 * 10), int(z + obj.height / 2 * 10))
        cv2.rectangle(im, c1, c2, (255, 0, 0), 2)
        cv2.circle(im, (x, z), 2, (255, 0, 0), -1)


class obj_3d(object):
    def __init__(self, box):
        self.xmin = box[0]
        self.ymin = box[1]
        self.xmax = box[2]
        self.ymax = box[3]
        self.prob = box[4]
        self.type = box[5]
        self.orientation = box[6]
        self.height = box[7]
        self.width = box[8]
        self.length = box[9]
        self.lof_xmin = box[10]
        self.lof_ymin = box[11]
        self.lof_xmax = box[12]
        self.lof_ymax = box[13]
        self.lor_xmin = box[14]
        self.lor_ymin = box[15]
        self.lor_xmax = box[16]
        self.lor_ymax = box[17]
        self.trunc_width = 0.0
        self.trunc_height = 0.0
        self.distance = 0.0
        self.theta = 0.0
        self.alpha = box[6]
        self.center = np.array([0.0, 0.0, 0.0])
        self.pt8s = np.zeros((8, 2))


class Yolo3d_model(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self):
        super(Yolo3d_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1)
        self.conv3_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)

        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv4_2 = nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1)
        self.conv4_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)

        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.conv5_2 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1)
        self.conv5_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.conv5_4 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1)
        self.conv5_5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)

        self.conv6_1_nodilate = nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=1)
        self.conv6_2 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.conv6_3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.conv6_4 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.conv6_5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)

        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(512 + 256, 512, kernel_size=3, padding=1, stride=1)

        self.conv_final = nn.Conv2d(512, 144, kernel_size=1, padding=0, stride=1)
        self.ori_origin = nn.Conv2d(512, 32, kernel_size=1, padding=0, stride=1)
        self.dim_origin = nn.Conv2d(512, 48, kernel_size=1, padding=0, stride=1)
        self.lof_origin = nn.Conv2d(512, 64, kernel_size=1, padding=0, stride=1)
        self.lor_origin = nn.Conv2d(512, 64, kernel_size=1, padding=0, stride=1)

        self.reduce1_lane = nn.Conv2d(512 + 256, 128, kernel_size=3, padding=1, stride=1)
        self.reduce2_lane = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.reduce3_lane = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.reduce4_lane = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)

        self.deconv1_lane = nn.ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2)
        self.deconv2_lane = nn.ConvTranspose2d(64, 32, kernel_size=2, padding=0, stride=2)
        self.deconv3_lane = nn.ConvTranspose2d(32, 16, kernel_size=2, padding=0, stride=2)
        self.deconv4_lane = nn.ConvTranspose2d(16, 8, kernel_size=2, padding=0, stride=2)

        self.reorg1 = nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1, bias=False)
        self.reorg2 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1, bias=False)
        self.reorg3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1, bias=False)
        self.reorg4 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False)

        self.conv_out = nn.Conv2d(16, 4, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, X):
        data_perm = X.permute(0, 3, 1, 2)
        data_scale = data_perm * 0.00392156885937
        conv1_relu = F.relu(self.conv1(data_scale), inplace=True)
        pool1 = self.pool1(conv1_relu)
        conv2_relu = F.relu(self.conv2(pool1), inplace=True)
        pool2 = self.pool2(conv2_relu)
        conv3_1_relu = F.relu(self.conv3_1(pool2), inplace=True)
        conv3_2_relu = F.relu(self.conv3_2(conv3_1_relu), inplace=True)
        conv3_3_relu = F.relu(self.conv3_3(conv3_2_relu), inplace=True)
        pool3 = self.pool3(conv3_3_relu)

        conv4_1_relu = F.relu(self.conv4_1(pool3), inplace=True)
        conv4_2_relu = F.relu(self.conv4_2(conv4_1_relu), inplace=True)
        conv4_3_relu = F.relu(self.conv4_3(conv4_2_relu), inplace=True)
        pool4 = self.pool4(conv4_3_relu)

        conv5_1_relu = F.relu(self.conv5_1(pool4), inplace=True)
        conv5_2_relu = F.relu(self.conv5_2(conv5_1_relu), inplace=True)
        conv5_3_relu = F.relu(self.conv5_3(conv5_2_relu), inplace=True)
        conv5_4_relu = F.relu(self.conv5_4(conv5_3_relu), inplace=True)
        conv5_5_relu = F.relu(self.conv5_5(conv5_4_relu), inplace=True)
        pool5 = self.pool5(conv5_5_relu)

        conv6_1_relu = F.relu(self.conv6_1_nodilate(pool5), inplace=True)
        conv6_2_relu = F.relu(self.conv6_2(conv6_1_relu), inplace=True)
        conv6_3_relu = F.relu(self.conv6_3(conv6_2_relu), inplace=True)
        conv6_4_relu = F.relu(self.conv6_4(conv6_3_relu), inplace=True)
        conv6_5_relu = F.relu(self.conv6_5(conv6_4_relu), inplace=True)

        conv7_1_relu = F.relu(self.conv7_1(conv6_5_relu), inplace=True)
        conv7_2_relu = F.relu(self.conv7_2(conv7_1_relu), inplace=True)

        concat8 = torch.cat([conv5_5_relu, conv7_2_relu], 1)

        conv9_relu = F.relu(self.conv9(concat8), inplace=True)

        conv_final = self.conv_final(conv9_relu)
        conv_final_permute = conv_final.permute(0, 2, 3, 1)

        slice = torch.split(conv_final_permute, (64, 16, 64), dim=3)
        loc_pred = slice[0]
        cls_reshape = slice[2].reshape(1, 24, -1, 4)
        cls_pred_prob = F.softmax(cls_reshape, dim=3)
        cls_pred = cls_pred_prob.reshape(1, 24, -1, 64)

        obj_pred = torch.sigmoid(slice[1])

        ori_origin = self.ori_origin(conv9_relu)
        dim_origin = self.dim_origin(conv9_relu)
        lof_origin = self.lof_origin(conv9_relu)
        lor_origin = self.lor_origin(conv9_relu)

        ori_pred = ori_origin.permute(0, 2, 3, 1)
        dim_pred = dim_origin.permute(0, 2, 3, 1)
        lof_perm = lof_origin.permute(0, 2, 3, 1)
        lor_perm = lor_origin.permute(0, 2, 3, 1)

        reduce1_lane_relu = F.relu(self.reduce1_lane(concat8), inplace=True)
        deconv1_lane_relu = F.relu(self.deconv1_lane(reduce1_lane_relu), inplace=True)

        reorg4_relu = F.relu(self.reorg4(conv4_3_relu), inplace=True)
        concat4 = torch.cat([reorg4_relu, deconv1_lane_relu], 1)

        reduce2_lane_relu = F.relu(self.reduce2_lane(concat4), inplace=True)
        deconv2_lane_relu = F.relu(self.deconv2_lane(reduce2_lane_relu), inplace=True)
        reorg3_relu = F.relu(self.reorg3(conv3_3_relu), inplace=True)
        concat3 = torch.cat([reorg3_relu, deconv2_lane_relu], 1)

        reduce3_lane_relu = F.relu(self.reduce3_lane(concat3), inplace=True)
        deconv3_lane_relu = F.relu(self.deconv3_lane(reduce3_lane_relu), inplace=True)

        reorg2_relu = F.relu(self.reorg2(conv2_relu), inplace=True)
        concat2 = torch.cat([reorg2_relu, deconv3_lane_relu], 1)

        reduce4_lane_relu = F.relu(self.reduce4_lane(concat2), inplace=True)
        deconv4_lane_relu = F.relu(self.deconv4_lane(reduce4_lane_relu), inplace=True)
        reorg1_relu = F.relu(self.reorg1(conv1_relu), inplace=True)
        concat1 = torch.cat([reorg1_relu, deconv4_lane_relu], 1)
        conv_out = self.conv_out(concat1)

        seg_prob = F.softmax(conv_out, dim=1)
        return cls_pred, obj_pred, ori_pred, dim_pred, lof_perm, lor_perm, seg_prob, loc_pred


def sigmoid(x):
    return 1.0 / (math.exp(-1.0 * x) + 1)


def draw_3d_box(im, lof, lor):
    lof_xmin, lof_ymin, lof_xmax, lof_ymax = lof
    lor_xmin, lor_ymin, lor_xmax, lor_ymax = lor
    cv2.rectangle(im, (lof_xmin, lof_ymin), (lof_xmax, lof_ymax), (255, 0, 255), 1)
    cv2.rectangle(im, (lor_xmin, lor_ymin), (lor_xmax, lor_ymax), (255, 0, 255), 1)
    cv2.line(im, (lof_xmin, lof_ymin), (lor_xmin, lor_ymin), (0, 255, 0), 1)
    cv2.line(im, (lof_xmax, lof_ymax), (lor_xmax, lor_ymax), (0, 255, 0), 1)
    cv2.line(im, (lof_xmin, lof_ymax), (lor_xmin, lor_ymax), (0, 255, 0), 1)
    cv2.line(im, (lof_xmax, lof_ymin), (lor_xmax, lor_ymin), (0, 255, 0), 1)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def transform_boxes(obj_blob, cls_blob, loc_blob, ori_blob, lof_blob, lor_blob, num_classes, anchors, im, fov_flag):
    t0=time.time()
    # batch = obj_blob.shape[0]
    height = obj_blob.shape[1]
    width = obj_blob.shape[2]
    num_anchors = anchors.shape[0]

    img_h, img_w, _ = im.shape
    obj_pred = obj_blob.reshape(-1)
    cls_pred = cls_blob.reshape(-1)
    loc_pred = loc_blob.reshape(-1)
    ori_pred = ori_blob.reshape(-1)
    dim_pred = dim_blob.reshape(-1)
    lof_pred = lof_blob.reshape(-1)
    lor_pred = lor_blob.reshape(-1)
    t1 = time.time()
    ret_list = []
    for i in range(500,min(height * width, 1500)):
        row = i / width
        col = i % width
        for n in range(num_anchors):
            obj_np = np.zeros(18)
            index = i * num_anchors + n
            scale = obj_pred[index]

            orientation = math.atan2(ori_pred[index + 1], ori_pred[index])

            dim_index = index * 3
            d3_h = dim_pred[dim_index + 0]
            d3_w = dim_pred[dim_index + 1]
            d3_l = dim_pred[dim_index + 2]

            box_index = index * 4
            cx = (col + sigmoid(loc_pred[box_index + 0])) / (width * 1.0)
            cy = (row + sigmoid(loc_pred[box_index + 1])) / (height * 1.0)
            w = math.exp(loc_pred[box_index + 2]) * anchors[n, 0] / (width * 1.0) * 0.5
            h = math.exp(loc_pred[box_index + 3]) * anchors[n, 1] / (height * 1.0) * 0.5

            # print("cx:{},cy:{},w:{},h:{}".format(cx, cy, w, h))

            lof_index = index * 4
            lof_x = lof_pred[lof_index + 0] * w * 2 + cx
            lof_y = lof_pred[lof_index + 1] * h * 2 + cy
            lof_w = math.exp(lof_pred[lof_index + 2]) * w
            lof_h = math.exp(lof_pred[lof_index + 3]) * h

            lor_index = index * 4
            lor_x = lor_pred[lor_index + 0] * w * 2 + cx
            lor_y = lor_pred[lor_index + 1] * h * 2 + cy
            lor_w = math.exp(lor_pred[lor_index + 2]) * w
            lor_h = math.exp(lor_pred[lor_index + 3]) * h

            cx = img_w * cx
            cy = img_h * cy
            w = img_w * w
            h = img_h * h

            lof_x = img_w * lof_x
            lof_y = img_h * lof_y
            lof_w = img_w * lof_w
            lof_h = img_h * lof_h

            lor_x = img_w * lor_x
            lor_y = img_h * lor_y
            lor_w = img_w * lor_w
            lor_h = img_h * lor_h

            class_index = index * num_classes
            for k in range(0, num_classes):
                prob = scale * cls_pred[class_index + k]
                if prob > obj_thresh:
                    obj_np[0] = cx - w
                    obj_np[1] = cy - h
                    obj_np[2] = cx + w
                    obj_np[3] = cy + h
                    obj_np[4] = prob
                    obj_np[5] = k
                    obj_np[6] = orientation
                    obj_np[7] = d3_w
                    obj_np[8] = d3_h
                    obj_np[9] = d3_l
                    obj_np[10] = lof_x - lof_w
                    obj_np[11] = lof_y - lof_h
                    obj_np[12] = lof_x + lof_w
                    obj_np[13] = lof_y + lof_h
                    obj_np[14] = lor_x - lor_w
                    obj_np[15] = lor_y - lor_h
                    obj_np[16] = lor_x + lor_w
                    obj_np[17] = lor_y + lor_h

                    ret_list.append(obj_np)
                    # print("add","i",i,"n",n,"index",index)
    t2 = time.time()
    if len(ret_list) != 0:
        ret_list = np.array(ret_list, dtype=np.float32)
        # torchvision.ops.nms(boxes, scores, iou_threshold=self.IOU_THRESHOLD).cpu()
        keep = py_cpu_nms(ret_list, nms_thresh)

        ret_list = ret_list[keep]
    t3 = time.time()
    for i in range(0, len(ret_list)):
        obj = ret_list[i]
        obj_xmin = int(obj[0])
        obj_ymin = int(obj[1])
        obj_xmax = int(obj[2])
        obj_ymax = int(obj[3])
        lof_xmin = int(obj[10])
        lof_ymin = int(obj[11])
        lof_xmax = int(obj[12])
        lof_ymax = int(obj[13])
        lor_xmin = int(obj[14])
        lor_ymin = int(obj[15])
        lor_xmax = int(obj[16])
        lor_ymax = int(obj[17])
        obj_cx = int((obj_xmin + obj_xmax) / 2.0)
        obj_cy = int((obj_ymin + obj_ymax) / 2.0)
        # cv2.rectangle(im,(obj_xmin,obj_ymin),(obj_xmax,obj_ymax),(0,0,255),1)
        draw_3d_box(im, (lof_xmin, lof_ymin, lof_xmax, lof_ymax), (lor_xmin, lor_ymin, lor_xmax, lor_ymax))

        # cv2.putText(im, str(int(obj[5])), (obj_cx, obj_cy-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # cv2.putText(im, str(round(obj[6],2)), (obj_cx, obj_cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # cv2.putText(im, str(round(obj[7],1)), (obj_cx, obj_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # cv2.putText(im, str(round(obj[8],1)), (obj_cx, obj_cy+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # cv2.putText(im, str(round(obj[9],1)), (obj_cx, obj_cy+40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    t4 = time.time()
    ret_obj = []
    if fov_flag:
        for box in ret_list:
            ret_obj.append(obj_3d(box))
    # print("t1-t0", t1 - t0)
    # print("t2-t1", t2 - t1)
    # print("t3-t2", t3 - t2)
    # print("t4-t3", t4 - t3)
    # print()
    return im, ret_obj


def save_caffe2pytorch(net, yolo3d):
    dict_new = yolo3d.state_dict().copy()  # isinstance(dict_new, OrderedDict) is True
    params_list = list(yolo3d.state_dict().keys())

    for i, param in enumerate(params_list):
        # print(param)
        if param.endswith(".weight"):
            weight = net.params[param.replace(".weight", "")][0].data
            weight = weight.reshape((dict_new[param]).shape)
            dict_new[param] = torch.tensor(weight, dtype=torch.float32)
        elif param.endswith(".bias"):
            bias = net.params[param.replace(".bias", "")][1].data
            bias = bias.reshape((dict_new[param]).shape)
            dict_new[param] = torch.tensor(bias, dtype=torch.float32)

    yolo3d.load_state_dict(dict_new)
    torch.save(yolo3d.state_dict(), 'yolo3d_pytorch.pth')
    print('saved done.')


if __name__ == "__main__":
    onnx_flag = True
    fov_flag = False
    caffe_flag = False
    half_flag = True
    if fov_flag:
        gcc = GeometryCameraConverter()
        gcc.init_camera_model(CameraConfig.intrinsic_mat, CameraConfig.w, CameraConfig.h, CameraConfig.distort_params)
        background = cv2.imread("back.jpeg")
        # background = cv2.resize(background, (1000, 1000))
        draw_axis(background)



    yolo3d = Yolo3d_model().to(device)
    yolo3d.load_state_dict(torch.load("yolo3d_pytorch.pth"))
    yolo3d.eval()
    if half_flag:
        yolo3d.half()

    if caffe_flag:
        import caffe

        rootDir = "/home/leo/PycharmProjects/py_yolo3d_apollo"
        deploy_file = os.path.join(rootDir, "deploy.pt")
        weights_file = os.path.join(rootDir, "deploy.md")
        mse = torch.nn.MSELoss()
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(deploy_file, weights_file, caffe.TEST)
        save_caffe2pytorch(net, yolo3d)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out = cv2.VideoWriter('./yolo3d_pytorch9.mp4', fourcc, 30.0, (960, 384), True)

    writer = cv2.VideoWriter()
    # ImageDir = "/media/leo/TOSHIBA/KITTI/dataset/kitti/training/image_2"
    ImageDir = "F:/CV_Bag/CV_Datasets/Apollo/asdt_sample_image/sample_image_9"

    filenames = glob.glob(ImageDir + "/*.jpg")
    for fileindex, filename in enumerate(filenames):

        t0=time.time()
        # 为了防止glob引起的排序错误
        filename = ImageDir + "/" + str(fileindex + 1) + ".jpg"

        im = cv2.imread(filename)
        im = im[156:156 + 768, :, :]
        im = cv2.resize(im, (960, 384))
        orig_p = im.copy()

        im_np = np.array(im, dtype=np.float32)
        X = torch.from_numpy(im_np)
        if half_flag:
            X = X.half()
        X = X.unsqueeze(0)
        X = X.to(device)

        if onnx_flag:
            torch.onnx.export(yolo3d, X, "yolo3d_pytorch.onnx", verbose=True, input_names=["X"],
                              output_names=["cls_pred", "obj_pred", "ori_pred", "dim_pred", "lof_perm", "lor_perm",
                                            "seg_prob", "loc_pred"])
            onnx_flag = False
        t1 = time.time()
        # pytorch
        with torch.no_grad():
            cls_pred, obj_pred, ori_pred, dim_pred, lof_perm, lor_perm, seg_prob, loc_pred = yolo3d(X)
        t2 = time.time()
        cls_blob, obj_blob, ori_blob, dim_blob, lof_blob, lor_blob, lane, loc_blob = cls_pred.cpu().detach().numpy(), obj_pred.cpu().detach().numpy(), ori_pred.cpu().detach().numpy(), dim_pred.cpu().detach().numpy(), lof_perm.cpu().detach().numpy(), lor_perm.cpu().detach().numpy(), seg_prob.cpu().detach().numpy(), loc_pred.cpu().detach().numpy()

        orig_p, ret_obj_p = transform_boxes(obj_blob, cls_blob, loc_blob, ori_blob, lof_blob, lor_blob, num_classes,
                                            anchors, orig_p, fov_flag)

        orig_p[lane[0, 0, :, :] < 0.4] = [255, 192, 0]
        t3 = time.time()
        cv2.imshow("im_pytorch", orig_p)
        out.write(orig_p)

        if fov_flag:
            background_p = background.copy()
            gcc.convert(ret_obj_p)
            show_fov(background_p, ret_obj_p)
            background_p = cv2.resize(background_p, (800, 800))
            cv2.imshow("fov", background_p)

        if caffe_flag:
            # caffe
            orig_c = im.copy()
            net.blobs['data'].data[...] = im_np
            net.forward()
            lane_c = net.blobs["seg_prob"].data
            obj_blob_c = net.blobs["obj_pred"].data
            cls_blob_c = net.blobs["cls_pred"].data
            loc_blob_c = net.blobs["loc_pred"].data
            ori_blob_c = net.blobs["ori_pred"].data
            dim_blob_c = net.blobs["dim_pred"].data
            lof_blob_c = net.blobs["lof_pred"].data
            lor_blob_c = net.blobs["lor_pred"].data

            print("cls_blob", mse(torch.from_numpy(cls_blob_c), torch.from_numpy(cls_blob)))
            print("obj_blob", mse(torch.from_numpy(obj_blob_c), torch.from_numpy(obj_blob)))
            print("loc_blob", mse(torch.from_numpy(loc_blob_c), torch.from_numpy(loc_blob)))
            print("ori_blob", mse(torch.from_numpy(ori_blob_c), torch.from_numpy(ori_blob)))
            print("dim_blob", mse(torch.from_numpy(dim_blob_c), torch.from_numpy(dim_blob)))
            print("lof_blob", mse(torch.from_numpy(lof_blob_c), torch.from_numpy(lof_blob)))
            print("lor_blob", mse(torch.from_numpy(lor_blob_c), torch.from_numpy(lor_blob)))
            print("lane", mse(torch.from_numpy(lane_c[0, 0, :, :]), torch.from_numpy(lane[0, 0, :, :])))

            orig_c, ret_obj_c = transform_boxes(obj_blob_c, cls_blob_c, loc_blob_c, ori_blob_c, lof_blob_c,
                                                lor_blob_c,
                                                num_classes, anchors, orig_c)
            lane0_c = lane_c[0, 0, :, :]
            orig_c[lane0_c < 0.35] = [0, 192, 255]
            cv2.imshow("im_caffe", orig_c)
        print("预处理",t1-t0,"\t推理",t2-t1, "\t后处理", t3-t2)
        if cv2.waitKey(1) == 27:
            break
    out.release()
