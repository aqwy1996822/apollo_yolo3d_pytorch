import caffe
import csv
import numpy as np
# np.set_printoptions(threshold='nan')

MODEL_FILE = 'deploy.pt'
PRETRAIN_FILE = 'deploy.md'

net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

p = []
index=0
layer_list=[]

need_layer_names=['conv1', 'conv2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'conv5_5', 'conv6_1_nodilate', 'conv6_2', 'conv6_3', 'conv6_4', 'conv6_5', 'conv7_1', 'conv7_2', 'conv9', 'conv_final', 'ori_origin', 'dim_origin', 'lof_origin', 'lor_origin', 'reduce1_lane', 'reduce2_lane', 'reduce3_lane', 'reduce4_lane', 'deconv1_lane', 'deconv2_lane', 'deconv3_lane', 'deconv4_lane', 'reorg1', 'reorg2', 'reorg3', 'reorg4', 'conv_out']


for param_name in need_layer_names:

    # print("param", len(net.params[param_name]))
    weight = net.params[param_name][0].data
    print(param_name, "weight", weight)
    p.append(weight)
    # print(index, param_name+".weight", net.params[param_name][0].data.shape)
    layer_list.append(param_name+".weight")
    index += 1
    if len(net.params[param_name])==2:
        bias = net.params[param_name][1].data
        print(param_name,"bias", bias)
        p.append(bias)
        # print(index, param_name + ".bias", net.params[param_name][1].data.shape)
        layer_list.append(param_name + ".bias")
        index += 1
print(layer_list)
np.save('params.npy', p)