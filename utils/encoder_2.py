import tensorflow as tf
import tf_util2
#import tflearn
#import utils.pointnet2_utils.tf_util
from utils.pointnet2_utils.pointnet_util import pointnet_sa_module, pointnet_fp_module, import_knn_point_2, sample_points
from utils.pointnet2_utils.pointSIFT_util import pointSIFT_res_module
#from utils.grouping.tf_grouping import knn_point_2
import numpy as np
import warnings

# from tflearn.layers.core import fully_connected, dropout
# from tflearn.layers.conv import conv_1d, avg_pool_1d, global_avg_pool
# from tflearn.layers.normalization import batch_normalization
# from tflearn.layers.core import fully_connected, dropout
# from tensorflow.contrib.framework import arg_scope
# from tensorflow.contrib.layers import batch_norm
# from tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers

def sample(npoint, xyz):
    outputs = sample_points(npoint, xyz)
    return outputs


def get_gen_model(point_cloud, is_training, scope, bradius = 1.0, reuse=None, use_rv=False, use_bn = False,use_ibn = False,
                  use_normal=False,bn_decay=None, up_ratio = 4,idx=None):

    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        l0_xyz = point_cloud[:,:,0:3]
        if use_normal:
            l0_points = point_cloud[:,:,3:]
        else:
            l0_points = None
        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.05,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=bradius*0.1,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer3')

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.3,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers
        up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                       scope='fa_layer1',bn=use_bn,ibn = use_ibn)

        up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                       scope='fa_layer2',bn=use_bn,ibn = use_ibn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn,ibn = use_ibn)

        ###concat feature
        with tf.variable_scope('up_layer',reuse=reuse):
            new_points_list = []
            for i in range(up_ratio):
                concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=-1)
                concat_feat = tf.expand_dims(concat_feat, axis=2)
                concat_feat = tf_util2.conv2d(concat_feat, 256, [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=False, is_training=is_training,
                                              scope='fc_layer0_%d'%(i), bn_decay=bn_decay)

                new_points = tf_util2.conv2d(concat_feat, 128, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=use_bn, is_training=is_training,
                                             scope='conv_%d' % (i),
                                             bn_decay=bn_decay)
                new_points_list.append(new_points)
            net = tf.concat(new_points_list,axis=1)

        #get the xyz
        coord = tf_util2.conv2d(net, 64, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='fc_layer1', bn_decay=bn_decay)

        coord = tf_util2.conv2d(coord, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='fc_layer2', bn_decay=bn_decay,
                             activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3
        coord = tf.squeeze(coord, [2])  # B*(2N)*3

    return coord, None


def get_local_features_with_sift(point_cloud, is_training, scope, bradius = 1.0, reuse=None, use_rv=False, use_bn = False,use_ibn = False,
                  use_normal=False,bn_decay=None, up_ratio = 4,idx=None):

    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        l0_xyz = point_cloud[:,:,0:3]
        if use_normal:
            l0_points = point_cloud[:,:,3:]
        else:
            l0_points = None
        # Layer 1
        sl0_xyz, sl0_points, sl0_indices = pointSIFT_res_module(l0_xyz, l0_points, radius=bradius*0.05, out_channel=32, is_training=is_training, bn_decay=bn_decay, scope='layer_sift_0', merge='concat')
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(sl0_xyz, sl0_points, npoint=num_point, radius=bradius*0.05,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        sl1_xyz, sl1_points, sl1_indices = pointSIFT_res_module(l1_xyz, l1_points, radius=bradius*0.2, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer_sift_1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(sl1_xyz, sl1_points, npoint=num_point/2, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        # l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
        #                                                    nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
        #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.3,bn=use_bn,ibn = use_ibn,
        #                                                    nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
        #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # # Feature Propagation layers
        # up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
        #                                scope='fa_layer1',bn=use_bn,ibn = use_ibn)

        # up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
        #                                scope='fa_layer2',bn=use_bn,ibn = use_ibn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, l1_points, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn,ibn = use_ibn)
        
        return up_l2_points, l1_points, l0_xyz

def get_local_features(point_cloud, is_training, scope, bradius = 1.0, reuse=None, use_rv=False, use_bn = False,use_ibn = False,
                  use_normal=False,bn_decay=None, up_ratio = 4,idx=None):

    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        l0_xyz = point_cloud[:,:,0:3]
        if use_normal:
            l0_points = point_cloud[:,:,3:]
        else:
            l0_points = None
        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.05,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')
        print 'l1_points:', l1_points.get_shape

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        # l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
        #                                                    nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
        #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.3,bn=use_bn,ibn = use_ibn,
        #                                                    nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
        #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # # Feature Propagation layers
        # up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
        #                                scope='fa_layer1',bn=use_bn,ibn = use_ibn)

        # up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
        #                                scope='fa_layer2',bn=use_bn,ibn = use_ibn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, l1_points, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn,ibn = use_ibn)
        print 'up_l2_points:', up_l2_points.get_shape
        
        return up_l2_points, l1_points, l0_xyz


        
def concat_features(features, up_l2_points, l1_points, global_feature, is_training, scope, reuse=None,
                  bn_decay=None):
        ###concat feature
        with tf.variable_scope(scope,reuse=reuse):
            up_ratio = 2
            new_points_list = []
            for i in range(up_ratio):
                concat_feat = tf.concat([features, up_l2_points, l1_points, global_feature], axis=-1)
                concat_feat = tf.expand_dims(concat_feat, axis=2)
                concat_feat = conv2d(concat_feat, 256, [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=False, is_training=is_training,
                                              scope='fc_layer0_%d'%(i), bn_decay=bn_decay)

                new_points = conv2d(concat_feat, 128, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=False, is_training=is_training,
                                             scope='conv_%d' % (i),
                                             bn_decay=bn_decay)
                new_points_list.append(new_points)
            net = tf.concat(new_points_list,axis=1)
            print 'net:', net.get_shape
        return net

def concat_features4(features, up_l2_points, l1_points, global_feature, is_training, scope, reuse=None,
                  bn_decay=None):
        ###concat feature
        with tf.variable_scope(scope,reuse=reuse):
            up_ratio = 4
            new_points_list = []
            for i in range(up_ratio):
                concat_feat = tf.concat([features, up_l2_points, l1_points, global_feature], axis=-1)
                concat_feat = tf.expand_dims(concat_feat, axis=2)
                concat_feat = conv2d(concat_feat, 256, [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=False, is_training=is_training,
                                              scope='fc_layer0_%d'%(i), bn_decay=bn_decay)

                new_points = conv2d(concat_feat, 128, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=False, is_training=is_training,
                                             scope='conv_%d' % (i),
                                             bn_decay=bn_decay)
                new_points_list.append(new_points)
            net = tf.concat(new_points_list,axis=1)
            print 'net:', net.get_shape
        return net

def feature_extraction(inputs, scope='feature_extraction', is_training=True, bn_decay=None):

    with tf.variable_scope(scope,reuse=False):

        use_bn = False
        use_ibn = False
        growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)
        print 'l0 shape is:', l0_features.get_shape
        #  <bound method Tensor.get_shape of <tf.Tensor 'feature_extraction/Squeeze:0' shape=(32, 1024, 24) dtype=float32>>

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer1", is_training=is_training, bn=use_bn, ibn=use_ibn,
                                                  bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84
        print 'l1 shape is:', l1_features.get_shape
        #  <bound method Tensor.get_shape of <tf.Tensor 'feature_extraction/concat:0' shape=(32, 1024, 120) dtype=float32>>


        l2_features = conv1d(l1_features, comp, 1,  # 24
                                     padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144
        print 'l2 shape is:', l2_features.get_shape
        # <bound method Tensor.get_shape of <tf.Tensor 'feature_extraction/concat_1:0' shape=(32, 1024, 240) dtype=float32>>
    return l2_features

def feature_extraction_original(inputs, scope='feature_extraction', is_training=True, bn_decay=None):

    with tf.variable_scope(scope,reuse=False):

        use_bn = False
        use_ibn = False
        growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)
        print 'l0 shape is:', l0_features.get_shape
        #  <bound method Tensor.get_shape of <tf.Tensor 'feature_extraction/Squeeze:0' shape=(32, 1024, 24) dtype=float32>>

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer1", is_training=is_training, bn=use_bn, ibn=use_ibn,
                                                  bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84
        print 'l1 shape is:', l1_features.get_shape
        #  <bound method Tensor.get_shape of <tf.Tensor 'feature_extraction/concat:0' shape=(32, 1024, 120) dtype=float32>>


        l2_features = conv1d(l1_features, comp, 1,  # 24
                                     padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144
        print 'l2 shape is:', l2_features.get_shape
        # <bound method Tensor.get_shape of <tf.Tensor 'feature_extraction/concat_1:0' shape=(32, 1024, 240) dtype=float32>>

        l3_features = conv1d(l2_features, comp, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204
        print 'l3 shape is:', l3_features.get_shape
        # l3 shape is: <bound method Tensor.get_shape of <tf.Tensor 'feature_extraction/concat_2:0' shape=(32, 1024, 360) dtype=float32>>

        l4_features = conv1d(l3_features, comp, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264

        l4_features = tf.expand_dims(l4_features, axis=2)

        print 'l4 shape is:', l4_features.get_shape
        # l4 shape is: <bound method Tensor.get_shape of <tf.Tensor 'feature_extraction/ExpandDims_1:0' shape=(32, 1024, 1, 480) dtype=float32>>

    return l4_features



def dense_conv(feature, n=3,growth_rate=64, k=16, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=False):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        print 'y shape is:',y.get_shape
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y,idx


def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = import_knn_point_2(k, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.
    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable
    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=None):
  """ 2D convolution with non-linear operation.
  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs
