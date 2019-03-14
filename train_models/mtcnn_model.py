#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
num_keep_radio = 0.7
#define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

def cls_ohem(cls_prob, label):

    #cls_prob=[batch_size,2]
    #lable = [batch_size]
    #生成一个label大小的0矩阵
    zeros = tf.zeros_like(label)
    #label=-1 --> label=0net_factory

    #pos -> 1, neg -> 0, others -> 0
    #当tf.less(label,0)为ture时，返回zeros，否则,返回label,即为[1 0 1 1 0 0 .....]
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    #把原来的batch_size行2列转换成batch_size*2行1列，相当于把预测值平铺
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    # get the number of rows of class_prob
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    #row = [0,2,4.....]
    row = tf.range(num_row)*2
    #表示当indices为单数时，表示脸部正例，当为单数时，表示为脸部负例
    indices_ = row + label_int
    #tf.gather(a,b)表示a按照b的索引顺序排列
    #此处的tf.gather()表示cls_prob_reshape按照indices_索引取数，分别取出对应的概率
    #表示正确分类的概率值
    #label_prob.shape = (384) ,其中每一位数，代表了其正确分类的概率,，单数表示pos的概率，双数表示neg的概率
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    #-tf.log(label_prob)表示的是log(p),其中的1e-10代表了10^(-10)=0.0000000001。主要是为了防止为0
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    # 因为face detection 中用到的数据是negatives + positives
    # 我们只需要正例和负例，并且在label_prob中，我们都已经给出了正例和负例的相关概率，所以对应的label设置为1
    # 对于其他事例我们把其label设置为0
    valid_inds = tf.where(label < zeros,zeros,ones)
    # get the number of POS and NEG examples
    # 计算正例和负例的总数
    num_valid = tf.reduce_sum(valid_inds)
    #num_keep_radio=0.7
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #FILTER OUT PART AND LANDMARK DATA
    #这个交叉熵损失函数所用的事例是正例和负例，其他事例因为label所对应的值为0，所以不影响计算，即pos的loss乘以1，neg的loss乘以1，其他的loss乘以0
    loss = loss * valid_inds
    #tf.nn.top_k(a,k)如果a是一维：表示返回a中前k个最大值，如果a是多维：表示返回每一行中的前k个最大值
    #此处表示返回前70%的损失函值
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)
def bbox_ohem_orginal(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    #pay attention :there is a bug!!!!
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    #keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

#label=1 or label=-1 then do regression
#bounding box regression 用 positives + part faces
# bbox_pred 表示网络训练得出的结果batch*4
# bbox_target表示目标bbox的实际未知信息
def bbox_ohem(bbox_pred,bbox_target,label):
    '''
    
    :param bbox_pred:
    :param bbox_target:
    :param label: class label
    :return: mean euclidean loss for all the pos and part examples
    '''
    #label.shape = (384,)
    #bbox_pred = [batch_size,4]
    #bbox_target = [batch_size,4]
    #label = [batch_size]
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    # keep pos and part examples
    #如果label的值为1或-1时，结果为ones_index,否则为zeros_index
    #因为1表示正例，-1表示part，这两部分都有人脸。所以可以找到具体的坐标位置
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    #(batch,)
    #calculate square sum
    #计算距离的平方差
    square_error = tf.square(bbox_pred-bbox_target)
    #
    square_error = tf.reduce_sum(square_error,axis=1)
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    # count the number of pos and part examples
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    # keep top k examples, k equals to the number of positive examples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)

def landmark_ohem(landmark_pred,landmark_target,label):
    '''

    :param landmark_pred:
    :param landmark_target:
    :param label:
    :return: mean euclidean loss
    '''
    #keep label =-2  then do landmark detection
    #landmark_pred = [batch_size,10]
    #landmark_target = [batch_size,10]
    #label = [batch_size]
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    #-2表示的是landmark
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    #返回的就是一个[BATCH_SIZE,10]的数组，其中每个值都是减后平方的结果
    square_error = tf.square(landmark_pred-landmark_target)
    #表示每一行的所有值相加，即10个landmark的坐标误差的加和
    square_error = tf.reduce_sum(square_error,axis=1)
    #验证集的总数量
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #只保留label是-2的数据，数据为square_error(每一行所有误差的平方和)。其他标签的数据为0，类似于[1.6531092 1.9187249 0.        0.        2.4839485 0.        1.6831231]这种数据形式
    square_error = square_error*valid_inds
    #tf.nn.top_k返回一个key,value形式的数据，_,表示了前keep_num大的具体的数值，k_index表示其坐标
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    #返回前keep_num大的square_error的值，其值为具体的误差平方和，类似于：[2.8851743 2.5473971 1.98878 ...... ]
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
    
def cal_accuracy(cls_prob,label):
    '''

    :param cls_prob:
    :param label:
    :return:calculate classification accuracy for pos and neg examples only
    '''
    # get the index of maximum value along axis one from cls_prob
    # 0 for negative 1 for positive
    #获取每一行中最大概率的坐标，例如：[0 1 1 0 1 .....]
    pred = tf.argmax(cls_prob,axis=1)
    #label的值为1,0，-1，-2
    label_int = tf.cast(label,tf.int64)
    # return the index of pos and neg examples
    #其中tf.greater_equal()表示大于等于
    #tf.where(boolean)表示返回值为true的列下标
    #表示返回pos和neg的下标
    cond = tf.where(tf.greater_equal(label_int,0))
    #print("cond:",cond.shape)
    #把原来的[size,1]的数组，squeeze成[size]的数组,就是去掉为1的列
    picked = tf.squeeze(cond)
    # gather the label of pos and neg examples
    #获取下标为pos和neg的下标所对应的概率值
    label_picked = tf.gather(label_int,picked)
    #返回预测值中所对应的pos和neg的下标所对应的概率值
    pred_picked = tf.gather(pred,picked)
    #calculate the mean value of a vector contains 1 and 0, 1 for correct classification, 0 for incorrect
    # ACC = (TP+FP)/total population
    #首先使用tr.equal(a,b)判断a和b是否相等，然后使用tf.cast(boolean)方法把boolean值转换成1和0.
    #然后通过tf.readuce_mean()方法计算平均值，即1存在的概率，也就是最后的精确值
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op


def _activation_summary(x):
    '''
    creates a summary provides histogram of activations
    creates a summary that measures the sparsity of activations

    :param x: Tensor
    :return:
    '''

    tensor_name = x.op.name
    print('load summary for : ',tensor_name)
    tf.summary.histogram(tensor_name + '/activations',x)
    #tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))





#construct Pnet
#label:batch
def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    #define common param
    #提供默认值
    with slim.arg_scope([slim.conv2d],
                        #activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):
        print(inputs.get_shape())

        #slim.conv2d(inputs,num_outputs,kernel_size,stride=1)
        #其中inputs是需要进行卷积的图像
        #num_outputs指定卷积核的个数(就是filter的个数)
        #kernel_size指用于指定卷积核的维度(宽和高)
        net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        _activation_summary(net)
        print(net.get_shape())
        #slim.max_pool2d(inputs,kernel_size)
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        _activation_summary(net)
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        _activation_summary(net)
        print(net.get_shape())
        #
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        _activation_summary(net)
        print(net.get_shape())
        #batch*H*W*2
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        _activation_summary(conv4_1)
        #conv4_1 = slim.conv2d(net,num_outputs=1,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.sigmoid)
        
        print (conv4_1.get_shape())
        #batch*H*W*4
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        _activation_summary(bbox_pred)
        print (bbox_pred.get_shape())
        #batch*H*W*10
        landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
        _activation_summary(landmark_pred)
        print (landmark_pred.get_shape())

        #print("PNetend:",end)
        # add projectors for visualization





        #cls_prob_original = conv4_1 
        #bbox_pred_original = bbox_pred
        if training:
            #batch*2
            # calculate classification loss
            #tf.squeeze(t)表示默认删除所有为1的维度，而[1,2]表示只删除下标为1和2维的1
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            #计算face classification的损失值
            cls_loss = cls_ohem(cls_prob,label)
            #batch
            # cal bounding box error, squared sum error
            # 由于bbox_pred.shape = batch * 1 * 1 * 4,所以进行squeeze去掉下标为1，2的两个为1的维度
            #squeeze后的bbox_pred维度为batch * 4
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            #batch*10
            landmark_pred = tf.squeeze(landmark_pred,[1,2],name="landmark_pred")
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)

            accuracy = cal_accuracy(cls_prob,label)
            #ge_regularization_losses()表示获取整体的正则化loss，应该是给后面计算用的，提前创建出来
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        #test
        else:
            #when test,batch_size = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
            return cls_pro_test,bbox_pred_test,landmark_pred_test
        
def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print (inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        print (net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        print(net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
        print(net.get_shape())
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1")
        print(fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print(bbox_pred.get_shape())
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        print(landmark_pred.get_shape())
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
    
def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    #提供一个新的作用域（scope），称为arg_scope，在该作用域（scope）中，用户可以定义一些默认的参数，用于特定的操作
    #默认值，表示在没有声明的情况下使用默认值，如果有声明，则被覆盖
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print(net.get_shape())
        #net 3*3*128
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        #1*1*256
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        print(fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print(bbox_pred.get_shape())
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        print(landmark_pred.get_shape())
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            #标签为-2的数据集数有num,则返回num数量的landmark的误差的平方和，每个平方和表示每行中10个landmark坐标的预测值和target值得差值的平方和
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred