
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float
from skimage import color
from skimage import transform
from skimage import io

def kmeans(features,k,num_iters=200):
    N,D=features.shape  #获取图像特征的像素个数以及每个像素点的通道数
    idxs=np.random.choice(N,size=k,replace=False)   #随机生成k个随机数
    centers=features[idxs]  #使用指定的随机数对应的像素点作为初始聚类中心
    assignments=np.zeros(N) #将每个像素点对应所属的类别初始化为0
    for n in range(num_iters):
        featureMap=np.tile(features,(k,1))  #将整个图像特征重复k次
        centerMap=np.repeat(centers,N,axis=0)   #将每一个聚类中心重复像素点个数次，从而使得featureMap和centerMap形成了所有的像素-聚类中心对应关系
        dist=np.linalg.norm(featureMap-centerMap,axis=1).reshape(k,N)   #通过对featureMap和centerMap每一个点（是像素或聚类中心）的做差结果求L2范数的方式计每一组像素和聚类中心之间的距离
        assignments=np.argmin(dist,axis=0)  #对每个像素找出离其最近的聚类中心的下标
        newCenters=np.zeros((k,D))  #初始化新的聚类中心
        for idx in range(k):
            index=np.where(assignments==idx)    #筛选出所有属于第idx类的像素
            newCenters[idx]=np.mean(features[index],axis=0)  #计算第idx类像素的新聚类中心
        if np.allclose(newCenters,centers): #如果新的中心点和上一次迭代的聚类中心位置完全相同则可以提前退出
            break
        else:
            centers=newCenters    #更新新的聚类中心
    return assignments

def hierarchical_clustering(features,k):
    N,D=features.shape  #获取图像特征的像素个数以及每个像素点的通道数
    assignments=np.arange(N)    #初始每个像素自身就是一个独立的簇
    centers=np.copy(features)   #初始每个像素自身就是所在簇的中心
    n_clusters=N
    while n_clusters>k:
        dist=pdist(centers) #计算所有簇的中心间的欧氏距离
        distMatrix=squareform(dist) #将对角阵转换为完整矩阵
        distMatrix=np.where(distMatrix!=0.0,distMatrix,1e5) # 因为squareform构建的完整矩阵上对角线是零，影响了最短距离的判断，此处将0变为1e5，从而避免影响
        #获取最小值所在的行和列，即距离最近的两个簇的index，取较小的那个作为保留，较大的那个进行合并
        minRow,minCol=np.unravel_index(distMatrix.argmin(),distMatrix.shape)
        saveIdx=min(minRow,minCol)
        mergeIdx=max(minRow,minCol)
        #将簇mergeIdx的点分配给saveIdx所在簇
        assignments=np.where(assignments!=mergeIdx,assignments,saveIdx)
        #因为要删除一个簇mergeIdx，所以下标的变化为:小于mergeIdx的不改变，大于mergeIdx的需要减一
        assignments=np.where(assignments<mergeIdx,assignments,assignments-1)
        #删除被合并的簇所在中心
        centers=np.delete(centers,mergeIdx,axis=0)
        #对新合并得到的簇，计算新的簇的中心
        saveIdxIndecies=np.where(assignments==saveIdx)
        centers[saveIdx]=np.mean(features[saveIdxIndecies],axis=0)
        n_clusters-=1
    return assignments

def color_features(img):
    H,W,C=img.shape #获取图像的高度，宽度和通道数
    img=img_as_float(img) #将图像像素转换成浮点数表示
    features=np.reshape(img, (H*W, C))  #将图像的每个像素的所有颜色通道作为特征通道，得到对应图像大小（高度、宽度）的图像特征
    return features

def color_position_features(img):
    H,W,C=img.shape #获取图像的高度，宽度和通道数
    color=img_as_float(img) #将图像像素转换成浮点数表示
    features=np.zeros((H*W, C+2))  #预留出两个通道的位置，用于存放像素的位置特征
    position=np.dstack(np.mgrid[0:H,0:W]).reshape((H*W,2))  #得到每个像素的位置信息
    features[:,0:C]=np.reshape(color,(H*W,C))    #将图像的每个像素的所有颜色通道作为特征通道
    features[:,C:C+2]=position  #特征通道再额外拼接上各个像素的位置信息这两个通道
    features=(features-np.mean(features,axis=0))/(np.std(features, axis=0)) #对图像特征中各个元素每个通道的数值进行中心化    
    return features

def color_position_gradient_features(img):
    H,W,C=img.shape #获取图像的高度，宽度和通道数
    color=img_as_float(img) #将图像像素转换成浮点数表示
    position=np.dstack(np.mgrid[0:H,0:W]).reshape((H*W,2))  #得到每个像素的位置信息
    grayImg=color.rgb2gray(img) #将图像换成灰度图形式
    gradient=np.gradient(grayImg)   #计算图像每个像素在水平和垂直方向上的梯度值
    gradient=np.abs(gradient[0])+np.abs(gradient[1])    #将两个方向上的梯度值进行相加
    features=np.zeros((H*W,C+3))
    features[:,0:C]=np.reshape(color,(H*W,C))    #将图像的每个像素的所有颜色通道作为特征通道
    features[:,C:C+2]=position  #特征通道再额外拼接上各个像素的位置信息这两个通道
    features[:,C+2]=gradient.reshape((H*W))  #特征通道再额外拼接上各个像素的两个方向上的梯度值之和这个通道
    features=(features-np.mean(features,axis=0))/(np.std(features, axis=0)) #对图像特征中各个元素每个通道的数值进行中心化
    return features

def compute_accuracy(mask_gt,mask):
    accuracy=np.mean(mask==mask_gt)   #计算分割方案和gt的重合度，作为该种分割方案的准确度
    return accuracy

def evaluate_segmentation(mask_gt, segments):
    num_segments=np.max(segments)+1 #将图像分割成的部分总数
    max_accuracy=0  #初始化准确率最大的分割部分的准确率
    for i in range(num_segments):
        mask=(segments==i).astype(int)  #获取所有被分割成第i部分的像素形成的掩模，转换成相当于只有这部分是白色，其余为黑色的分割结果
        accuracy=compute_accuracy(mask_gt,mask) #对比选取分割出的白的为当前掩模部分的情况下，和gt的重合率
        max_accuracy=max(accuracy,max_accuracy) #和gt重合率最大的即为准确率最高的分割方案
    for i in range(num_segments):
        mask=(segments==i).astype(int)  #获取所有被分割成第i部分的像素形成的掩模，转换成相当于只有这部分是白色，其余为黑色的分割结果
        accuracy=compute_accuracy(mask_gt,mask) #对比选取分割出的白的为当前掩模部分的情况下，和gt的重合率
        if (accuracy==max_accuracy):
            return max_accuracy,mask

def visualize_mean_color_image(img, segments):
    img=img_as_float(img)   #将图像像素转换成浮点数表示
    k=np.max(segments)+1    #将图像分割成的部分总数
    mean_color_img=np.zeros(img.shape)  #初始化每个分割部分的颜色
    for i in range(k):
        mean_color=np.mean(img[segments==i], axis=0)    #计算一个分割部分（聚出的类）的所有像素的平均颜色
        mean_color_img[segments==i]=mean_color  #将该分割部分中所有像素的颜色修改为该平均颜色
    plt.imshow(mean_color_img)  #可视化各个分割部分
    plt.axis('off')
    plt.show()

def compute_segmentation(img,k,clustering_fn=kmeans,feature_fn=color_position_features,scale=0):
    H,W,C=img.shape #获取图像的高度，宽度和通道数
    if scale>0:
        img=transform.rescale(img,scale,multichannel=True)  #将图像进行缩小处理，加快计算速度
    features=feature_fn(img)    #获取图像的特征图
    assignments=clustering_fn(features,k)   #获取图像的聚类结果
    segments=assignments.reshape((img.shape[:2]))   #将分类结果重塑成和原图像像素点一一对应的形式
    if scale>0:
        segments=transform.resize(segments,(H,W),preserve_range=True)  #将分割后的图像还原至原本大小
        segments=np.rint(segments).astype(int)  #将线性插值得到的聚类结果四舍五入成整型
    return segments


def load_dataset(data_dir): #加载所有原图像和所有gt的掩模
    imgs=[]
    gt_masks=[]
    my_fname=[]
    for fname in sorted(os.listdir(os.path.join(data_dir,'imgs'))):
        if fname.endswith('.jpg'):  #原图像为jpg格式文件
            img=io.imread(os.path.join(data_dir,'imgs',fname))
            imgs.append(img)
            mask_fname=fname[:-4]+'.png'    #gt是原图像对应的文件名，但格式是png
            gt_mask=io.imread(os.path.join(data_dir,'gt',mask_fname))
            gt_mask=(gt_mask!=0).astype(int)
            gt_masks.append(gt_mask)
            my_fname.append(mask_fname)
    return imgs,gt_masks,my_fname