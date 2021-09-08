from Common import *
import PIL

imgs,gt_masks,my_fnames=load_dataset('./data')
num_segments=3  #将原图像分割成num_segments部分
clustering_fn=kmeans   #指定聚类方法为Kmeans
feature_fn=color_position_features  #指定获取特征图的种类
scale=0.5   ##缩小原图
mean_accuracy=0.0   #初始化平均准确率
segmentations=[]

for i,(img,gt_mask,my_fname) in enumerate(zip(imgs,gt_masks,my_fnames)):   #遍历每一对原图像和gt
    segments=compute_segmentation(img,num_segments,clustering_fn=clustering_fn,feature_fn=feature_fn,scale=scale)   #计算出原图像的分割
    segmentations.append(segments)
    accuracy,mask=evaluate_segmentation(gt_mask,segments)  #计算图像分割的准确率
    print('第%d张图像的准确率为: %0.4f' %(i+1,accuracy))
    mean_accuracy+=accuracy
    mask=(mask==0).astype(int)
    plt.imshow(mask,cmap='binary')
    plt.axis('off')
    plt.savefig("./data/my/Kmeans/"+my_fname)

mean_accuracy=mean_accuracy/len(imgs)   #计算所有图像分割的平均准确率
print('所有图像的平均准确率为: %0.4f' % mean_accuracy)
