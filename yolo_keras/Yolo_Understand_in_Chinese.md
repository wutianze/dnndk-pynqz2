<!--
 * @Author: Sauron Wu
 * @GitHub: wutianze
 * @Email: 1369130123qq@gmail.com
 * @Date: 2019-11-06 17:35:44
 * @LastEditors: Sauron Wu
 * @LastEditTime: 2019-11-20 09:51:13
 * @Description: 
 -->
# 简介
下面记录下一些自己在学习yolo过程中遇到的一些问题的梳理，网上有很多解析yolo的文章，但是感觉说的大概意思对了但是很多细节不是很清晰，以至于在自己去修改yolo源码的时候或者高级使用的时候会有困惑。文章总结了一些比较难懂的又比较关键的地方，并不全，但基本上看完以后就大概能去改yolo的代码了，要深入了解Yolo可以看本文的参考链接。

# 文中常数为：
- For v1:每个cell有2个bbox，20个类别，图片拆成7\*7
- For v2:每个cell有5个anchor box，图片拆成13\*13个

# 重要概念
- bounding box = bbox，每个bbox预测其位置信息（中心点坐标和长宽共4个值）和一个置信度。
- 每个网格cell可以有多个bbox，同时每个网格会另外预测n个类别的概率，即每个格子有5\*bbox_num + class_num个值。
- ground true box（人工标记的物体）
- confidence置信度，confience=Pr（ground true box在该网格中则为1，不在则为0）\*IOU(预测的bbox和ground true box之间的重合度)
- NMS的打分score，score是针对某个cell的某个类别的某个bbox的，则每个bbox会有20个score，每个网格有2\*20个score，每个类别有7\*7\*2个score，score=该网格该类别的概率\*该bbox的confidence，它代表了预测的bbox属于某一类的概率以及该bbox是这个物体的可能性。
- anchor box，yolov2中新加入的box，在最后的特征图中每个cell有n个anchor box，这些anchor box是通过k-means在训练集中聚类出来的。它只是一个形状和尺度信息，用来更好地规范bbox的尺度和形状，而bbox还有中心点位置信息。

# yoloV1
1. 每个格子预测n个bounding box和这个格子是否是某一类物体的中心，则最终输出为7\*7（分成7\*7个格子）\*30，30=（20类概率+2（两个bounding box）\* 5）
2. 输入为一张图片，输出一个7\*7\*30的结果向量，之后通过NMS（非极大值抑制）来选择最终结果。NMS通过打分来选出最好的结果，每个bbox会有20个这样的score。所以对于每个网格有20\*2个，每个对象有7\*7\*2个得分，总共有7\*7\*2\*20个得分，这里的2是bounding box的数目,20是类别数量。
3. 之后会先进行一波清理，设置一个阈值，把所有score低于该值的bounding box全舍弃，原本每个类有7\*7\*2个bbox，清理完会少一些。然后就进行NMS来删除重复的框。
4. NMS具体过程：
    - 遍历每个类（此时这个对象拥有的bounding box已经是经过筛选了）：
        - 选出分数最高的bounding box并将其加入输出列表
        - 将该对象的其他的bbox与上面选出的分数最高的那个计算IOU，设置一个阈值，大于阈值表示重叠度高，则把它分数设置为0
        - 然后在剩余的bounding box列表中选择分数最高的然后重复以上过程，直到没有剩余的bbox或者剩余的bbox分数都为0，那么结束
5. 此时，得到一个bbox的列表，每个类别都会有一个列表，不同类别留下来的bbox可能会有重叠，这时候会取每个cell对应的<=2\*20个score（其中有一些的分数可能已经被设置成0从而没有进入列表，有些版本的NMS算法会把所有bbox都加入列表不过值为0也会被舍弃，所以是一样的）选其中score最高的（别的版本NMS则最高的score不能为0）则为该cell对应的类别（所以一个cell只有一个预测，有多个bbox只取所有类别中得分最高的）。

# yoloV2
1. 相比于v1，加上了batch normalization，用了更高分辨率的输入，移除全连接层改用Faster R-CNN中的anchor思想。
2. v1中最终输出为7\*7\*(5\*2+20)，而v2中为13\*13（特征图长宽为13）\*5（每个cell有5个anchor box，每个anchor box对应1个bbox）\*(5（坐标信息4个加一个confidence）+20（20类别）)。
3. 在基于region proposal的目标检测算法中，是通过预测tx,ty来得到x,y的值，即预测的偏移为(tx,ty,tw,th)，其中x,y为框的中心坐标，(xa,ya,wa,ha)为anchor的位置和大小
    ```
    tx = (x - xa)/wa
    ty = (y - ya)/ha
    tw = log(w/wa)
    th = log(h/ha)
    ```
    但这样的话tx,ty没有限制，预测的框容易向任何方向偏移，导致预测的边界框可能处于图片中任一位置所以模型不稳定。  
    所以yolov2没有采用这种方法，而是和yolov1一样预测边界框的中心点相对于对应网格左上角的偏移值，每个cell有5个anchor box来预测5个bbox，每个bbox预测5个值(tx,ty,tw,th)分别为bbox的坐标和边长信息的偏移值offset，通过它们可以算出最终的bbox形状和位置(bx,by,bw,bh)。(Cx,Cy)为当前cell左上角点相对于图像左上角的坐标，anchor box的宽和高为(Pw,Ph)。为了将bbox的中心约束在当前cell中，使用sigmoid(σ函数)将tx,ty做归一化。
    ```
    bx = σ(tx) + Cx
    by = σ(tx) + Cy
    bw = Pw * e^tw
    bh = Ph * e^th
    ```
4. v2中添加了一个转移层，把高分辨率的浅层特征连接（通过堆积channel）到低分辨率的深层特征，然后进行融合和检测。具体就是获取到前层的26\*26\*512的特征图变为13\*13\*2048，然后与最后输出的13\*13\*1024的特征图进行连接为13\*13\*3072，之后再做卷积预测。这样对小目标检测能力提升。

# yoloV3
1. 引入resnet残差模块，这样能解决深层网络的梯度问题。
2. 相比v2的两种尺度的特征图，v3编程了3中尺度：1/32, 1/16, 1/8。在79层之后经过几个卷积得到1/32(13\*13)的结果，特征图的感受野较大，所以检测物体尺寸大；上面的结果经过上采样再与61层的结果进行concat再经过几个卷积得到1/16的预测结果，适合中等尺寸；91层的结果经过上采样再与36层的进行concat再经过几个卷积得到1/8的结果，感受野最小，适合小尺寸。
3. yolov3对bbox进行预测的时候，采用了logistic regression。yolo v3每次对bbox进行predict时，输出和v2一样都是(tx,ty,tw,th,to)​ ,至此机器学习模型的部分已经结束，(tx,ty,tw,th,to)\*总的bbox数量即为模型最终的输出，接下去的处理得另外写代码进行，这一点非常关键，也就意味你需要自己写代码去把输出转化成绝对坐标值和尺寸(x, y, w, h, c)，并且需要自己写代码去完成筛选比如NMS算法。  
4. yolov3的输入为416\*416\*3，通过darknet得到3种不同尺度的预测结果。分别为：13\*13\*3（3个先验框，也可理解为bbox）\*(5+80（类别数）)；26\*26*3\*(5+80)；52\*52\*3\*(5+80)。每个bbox即为一个预测出来的可能目标，包含(5+80)个值。

# 输入输出总结
- yolov3的模型输入为416\*416\*3的图片，实际上可以输入任意大小的图片，在输入模型前做resize（现在的做法是先把图片进行按比例缩放尽可能填满新尺寸，然后在图片周围填充(128,128,128)像素填满新图片），之后会/255来归一化，如果是opencv读入的还会做一个bgr转rgb。
- 普通yolo输出层为3层，tiny为2层，输出为很多个bbox，然后需要过滤掉一些可能性较低的bbox，再把剩余的bbox的坐标转变为图片实际位置坐标，最后输出。

# Reference
[YOLOv1到YOLOv3的演变过程及每个算法详解](https://www.bigdatasafe.org/post/72042.html)
[一文看懂YOLO v2](https://blog.csdn.net/litt1e/article/details/88852745)
[【YOLO学习笔记】之YOLO v1 论文笔记1（超详细：翻译+理解）](https://blog.csdn.net/shuiyixin/article/details/82533849)
[YOLOv2——引入：Anchor+特征融合 (目标检测)(one-stage)(深度学习)(CVPR 2017)](https://blog.csdn.net/Gentleman_Qin/article/details/84349144)