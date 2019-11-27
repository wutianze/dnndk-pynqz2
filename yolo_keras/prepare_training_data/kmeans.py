import glob
import numpy as np
import argparse


class YOLO_Kmeans:

    def __init__(self, cluster_number, label_path):
        self.cluster_number = cluster_number
        self.label_path = label_path

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data,output):
        f = open(output, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self,w,h):
        dataSet = []
        files = glob.glob(self.label_path+'*.txt')
        for f in files:
            if(f != self.label_path+"classes.txt"):
                ff = open(f,'r')
                for line in ff:
                    infos = line.split(" ")
                    width = float(infos[3])*w
                    height = float(infos[4])*h
                    dataSet.append([width, height])
                ff.close()

        result = np.array(dataSet)
        print(result)
        return result

    def txt2clusters(self,width,height,output):
        all_boxes = self.txt2boxes(width,height)
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result,output)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("description='--cluster_number:how many clusters to make, --width --height:the image size, --label_path:the labels dir, --output:anchors file name'")
    parser.add_argument("--cluster_number", type=int,default=5)
    parser.add_argument("--width", type=int,default=416)
    parser.add_argument("--height", type=int,default=416)
    parser.add_argument("--label_path", type=str,default="images/labels/")
    parser.add_argument("--output", type=str,default="yolo_anchors.txt")
    
    args = parser.parse_args()
    
    kmeans = YOLO_Kmeans(args.cluster_number, args.label_path)
    kmeans.txt2clusters(args.width,args.height,"generated/"+args.output)
