import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import resnext50_32x4d


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "../../test/imgs"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnext50_32x4d(num_classes=5).to(device)

    # load model weights
    weights_path = "./resnext50_32x4d_30.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)

    # prediction
    model.eval()
    batch_size = 2  # 每次预测时将多少张图片打包成一个batch(程序有bug，只能一个batch全拿出来)
    with torch.no_grad():
        # 一会用来装画图时候的每张图片上的标签
        title_list = []
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            title_batch_list = []
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
                title_img_information = "image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                    class_indict[str(cla.numpy())],
                                                                    pro.numpy())
                # 得到每一个小批量的图片标题
                title_batch_list.append(title_img_information)
            # 把小批量的图像标题整合到一起
            title_list.extend(title_batch_list)
        
        #绘图
        img_list=os.listdir(imgs_root)
        plt.figure(figsize=(20,10)) #设置窗口大小
        plt.suptitle('Multi_Image') # 图片名称
        counter=0
        for img in img_list:
            counter+=1
            full_path=os.path.join(imgs_root,img)
            images = Image.open(full_path)
            plt.subplot(2, 3, counter),plt.title(title_list[counter-1])
            plt.imshow(images),plt.axis('on')
        plt.show()


if __name__ == '__main__':
    main()
    