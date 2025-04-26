import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

class datatrcsie(data.Dataset):
    def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.pdlabels = []
        count = 0

        for i in [0,1,2,3,4,5,6,7,8]:
            for j in range(1,7):
                for k in [2,3,4,5]:
                    for o in [1,2,3,4,5]:
                        for n in [1,2,3,4]:
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)
                            self.pdlabels.append(j-j)
                            count = count + 1
        self.n_data = count
        print(count)

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pdlabel':
            self.pdlabels=np.array(self.pdlabels)
            self.pdlabels[tindex.astype(int)] = tlabels.astype(int)
            self.pdlabels.tolist()

    def __getitem__(self, item):
        img_paths, labels, pdlabels = self.img_paths[item], self.img_labels[item] , self.pdlabels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = self.transform(inputs)
            labels = int(labels)
            pdlabels = int(pdlabels)

        return inputs, labels, pdlabels, item

    def __len__(self):
        return self.n_data

class datatecsie(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.pdlabels = []
        count = 0

        for i in [0,1,2,3,4,5,6,7,8]:
            for j in range(1,7):
                for k in  [1]:
                    for o in  [1,2,3,4,5]:
                        for n in [1,2,3,4,5]:
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)
                            self.pdlabels.append(j-j)
                            count = count + 1
        self.n_data = count
        print(count)

     def __getitem__(self, item):
        img_paths, labels, pdlabels = self.img_paths[item], self.img_labels[item] , self.pdlabels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = self.transform(inputs)
            labels = int(labels)
            pdlabels = int(pdlabels)

        return inputs, labels, pdlabels, item

     def __len__(self):
        return self.n_data