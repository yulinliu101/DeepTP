# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:46:49 2017

@author: Yulin Liu
"""
try:
    import cPickle as pickle
except:
    import _pickle as pickle
import numpy as np
import os

IMG_SIZE = 20
Label_Correspondence = {
0:0,
1:3,
2:6,
3:9,
4:12,
5:15,
6:18,
7:18,
8:18,
9:18,
10:18,
11:18,
12:18,
13:18,
14:18,
15:18,
16:18,
17:18,
18:18,
19:18,
20:18,
21:18,
22:18,
23:18,
24:1,
25:4,
26:7,
27:10,
28:13,
29:16,
30:19,
31:19,
32:19,
33:19,
34:19,
35:19,
36:19,
37:19,
38:19,
39:19,
40:19,
41:19,
42:19,
43:19,
44:19,
45:19,
46:19,
47:19,
48:2,
49:5,
50:8,
51:11,
52:14,
53:17,
54:20,
55:20,
56:20,
57:20,
58:20,
59:20,
60:20,
61:20,
62:20,
63:20,
64:20,
65:20,
66:20,
67:20,
68:20,
69:20,
70:20,
71:20
}

Label_Correspondence_2 = {
0:0,
1:0,
2:1,
3:2,
4:3,
5:0,
6:0,
7:0,
8:0,
9:0,
10:0,
11:0,
12:0,
13:0,
14:0,
15:0,
16:0,
17:0,
18:0,
19:0,
20:0,
21:0,
22:0,
23:0,
24:0,
25:0,
26:1,
27:2,
28:3,
29:0,
30:0,
31:0,
32:0,
33:0,
34:0,
35:0,
36:0,
37:0,
38:0,
39:0,
40:0,
41:0,
42:0,
43:0,
44:0,
45:0,
46:0,
47:0,
48:0,
49:0,
50:1,
51:2,
52:3,
53:0,
54:0,
55:0,
56:0,
57:0,
58:0,
59:0,
60:0,
61:0,
62:0,
63:0,
64:0,
65:0,
66:0,
67:0,
68:0,
69:0,
70:0,
71:0}
def convertDataset(image_dir):
    num_labels = 72
    label = np.eye(num_labels)  # Convert labels to one-hot-vector
    num_labels2 = 21
    label2 = np.eye(num_labels2)  # Convert labels to one-hot-vector
    num_labels3 = 4
    label3 = np.eye(num_labels3)  # Convert labels to one-hot-vector
                   
    i = 0
    j = 0
    img_out = []
    label_out = []
    label_out_2 = []
    label_out_3 = []
    Point_Order = []
    
    for dirName in os.listdir(image_dir):
        label_i = label[int(dirName)]
        label_i_2 = label2[Label_Correspondence[int(dirName)]]
        label_i_3 = label3[Label_Correspondence_2[int(dirName)]]
        
        print("ONE_HOT_ROW = ", int(dirName))
        path = os.path.join(image_dir, dirName)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            if os.path.isfile(img_path) and img.endswith('p'):
                try:
                    img_in = pickle.load(open(img_path,'rb')).reshape(IMG_SIZE*IMG_SIZE*6)
                    if sum(img_in) == 0:
                        os.remove(img_path)
                        j += 1
                        print('remove: ',img_path)
                        pass
                    else:
                        i += 1
                        img_out.append(img_in)
                        label_out.append(label_i)
                        label_out_2.append(label_i_2)
                        label_out_3.append(label_i_3)
                        Point_Order.append(img)
                except:
                    j += 1
                    print(img_path)
                    pass
            if i % 103000 == 0:
                X_name = 'All_image_' + str(i) + '.p'
                Y_name = 'All_image_label_' + str(i) + '.p'
                Y_name_2 = 'All_image_label_2' + str(i) + '.p'
                P_name = 'Point_Order' + str(i) + '.p'
                
                X = np.asanyarray(img_out)
                Y = np.asanyarray(label_out)
                Y_2 = np.asanyarray(label_out_2)
                Point_Idx = np.asanyarray(Point_Order)
                pickle.dump(X, open(X_name, 'wb'), protocol = 2)
                pickle.dump(Y, open(Y_name, 'wb'), protocol = 2)
                pickle.dump(Y_2, open(Y_name_2, 'wb'), protocol = 2)
                pickle.dump(Point_Idx, open(P_name, 'wb'), protocol = 2)
                img_out = []
                label_out = []
                label_out_2 = []
                Point_Order = []
            
    print('Total valid: ', i)                
    print('Total removal/waste: ', j)
    return np.asanyarray(img_out), np.asarray(label_out), np.asarray(label_out_2), np.asarray(label_out_3),Point_Order


def saveDataset(IMAGE_DIR, X_name, Y_name, Y_name_2, Y_name_3):
    X, Y, Y_2, Y_3, Point_Idx = convertDataset(IMAGE_DIR)

    pickle.dump(X, open(X_name, 'wb'), protocol = 2)
    pickle.dump(Y, open(Y_name, 'wb'), protocol = 2)
    pickle.dump(Y_2, open(Y_name_2, 'wb'), protocol = 2)
    pickle.dump(Y_3, open(Y_name_3, 'wb'), protocol = 2)
    pickle.dump(Point_Idx, open('Point_Order.p', 'wb'), protocol = 2)