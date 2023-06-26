'''Label file for different datasets.'''

SCANNET_LABELS_20 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture','counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
                     'toilet', 'sink', 'bathtub', 'otherfurniture')

# SCANNET_LABELS_20 = ('stool', 'tissue box', 'toilet', 'toaster', 'floor', 'door', 'toaster oven', 'clock', 'mirror',
#                      'tv', 'guitar', 'bed', 'laundry basket', 'microwave', 'refrigerator', 'object', 'bicycle',
#                      'curtain', 'coffee table', 'couch', 'dish rack', 'doorframe', 'guitar case', 'desk', 'sink',
#                      'kitchen cabinets', 'kitchen counter', 'window', 'shelf', 'trash can', 'cabinet', 'table',
#                      'wall', 'ceiling', 'shower', 'nightstand', 'shoes', 'scale', 'backpack', 'pillow')

SCANNET_COLOR_MAP_20 = {
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    16: (219., 219., 141.),
    24: (255., 127., 14.),
    28: (158., 218., 229.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    36: (227., 119., 194.),
    39: (82., 84., 163.),
    0: (0., 0., 0.), # unlabel/unknown
}

