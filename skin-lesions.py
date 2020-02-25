import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.callbacks import *
from torchvision.models import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

path = Path("../input")
labels = pd.read_csv('../input/HAM10000_metadata.csv', sep=',')
labels.head()

imageid = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(path, '*', '*.jpg'))}

labels['path'] = labels['image_id'].map(imageid.get)
labels['path'] = labels['path'].str[9:]
labels.sample(5)
f, ax1 = plt.subplots(1,1, figsize=(10,5))
sns.boxplot(x=labels['dx'], y=labels['age'], hue=labels['sex'], ax=ax1)
plt.title('Diagnosis by gender')
plt.show()
tfms = get_transforms(do_flip=True, 
                      flip_vert=True,
                      max_zoom=1.1,
                      max_warp=0.2, 
                      p_affine=0.5,
                      xtra_tfms=[rotate(degrees=(-45,45),p=.1),
                                brightness(change=(0.35,0.65),p=.5),
                                contrast(scale=(0.8,1.2),p=.5),
                                dihedral(p=1)])

np.random.seed(21)
data = ImageDataBunch.from_df(path='../input/', df=labels,
                              ds_tfms=tfms, size=224,bs=16,
                               valid_pct=0.2, fn_col='path', 
                              label_col='dx'
                              ).normalize(imagenet_stats)

data.show_batch(rows=3)

arch=densenet121
learner = create_cnn(data, arch=arch, metrics=[accuracy],ps=.5,model_dir="/tmp/model/",
                    callback_fns=ShowGraph).to_fp16()

learner.lr_find()
learner.recorder.plot()

learner.fit(1, 1e-2)
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(3, max_lr=slice(1e-5, 1e-3), wd=0.1)

