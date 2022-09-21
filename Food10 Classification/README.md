Another simple course code without any advance trick. In this project, I use data augmentation before training, and in the neural network I use Batchnorm, CNN and MaxPooling as a block before a full connected layer.  
There are some unlabeled pictures in the dataset, and I try a direct way to use these data that giving these pictures pseudo labels using current model and concat this dataset with train dataset.  
It didn't work so well without doubt. Well....maybe I should learn some other methods about semi-supervised learning.
