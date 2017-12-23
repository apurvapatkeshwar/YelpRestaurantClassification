from os.path import join
from utils import check_mkdir

caffe_mode = "GPU"  # GPU|CPU
caffe_device = 0

caffe_root = "/home/avp329/CV/Yelp/caffe/"
data_root = "/scratch/avp329/YelpData/"

check_mkdir(data_root +  "feature_set")
check_mkdir("submission")

# Models: (model_weights, model_prototxt, mean_image, image_feature_file, biz_feature_file)
models = (
    (caffe_root + "models/bvlc_reference_caffenet.caffemodel",
     caffe_root + "models/bvlc_reference_caffenet.prototxt",
     caffe_root +  "models/ilsvrc_2012_mean.npy",
     data_root +  "feature_set/bvlc_reference_{}_image_features.h5",
     data_root +  "feature_set/bvlc_reference_{}_biz_features_{}.pkl"
    ),
    (caffe_root + "models/places205CNN_iter_300000.caffemodel",
     caffe_root + "models/places205CNN.prototxt",
     caffe_root +  "models/places205CNN_mean.npy",
     data_root +  "feature_set/places205CNN_{}_image_features.h5",
     data_root +  "feature_set/places205CNN_{}_biz_features_{}.pkl"
    ),
    (caffe_root + "models/hybridCNN_iter_700000.caffemodel",
     caffe_root + "models/places205CNN.prototxt",
     caffe_root +  "models/hybridCNN_mean.npy",
     data_root +  "feature_set/hybridCNN_{}_image_features.h5",
     data_root +  "feature_set/hybridCNN_{}_biz_features_{}.pkl"
    ),
    (caffe_root + "models/bvlc_alexnet.caffemodel",
     caffe_root + "models/bvlc_alexnet.prototxt",
     caffe_root +  "models/ilsvrc_2012_mean.npy",
     data_root +  "feature_set/alexnet_{}_image_features.h5",
     data_root +  "feature_set/alexnet_{}_biz_features_{}.pkl"
    ))
comb_modes = ("mean", "max")
  
# Data_sets: (mode, image_folder, photo_to_restaurant, labels)
data_sets = (
    ("train",
     data_root + "train_photos",
     data_root + "train_photo_to_biz_ids.csv",
     data_root + "train.csv"
    ),
    ("test",
     data_root + "test_photos",
     data_root + "test_photo_to_biz.csv",
     data_root + "sample_submission.csv"
    ))

# Blending parameters
n_folds = 10
C = 0.7
threshold = 0.5

# Results of blending
blendLR_feature_fl = data_root + "feature_set/all_models_blendLR_CV_features.pkl"
blendKeras_feature_fl = data_root + "feature_set/all_models_blendKeras_CV_features.pkl"

# Submission files
submission_LRblend_CV_fl = "submission/all_models_blendLR.csv"
submission_nomeuf_fl = "submission/keras_blend_noMEUF.csv"
submission_meuf_fl = "submission/keras_blend_MEUF.csv"
