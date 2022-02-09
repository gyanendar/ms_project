import imp
import importlib
import os
from models.basemodel import call_model,MODEL

TRAIN_DIRECTORY = 'dataset/RIM-ONE_DL_images/partitioned_randomly/training_set'
TEST_DIRECTORY  = 'dataset/RIM-ONE_DL_images/partitioned_randomly/test_set'
print('total training normal images:', len(os.listdir(TRAIN_DIRECTORY+'/normal')))
print('total training glaucoma images:', len(os.listdir(TRAIN_DIRECTORY+'/glaucoma')))
print('total validation normal images:', len(os.listdir(TEST_DIRECTORY+'/normal')))
print('total validation glaucoma images:', len(os.listdir(TEST_DIRECTORY+'/glaucoma')))

call_model(TRAIN_DIRECTORY,TEST_DIRECTORY,MODEL.BASELINE_CNN,(256,256,3),iterations=70)