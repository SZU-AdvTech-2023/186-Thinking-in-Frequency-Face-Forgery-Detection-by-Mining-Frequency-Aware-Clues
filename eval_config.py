from util import mkdir
from options.test_options import TestOptions

# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = './dataset/test/'

# list of synthesis algorithms
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
        'crn', 'imle', 'deepfake', 'stylegan2']
# indicates if corresponding testset has multiple classes
# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
# model
# model_path = 'weights/blur_jpg_prob0.5.pth'

# model_path = '/qiuyx/zoujian/developmentEnv/CNNDetection/checkpoints/method_new_Both/method_new_Both.pth'