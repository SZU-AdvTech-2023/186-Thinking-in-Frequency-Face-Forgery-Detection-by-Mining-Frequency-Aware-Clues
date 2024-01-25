import os
import csv
import torch

from validate import validate
# from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
from networks.improved_method import F3Net_improved
from networks.models import F3Net
# import pydevd_pycharm
# pydevd_pycharm.settrace('172.26.189.245', port=12321, stdoutToServer=True,
#                         stderrToServer=True)
# Running tests
opt = TestOptions().parse(print_options=False)
if opt.elevated == True:
        model_path = './checkpoints/method_new_Both/method_new_Both.pth'
else:
        model_path = './checkpoints/method_old_Both/method_old_Both.pth'

model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    # 精髓=============================
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    # ============================
    opt.no_resize = True    # testing without resizing by default

    # model = resnet50(num_classes=1)
    # model=F3Net(mode='FAD')
    # model = F3Net_improved(mode='Both')
    if opt.elevated == True:
        model = F3Net_improved(mode='Both')
    else:
        model = F3Net(mode='Both')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, _, _, _, _ = validate(model, opt)
    rows.append([val, acc, ap])
    # rows.append([val, acc])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
