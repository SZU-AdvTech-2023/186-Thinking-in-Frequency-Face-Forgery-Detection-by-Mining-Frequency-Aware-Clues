原始方法训练：
python train.py --name method_old_Both --mode_F3 Both  --gpu_ids 1 --niter 100 --batch_size 8 --blur_prob 0 --blur_sig 0.5 --jpg_prob 0.0 --jpg_method cv2,pil --jpg_qual 100 --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse

改进方法训练：
python train.py --name method_new_Both --elevation --gpu_ids 0 --niter 100 --batch_size 8 --blur_prob 0 --blur_sig 0.5 --jpg_prob 0.0 --jpg_method cv2,pil --jpg_qual 100 --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse

原始方法测试：
python eval.py 
改进方法测试：
python eval.py --elevated
