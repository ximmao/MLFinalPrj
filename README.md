# MLFinalPrj
Final Project for Comp 652 on Image Style Transfer and Combination using StarGAN

Implementation based on official StarGAN code https://github.com/yunjey/stargan, dataset loaded with "RaFD" mode of original implementation hence the name 

Some Changes:
Modify solver.py to solver_combine_trainLConly.py with added feature to support combined target label
Add pixelwiseLC.py for network training linear combination coefficients.

Keep all the arguments in the original implementation, add two: 
   --label_combine_list: show combination in samples during training
   --train_combine_label: switch between transfer and combination mode
   
Sample scripts:
To train further
python main.py --mode train --dataset RaFD --rafd_crop_size 256 --image_size 256 --rafd_image_dir data/RaFD/train --c_dim 7  --sample_dir stargan_LC_t/samples --log_dir stargan_LC_t/logs --model_save_dir stargan_LC_t/models --result_dir stargan_LC_t/results --use_tensorboard False --g_conv_dim 64 --d_conv_dim 64 --batch_size 16 --resume_iters 150000 --label_combine_list 15 02 06 36 03 34 --train_combine_label True --num_iters 400000 --num_iters_decay 100000 --g_lr 0.00001 --d_lr 0.00001

To generate the test results
python main.py --mode test --dataset RaFD --rafd_crop_size 256 --image_size 256 --rafd_image_dir data/RaFD/test --c_dim 7  --sample_dir stargan_LC_t/samples --log_dir stargan_LC_t/logs --model_save_dir stargan_LC_t/models --result_dir stargan_LC_t/results --use_tensorboard False --g_conv_dim 64 --d_conv_dim 64 --batch_size 16 --test_iters 150000 --label_combine_list 15 02 06 36 03 34 --train_combine_label True
