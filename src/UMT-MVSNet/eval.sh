python3 eval.py --model_version=V1 --dataset=data_infer --testpath=/home/hadoop/scx/buaa/test_data/ws_scan9/dense/0 --loadckpt=./checkpoints/model_blended.ckpt --outdir=/home/hadoop/scx/buaa/test_data/ws_scan9/dense/0/stereo/depth_maps


python3 eval.py --model_version=V1 --dataset=data_infer --testpath=/home/hadoop/scx/mvsnet/mvsnet/test_data/scan9/dense --loadckpt=./checkpoints/model_blended.ckpt

python3 eval.py --model=V1 --dataset=data_eval_transform --testpath=/home/hadoop/scx/mvsnet/mvsnet/test_data/scan9/ --loadckpt=./checkpoints/model_blended.ckpt --syncbn=False --batch_size=1 --inverse_cost_volume --inverse_depth=False --origin_size=False --gn=True --refine=False --save_depth=True --fusion=True --ngpu=1 --fea_net=FeatNet --cost_net=UNetConvLSTMV4 --numdepth=512 --interval_scale=0.4 --max_h=360 --max_w=480 --image_scale=1.0 --pyramid=0 

python fusion.py --testpath=/home/hadoop/scx/mvsnet/mvsnet/test_data/scan9 --testlist=/home/hadoop/scx/mvsnet/mvsnet/UMT-MVSNet/lists/dtu/test.txt --outdir=/home/hadoop/scx/mvsnet/mvsnet/test_data/scan9/refine2/checkpoints_backup_model_blended.ckptmodel_blended.ckpt/ --test_dataset=dtu