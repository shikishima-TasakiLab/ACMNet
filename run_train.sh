# if trained on one GPU (Tesla V100)
#python train.py --clip  --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0
# if trained on two GPUs (Tesla V100)
# python train.py --clip  --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 8 --gpu_ids 0,1

python train.py --model dcomp --channels 64 --knn 6 6 6 --nsamples 10000 5000 2500 --batchSize 4 --gpu_ids 0 -td datasets/kitti360_seq03.hdf5 -tdc datasets/kitti360_acmnet_train.json -bs 16
