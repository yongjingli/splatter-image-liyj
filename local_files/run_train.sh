#$dataset_name
#gso (Google Scanned Objects),
#objaverse (Objaverse-LVIS),
#nmr (multi-category ShapeNet),
#hydrants (CO3D hydrants),
#teddybears (CO3D teddybears),
#cars (ShapeNet cars),
#chairs (ShapeNet chairs).

cd ../

# 数据位置的定义在相应的数据类里定义, 如datasets/srn.py 里 SHAPENET_DATASET_ROOT = "/home/pxn-lyj/Egolee/data/shapenet_srn_data"
#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=1,2
export CUDA_VISIBLE_DEVICES=3
dataset_name="cars"
echo $dataset_name
# The first stage is ran with:
#python train_network.py +dataset=$dataset_name

#Run second stage with
#python train_network.py +dataset=$dataset_name +experiment=$lpips_experiment_name

#To train a 2-view model run:
python train_network.py +dataset=$dataset_name cam_embd=pose_pos data.input_images=2 opt.imgs_per_obj=5

