ln -s ../../hpatches-sequences-release .

python get_kpts_desc.py  --conf_thresh 0.001 --network_version SuperPoint

python get_kpts_desc.py  --network_version SuperPoint_baseline