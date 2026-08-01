cd /home/longlive/libero2lerobot
source ./.venv/bin/activate
python -m pdb ./libero_rlds_converter.py --data-dir ~/stack_cube_datasets/hdf5/ --repo-id stack-cube/franka --private --use-videos --clean-existing --verbose