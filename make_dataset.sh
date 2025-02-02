python3 multihost_runner.py --TPU_PREFIX=node-1 --COMMAND="bash setup.sh" --INTERNAL_IP=true
python3 multihost_runner.py --TPU_PREFIX=node-1 --COMMAND="pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" --INTERNAL_IP=true
python3 multihost_runner.py --TPU_PREFIX=node-1 --COMMAND="pip install 'git+https://github.com/flyingblackshark/DAC-JAX.git'" --INTERNAL_IP=true
python3 multihost_runner.py --TPU_PREFIX=node-1 --COMMAND="mkdir ~/bucket && gcsfuse fbs-us2 ~/bucket" --INTERNAL_IP=true
python3 multihost_runner.py --TPU_PREFIX=node-1 --COMMAND="python3 MaxText/hf_dac_encode_mls_eng_dataset.py" --INTERNAL_IP=true