## Installation

### Requirements:
- PyTorch 1.1.0
- torchvision
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark -y
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
#conda install -c pytorch pytorch-nightly torchvision cudatoolkit
pip install torch==1.1.0
pip install torchvision==0.3.0

export INSTALL_DIR=$PWD

git clone https://github.com/hzhupku/DCNet

# install pycocotools
# modify "cocoeval.py" to store raw results in "~/coco_result.txt"
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex-96b017a8b40f137abb971c4555d61b2fcbb87648
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 96b017a
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
cd DCNet
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR
```
