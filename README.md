# pointnet2_ops_lib
This mainly involves FPS in point cloud processing, as well as kernel density estimation. This code repository primarily focuses on certain functionalities of the 'pointnet2_ops_lib' operator, providing methods for function calls and implementation within the PyTorch framework.

Added a feature **'pts_cnt'** to record the number of neighboring points within the spherical radius of the current query point in **'pointnet2_ops_lib\pointnet2_ops\_ext-src\src\ball_query.cpp'**.

# Usage
## Requirements
PyTorch >= 1.7.0<br>
python >= 3.7<br>
CUDA >= 9.0<br>
GCC >= 4.9<br>
torchvision<br>

    cd pointnet2_ops_lib/
    python setup.py install
    
    python FPS_kde.py
    
