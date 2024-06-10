# How to install
```bash
$ conda create -n cv_final python=3.12.3
$ conda activate cv_final
$ pip install -r requirement.txt
```

# How to execute
```bash
$ ./run.sh
```
`-s`: path to save solutions
`-g`: path to ground truth
`-st`: start frame
`-end`: end frame

# How to evaluation
```bash
$ ./test.sh
```
`-s`: path to solution
`-g`: path to ground truth

# Directory architecture

|----- gt
      |----- 000.png
      |----- ...
      |----- 128.png

|----- homography
      |----- 001_L.npy
      |----- 001_R.npy
      |----- ...
      |----- 128_L.npy
      |----- 128_R.npy

|----- solution
      |----- 001.png, s_001.txt, m_001.txt, H1_001.npy, H2_001.npy
      |----- ...
      |----- 128.png, s_128.txt, m_128.txt, H1_128.npy, H2_128.npy

|----- eval.py, functions.py, gen_frame.py

|----- run.sh, test.sh

|----- readme.md, method.md, requirement.txt, result.txt, output.yuv
