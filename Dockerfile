# torch
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    tmux \
    vim \
    libgl1-mesa-glx\
#    _libgcc_mutex=0.1=main \
#    brotlipy=0.7.0=py38h27cfd23_1003 \
#    bzip2=1.0.8=h7b6447c_0 \
#    ca-certificates=2020.10.14=0 \
#    certifi=2020.6.20=py38_0 \
#    cffi=1.14.5=py38h261ae71_0 \
#    chardet=4.0.0=py38h06a4308_1003 \
#    conda-package-handling=1.7.3=py38h27cfd23_1 \
#    cryptography=3.4.7=py38hd23ed53_0 \
#    expat=2.2.10=he6710b0_2 \
#    idna=2.10=pyhd3eb1b0_0 \
#    krb5=1.18.2=h173b8e3_0 \
#    ld_impl_linux-64=2.33.1=h53a641e_7 \
#    libcurl=7.71.1=h20c2e04_1 \
#    libedit=3.1.20191231=h14c3975_1 \
#    libffi=3.3=he6710b0_2 \
#    libgcc-ng=9.1.0=hdf63c60_0 \
#    libssh2=1.9.0=h1ba5d50_1 \
#    libstdcxx-ng=9.1.0=hdf63c60_0 \
#    libuv=1.40.0=h7b6447c_0 \
#    lz4-c=1.9.2=heb0550a_3 \
#    ncurses=6.2=he6710b0_1 \
#    openssl=1.1.1l=h7f8727e_0 \
#    pip=21.0.1=py38h06a4308_0 \
#    pycosat=0.6.3=py38h7b6447c_1 \
#    pycparser=2.20=py_2 \
#    pyopenssl=20.0.1=pyhd3eb1b0_1 \
#    pysocks=1.7.1=py38h06a4308_0 \
#    python=3.8.5=h7579374_1 \
#    readline=8.1=h27cfd23_0 \
#    requests=2.25.1=pyhd3eb1b0_0 \
#    rhash=1.4.0=h1ba5d50_0 \
#    ruamel_yaml=0.15.100=py38h27cfd23_0 \
#    setuptools=52.0.0=py38h06a4308_0 \
#    six=1.15.0=py38h06a4308_0 \
#    sqlite=3.35.4=hdfb4753_0 \
#    tk=8.6.10=hbc83047_0 \
#    tqdm=4.59.0=pyhd3eb1b0_1 \
#    urllib3=1.26.4=pyhd3eb1b0_0 \
#    wheel=0.36.2=pyhd3eb1b0_0 \
#    xz=5.2.5=h7b6447c_0 \
#    yaml=0.2.5=h7b6447c_0 \
#    zlib=1.2.11=h7b6447c_3 \
#    zstd=1.4.5=h9ceee32_0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

#    - absl-py==0.13.0
#    - aiohttp==3.7.4.post0
#    - antlr4-python3-runtime==4.8
#    - asttokens==2.0.5
#    - async-timeout==3.0.1
#    - attrs==21.2.0
#    - backcall==0.2.0
#    - beautifulsoup4==4.10.0
#    - cachetools==4.2.2
#    - click==8.0.1
#    - cmake==3.21.3
#    - colorama==0.4.4
#    - commonmark==0.9.1
#    - configparser==5.0.2
#    - cycler==0.10.0
#    - cython==0.29.24
#    - decorator==5.1.1
#    - docker-pycreds==0.4.0
RUN python -m pip install easydict==1.9
#    - executing==0.8.2
#    - filelock==3.6.0
#    - freetype-py==2.2.0
#    - fsspec==2021.6.1
#    - future==0.18.2
#    - fvcore==0.1.5.post20211023
#    - gdown==4.4.0
#    - gitdb==4.0.7
#    - gitpython==3.1.18
#    - glumpy==1.2.0
#    - google-auth==1.32.1
#    - google-auth-oauthlib==0.4.4
#    - grpcio==1.38.1
#    - h5py==3.6.0
#    - hydra-core==1.1.0
#    - imageio==2.9.0
#    - importlib-resources==5.2.0
#    - iopath==0.1.9
#    - ipython==8.1.0
#    - jedi==0.18.1
#    - joblib==1.0.1
#    - kiwisolver==1.3.1
#    - lightly==1.1.15
#    - lightly-utils==0.0.1
#    - llvmlite==0.38.0
#    - markdown==3.3.4
RUN python -m pip install mathutils==2.81.2
#    - matplotlib==3.4.2
#    - matplotlib-inline==0.1.3
#    - meshio==5.3.0
#    - multidict==5.1.0
#    - networkx==2.6.2
#    - nibabel==3.2.1
#    - numba==0.55.1
#    - numpy==1.20.3
#    - oauthlib==3.1.1
#    - omegaconf==2.1.0
#    - opencv-python==4.5.2.52
#    - packaging==20.9
RUN python -m pip install pandas==1.3.1
#    - parso==0.8.3
#    - pathtools==0.1.2
#    - pexpect==4.8.0
#    - pickleshare==0.7.5
#    - pillow==8.2.0
#    - plyfile==0.7.4
#    - portalocker==2.3.2
#    - progressbar==2.5
#    - promise==2.3
#    - prompt-toolkit==3.0.28
RUN python -m pip install protobuf==3.16.0
#    - psutil==5.8.0
#    - ptyprocess==0.7.0
#    - pure-eval==0.2.2
#    - pyasn1==0.4.8
#    - pyasn1-modules==0.2.8
#    - pydeprecate==0.3.0
#    - pyglet==1.5.16
#    - pygments==2.11.2
#    - pyopengl==3.1.0
#    - pyparsing==2.4.7
#    - pypng==0.0.21
#    - pyrender==0.1.45
#    - python-dateutil==2.8.1
#    - pytorch-lightning==1.3.8
#    - pytz==2021.1
#    - pyyaml==5.4.1
#    - requests-oauthlib==1.3.0
#    - rich==11.1.0
#    - rsa==4.7.2
#    - ruamel-yaml==0.17.16
#    - ruamel-yaml-clib==0.2.6
#    - scikit-learn==0.24.2
RUN python -m pip install scipy==1.6.3
#    - seaborn==0.11.2
#    - sentry-sdk==1.3.1
#    - shortuuid==1.0.1
#    - sklearn==0.0
#    - smmap==4.0.0
#    - soupsieve==2.3.1
#    - stack-data==0.2.0
#    - structlog==21.5.0
#    - subprocess32==3.5.4
#    - tabulate==0.8.9
#    - tensorboard==2.4.1
#    - tensorboard-plugin-wit==1.8.0
RUN python -m pip install tensorboardx==2.2
#    - termcolor==1.1.0
#    - threadpoolctl==2.2.0
#    - torch==1.9.0
#    - torchmetrics==0.4.1
RUN python -m pip install torchsummary
#    - torchvision==0.10.0
#    - traitlets==5.1.1
#    - triangle==20200424
#    - trimesh==3.9.25
#    - ttach==0.0.3
#    - typing-extensions==3.10.0.0
#    - vispy==0.6.6
#    - wandb==0.12.1
#    - wcwidth==0.2.5
#    - werkzeug==2.0.1
#    - wget==3.2
#    - yacs==0.1.8
#    - yarl==1.6.3
#    - zipp==3.5.0

RUN git clone https://github.com/sThalham/TraM6D.git /TraM6D

WORKDIR /TraM6D
