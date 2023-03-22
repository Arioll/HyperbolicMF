FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

USER root

RUN export PATH="/usr/local/cuda-11.6/bin:$PATH"
RUN export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"

# Install cmake 3.20
RUN apt-get update
RUN apt-get install -y build-essential libssl-dev
# RUN apt-get install python-scipy
# RUN rm -rf tmp
# RUN mkdir tmp
# WORKDIR tmp
# RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
# RUN tar -zxvf cmake-3.20.0.tar.gz
# WORKDIR cmake-3.20.0
# RUN ./bootstrap
# RUN make
# RUN make install
# RUN rm -rf tmp
# WORKDIR /

# Install ripser++
# RUN git clone --recurse-submodules https://github.com/simonzhang00/ripser-plusplus.git
# WORKDIR ripser-plusplus/ripserplusplus
# RUN mkdir build
# WORKDIR build
# RUN cmake .. && make -j$(nproc)

# RUN pip install ripserplusplus

# Install additional libraries
RUN pip install jupyter jupyterlab
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install tqdm==4.63 matplotlib numpy==1.23.1 pandas scipy ipypb scikit-learn numba lightfm openpyxl
RUN pip install cupy
RUN pip install --upgrade git+https://github.com/evfro/polara.git@develop#egg=polara
# RUN pip install git+https://github.com/ArGintum/giotto-ph-ench.git

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8889
