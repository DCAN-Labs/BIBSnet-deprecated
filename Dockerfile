ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3
FROM ${FROM_IMAGE_NAME} 

ENV nnUNet_raw_data_base="/opt/nnUNet/nnUNet_raw_data_base/"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet /results
COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /results
RUN cd /results && unzip -qq Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip

<<<<<<< HEAD
ADD ./requirements.txt .
RUN pip install --disable-pip-version-check -r requirements.txt
RUN pip install monai==0.8.0 --no-dependencies
RUN pip uninstall -y torchtext

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip -qq awscliv2.zip
RUN ./aws/install
RUN rm -rf awscliv2.zip aws

WORKDIR /workspace/nnunet_pyt
ADD . /workspace/nnunet_pyt
RUN cd /workspace/nnunet_pyt
ENTRYPOINT ["main.py"] 
=======
# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    apt-utils \
                    autoconf \
                    build-essential \
                    bzip2 \
                    ca-certificates \
                    curl \
                    gcc \
                    git \
                    gnupg \
                    libtool \
                    lsb-release \
                    pkg-config \
                    unzip \
                    wget \
                    xvfb \
		    zlib1g && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
#RUN apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
#RUN apt-get update && \
#    apt-get -y install cuda
#COPY container_materials/cudnn-linux-x86_64-8.3.2.44_cuda10.2-archive/include/cudnn*.h /usr/local/cuda/include/ 
#COPY container_materials/cudnn-linux-x86_64-8.3.2.44_cuda10.2-archive/lib/libcudnn* /usr/local/cuda/lib64/
#RUN chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users bibsnet
WORKDIR /home/bibsnet
ENV HOME="/home/bibsnet" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
# Install python 3.8.3 version of miniconda
#RUN echo "Installing miniconda ..." && \
#    curl -sSLO https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
#    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /usr/local/miniconda && \
#    rm Miniconda3-py38_4.10.3-Linux-x86_64.sh 
#RUN ln -s /usr/local/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /usr/local/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate base" >> ~/.bashrc
#RUN echo ". /usr/local/miniconda/etc/profile.d/conda.sh" >> $HOME/.bashrc && \
#    echo "conda activate base" >> $HOME/.bashrc
# create conda environment
#ENV PATH=$PATH:/usr/local/miniconda/condabin:/usr/local/miniconda/bin:/usr/local/cuda/bin \
#    CPATH="/usr/local/miniconda/include:$CPATH" \
#    LD_LIBRARY_PATH="/usr/local/miniconda/lib:$LD_LIBRARY_PATH" \
#    LANG="C.UTF-8" \
#    LC_ALL="C.UTF-8" \
#    PYTHONNOUSERSITE=1 

#RUN conda install -y pip numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
#RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
#ENV CUDACXX=/usr/local/miniconda/bin \
#    CMAKE_PREFIX_PATH=/usr/local/miniconda/bin \
#    CUDNN_LIB_DIR=/usr/local/cuda/lib64 \
#    CUDNN_INCLUDE_DIR=/usr/local/cuda/include \
#    CUDNN_LIBRARY=/usr/local/cuda/lib64
RUN cd .. && \
    git clone https://github.com/MIC-DKFZ/nnUNet.git && \
    cd nnUNet && \
    pip install -e .

ENV nnUNet_raw_data_base="/output"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet
RUN cd /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet && unzip -qq Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip
COPY run.py /home/bibsnet/run.py
RUN cd /home/bibsnet/ && chmod 555 run.py

ENTRYPOINT ["/home/bibsnet/run.py"] 


>>>>>>> in-house-attempt
