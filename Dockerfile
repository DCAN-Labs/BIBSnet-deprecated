ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3
FROM ${FROM_IMAGE_NAME} 

ENV nnUNet_raw_data_base="/opt/nnUNet/nnUNet_raw_data_base/"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet /results

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

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users bibsnet
WORKDIR /home/bibsnet
ENV HOME="/home/bibsnet" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
RUN cd .. && \
    git clone https://github.com/ylugithub/nnUNet.git && \
    cd nnUNet && \
    pip install -e .

# set up nnU-Net specific things
ENV nnUNet_raw_data_base="/output"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet

RUN wget https://s3.msi.umn.edu/CABINET_data/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip -O /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip && \
    wget https://s3.msi.umn.edu/CABINET_data/Task551.zip -O /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/Task551.zip && \
    wget https://s3.msi.umn.edu/CABINET_data/Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.zip -O /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.zip && \
    wget https://s3.msi.umn.edu/CABINET_data/Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.zip -O /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.zip
RUN cd /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet && \
    unzip -qq Task551.zip && \
    unzip -qq Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip && \
    unzip -qq Task514_BCP_ABCD_Neonates_SynthSeg_T1Only.zip && \
    unzip -qq Task515_BCP_ABCD_Neonates_SynthSeg_T2Only.zip
COPY run.py /home/bibsnet/run.py
RUN cd /home/bibsnet/ && chmod 555 run.py
RUN chmod 666 /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/3d_fullres/Task*/nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json

ENTRYPOINT ["/home/bibsnet/run.py"]
