ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3
FROM ${FROM_IMAGE_NAME} 

ENV nnUNet_raw_data_base="/opt/nnUNet/nnUNet_raw_data_base/"
ENV nnUNet_preprocessed="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models"

RUN mkdir -p /opt/nnUNet/nnUNet_raw_data_base/ /opt/nnUNet/nnUNet_raw_data_base/nnUNet_preprocessed /opt/nnUNet/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet /results
COPY trained_models/Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip /results
RUN cd /results && unzip -qq Task512_BCP_ABCD_Neonates_SynthSegDownsample.zip

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
