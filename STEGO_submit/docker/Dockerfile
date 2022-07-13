FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/pytorch-1.9

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    pytorch=1.9 \
    torchvision=0.10.0 \
    cudatoolkit=11.1 \
    numpy=1.19 \
    scipy=1.6.3 \
    pip=22.0.4 \
    cython \
    -c pytorch -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
                'psutil>=5.8,<5.9' \
                'azureml-core==1.41.0' \
                'azureml-defaults==1.41.0' \
                'azureml-telemetry==1.41.0' \
                'azureml-train-restclients-hyperdrive==1.41.0' \
                'azureml-train-core==1.41.0' \
                'azureml-mlflow==1.41.0' \
                'pillow==8.3.1' \
                'matplotlib' \
                'cython==0.29.24' \
                'tqdm==4.64.0' \
                'pytorch-lightning==1.6.3' \
                'omegaconf==2.1.2' \
                'hydra-core==1.1.2' \
                'wget==3.2' \
                'seaborn==0.11.2'

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH