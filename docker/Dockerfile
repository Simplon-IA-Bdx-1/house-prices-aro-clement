FROM continuumio/anaconda3:2019.03
USER root


RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
        libpq-dev \
        build-essential \
        git \
        sudo \
        cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev libboost-all-dev libsdl2-dev swig \
        unzip zip \
    && rm -rf /var/lib/apt/lists/*

RUN conda update -n base conda
RUN conda install -y \
        joblib \
        PyYAML==3.13
RUN conda install -y -c conda-forge \
        pyopengl \
        xgboost \
        nbdime
RUN pip install "urlextract"
RUN pip install "gym[atari,box2d,classic_control]"
RUN pip install "tensorflow-hub"
RUN pip install "tensorflow-serving-api"
RUN pip install "tfx"
RUN pip install "tf-agents-nightly"
RUN pip install "tfds-nightly"
RUN pip install "tfp-nightly"
RUN pip uninstall -y tensorflow
RUN pip uninstall -y tensorboard
RUN pip install "tf-nightly-2.0-preview"
RUN pip install "tb-nightly"
RUN pip install rise
RUN pip install jupyterthemes
RUN pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install
RUN pip install bigml
RUN pip install kaggle
RUN pip install qgrid
RUN pip install --upgrade jupyterthemes
RUN pip install --upgrade notebook
RUN pip install tensorflow
RUN pip install keras
RUN jupyter nbextension enable --py --sys-prefix qgrid
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension 



ARG username
ARG userid


ARG home=/home/${username}
ARG workdir=${home}

RUN adduser ${username} --uid ${userid} --gecos '' --disabled-password \
    && echo "${username} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${username} \
    && chmod 0440 /etc/sudoers.d/${username}

WORKDIR ${workdir}
RUN chown ${username}:${username} ${workdir}

USER ${username}
WORKDIR ${workdir}

RUN git-nbdiffdriver config --enable --global


RUN git config --global diff.jupyternotebook.command 'git-nbdiffdriver diff --ignore-details'


COPY docker/bashrc.bash /tmp/
RUN cat /tmp/bashrc.bash >> ${home}/.bashrc
RUN echo "export PATH=\"${workdir}/docker/bin:$PATH\"" >> ${home}/.bashrc
RUN sudo rm /tmp/bashrc.bash
