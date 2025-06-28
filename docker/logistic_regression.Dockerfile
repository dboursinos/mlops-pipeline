FROM debian:bookworm-20250520-slim

# Set the working directory
WORKDIR /app

# Install dependencies for building Python
RUN apt-get update && apt-get install -y --no-install-recommends \
  software-properties-common \
  build-essential \
  zlib1g-dev \
  libncurses5-dev \
  libgdbm-dev \
  libnss3-dev \
  libssl-dev \
  libreadline-dev \
  libffi-dev \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.13
RUN wget https://www.python.org/ftp/python/3.13.4/Python-3.13.4.tar.xz && \
  tar -xf Python-3.13.4.tar.xz && \
  cd Python-3.13.4 && \
  ./configure --enable-optimizations && \
  make -j 4 && \
  make install && \
  cd .. && \
  rm -rf Python-3.13.4

# Set python3.13 as default python
RUN ln -s /usr/local/bin/python3.13 /usr/local/bin/python
RUN ln -s /usr/local/bin/pip3.13 /usr/local/bin/pip

# Copy the requirements file
COPY ./docker/requirements/requirements.sklearn.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Configure DVC (You may need to set AWS credentials as environment variables)
# Example:
# ENV AWS_ACCESS_KEY_ID=<your_access_key>
# ENV AWS_SECRET_ACCESS_KEY=<your_secret_key>


# Copy the training script
COPY ../src/training/train.py .

# Command to run the training script
CMD ["python", "train.py"]
