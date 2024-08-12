# Use a minimal base image with Python and OpenSlide dependencies
FROM python:3.9-slim

# Set the DEBIAN_FRONTEND environment variable to noninteractive
ENV DEBIAN_FRONTEND="noninteractive"

# Install necessary system dependencies
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    build-essential \
    locales \
    tzdata \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    openslide-tools \
    python3-openslide && \
    echo "en_GB.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

# Set the PATH environment variable
ENV PATH="/root/.local/bin:${PATH}"

# Copy the requirements.txt file to the container
COPY requirements.txt ./

# Uninstall existing versions of NumPy and OpenCV
RUN pip uninstall -y numpy opencv-python-headless

# Install Python packages from requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall numpy opencv-python-headless

# Set the working directory to /workspace
WORKDIR /workspace

# Copy the src directory contents into the container at /workspace/src
COPY src ./src

# Specify the command to run when starting the container
CMD ["bash"]
