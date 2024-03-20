# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install ffmpeg, libgl1 (for opencv), and imagemagick
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Relax ImageMagick policies
RUN sed -i '/<policy domain="coder" rights="none" pattern="MVG" \/>/d' /etc/ImageMagick-6/policy.xml && \
    sed -i '/<policy domain="path" rights="none" pattern="@*" \/>/d' /etc/ImageMagick-6/policy.xml


# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
