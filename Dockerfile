# Base image with Python 3.10 (compatible with most libraries)
FROM python:3.10-slim

# Set environment variables to avoid interactive prompts during install
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Create a virtual environment and activate it, then install dependencies
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8000

# Specify the default command to run the app
CMD ["/opt/venv/bin/python", "app.py"]
