# Use an official Python runtime as a parent image
FROM python:3.10.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set any environment variables, if needed
# ENV VARIABLE_NAME=value

# Expose any ports needed by your application
# EXPOSE <port-number>

# Set the command to run your script when the container launches
# CMD ["python", "your_script.py"]
