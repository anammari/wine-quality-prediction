# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Install any needed packages specified in Pipfile
RUN pip install --upgrade pip
RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

# Make port 8888 available to the world outside this container, as Jupyter runs on this port
EXPOSE 8888
EXPOSE 5000

# Run jupyter notebook when the container launches, if you want it to be your default app
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]