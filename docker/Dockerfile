FROM python:3.7-slim as vin

# Make a working directory
WORKDIR /root

# Install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENTRYPOINT []
CMD ["/bin/bash"]

FROM vin as vin_note

RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab

EXPOSE 8888/tcp
ENV SHELL /bin/bash
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

