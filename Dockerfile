FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc

RUN apt-get install pkg-config -y
RUN apt install libhdf5-dev -y
RUN apt-get install gcc -y
RUN pip install --no-binary=h5py h5py

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "test.py" ]