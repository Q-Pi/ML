FROM ubuntu

COPY . /

RUN apt-get -y update && \
apt-get -y install python3  && \
apt-get -y install python3-pip  && \
apt-get -y install git  && \
pip3 install --upgrade pip  && \
pip3 install -r /requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3", "api.py"]
