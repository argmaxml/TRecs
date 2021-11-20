FROM pypy:3


COPY ./src /app/src/
COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r /app/requirements.txt
WORKDIR /app/src


EXPOSE 5000
CMD [ "pypy3", "./endpoint.py" ]
