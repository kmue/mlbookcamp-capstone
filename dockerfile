FROM public.ecr.aws/lambda/python:3.9

RUN pip install tensorflow
RUN pip install io
RUN pip install urllib
RUN pip install PIL

COPY cnn_2.1 .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]