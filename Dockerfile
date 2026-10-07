# LayerWatch worker as an AWS Lambda container image for arm64 (Graviton).
#   docker build --platform linux/arm64 -t layerwatch-worker .
FROM public.ecr.aws/lambda/python:3.12-arm64
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt
COPY layerwatch ${LAMBDA_TASK_ROOT}/layerwatch
COPY server ${LAMBDA_TASK_ROOT}/server
COPY models/layerwatch_unet.onnx ${LAMBDA_TASK_ROOT}/models/layerwatch_unet.onnx
ENV LAYERWATCH_ONNX=${LAMBDA_TASK_ROOT}/models/layerwatch_unet.onnx \
    OPENCV_NUM_THREADS=2
CMD ["server.handler.lambda_handler"]
