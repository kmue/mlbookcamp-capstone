import bentoml

import numpy as np
from bentoml.io import Image
from bentoml.io import JSON

runner = bentoml.keras.get("kitchenware_classifier:latest").to_runner()

svc = bentoml.Service("kitchenware_classifier_service", runners=[runner])

@svc.api(input=Image(), output=JSON())

async def predict(img):

    from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

    img = img.resize((299, 299))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = await runner.async_run(arr)
    classes = ['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']
    return classes[preds.argmax(axis=1)[0]]
