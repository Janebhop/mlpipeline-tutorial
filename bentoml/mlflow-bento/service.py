import bentoml
from bentoml.io import Image,NumpyNdarray
import PIL
import mlflow
import numpy as np
mlflow.set_tracking_uri('http://localhost:5000')
model_url = 'models:/bentoml-mlflow/Staging'
class ClothingClassificationRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False
    def __init__(self):
        self.model = mlflow.pyfunc.load_model(model_uri=model_url)
    @bentoml.Runnable.method(batchable=False)
    def predict(self, input_image: Image) -> list:
        img = PIL.ImageOps.grayscale(input_image)
        img = img.resize((28,28))
        np_img = np.array(img).astype(np.uint8)
        tensor_input = np.expand_dims(np_img, axis = 0)
        result = self.model.predict(tensor_input)
        return result
#create runner
clothing_classification_runner = bentoml.Runner(ClothingClassificationRunner)
classification = bentoml.Service(
    #service_name
    "cloting_classification_service",
    #set runner
    runners=[clothing_classification_runner],
    )
#build api
#set input to image file and output to array
@classification.api(input=Image(), output=NumpyNdarray())
def image_classify(input_data: PIL.Image) -> np.array:
    predict_result = clothing_classification_runner.predict.run(input_data)
    return np.array(predict_result)