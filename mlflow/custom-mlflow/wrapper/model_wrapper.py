import mlflow
import tensorflow as tf
class ModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to train and use scratch model
    """
    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        ## load model implement
        import json
        with open(context.artifacts["class_names_path"], "r") as fp:
            self.class_names = json.load(fp)
        self.model = tf.keras.models.load_model(context.artifacts["artifact_path"])
 
    def predict(self, context, model_input):
        """Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output. For more information about the pyfunc input/output API, see the Inference API.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        # return your model prediction
        import numpy as np
        probability_model = tf.keras.Sequential([self.model, 
                                         tf.keras.layers.Softmax()])
        prob_predict_result = probability_model.predict(model_input)
        predict_class_name = [self.class_names[np.argmax(item)] for item in prob_predict_result]
        predict_prob_max = [np.max(item) for item in prob_predict_result]
        result = [(class_name,prob) for class_name,prob in zip(predict_class_name,predict_prob_max)]
        return result