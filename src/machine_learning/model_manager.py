"""A module for managing Keras and TFLite models and performing predictions"""

from tensorflow.keras.models import load_model
import tensorflow as tf


class ModelManager:
    """
    Class for managing and interacting with machine learning models in either
    Keras or TensorFlow Lite (TFLite) format.
    """

    def __init__(self, type):
        """Initialize the ModelManager with the specified model type

        Args:
            type (str): Type of the model to manage, 'keras' or 'tflite'
        """

        self.type = type

    def load(self, path):
        """Load the specified model from the given path and initialize

        Args:
            path (str): The file path to the model
        """

        if self.type == "keras":
            self.model = load_model(path)

        if self.type == "tflite":
            self.interpreter = tf.lite.Interpreter(model_path=path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()


    def predict(self, img_object):
        """Make a prediction using the loaded model on a provided image object

        Args:
            img_object (numpy array): The input data to be predicted. 
                                Must match the model's expected input shape.

        Returns:
            numpy array: The prediction output from the model
        """

        if self.type == "keras":
            return self.model.predict(img_object, verbose=0)

        if self.type == "tflite":
            input_index = self.input_details[0]["index"]
            output_index = self.output_details[0]["index"]
            self.interpreter.set_tensor(input_index, img_object)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_index)
            return output_data
