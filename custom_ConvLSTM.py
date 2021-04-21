import tensorflow as tf


class CustomConvLSTM(tf.keras.Model):
    def __init__(self):
        super(CustomConvLSTM, self).__init__()
        self.model = tf.saved_model.load('./models/pm25/1')
        self.labels = None

    # Design your API with 'tf.function' decorator
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def call(self, lstm_input):
        predictions = self.model(lstm_input)
        # Using list didn't work with @tf.function
        # labels = []
        # shape = tf.shape(logits)
        # shape = shape[0]
        # index = tf.constant(0, tf.int32)
        # one = tf.constant(1, tf.int32)
        ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        # while index < shape:
        i = 0
        for prediction in predictions:
            # The integer numbers for labels could be used with a configuration
            if prediction > tf.constant(30, tf.float32): # logits[i]
                ta = ta.write(i, 30)
                # labels.append(1)
            elif prediction > tf.constant(20, tf.float32):
                ta = ta.write(i, 20)
                # labels.append(2)
            elif prediction > tf.constant(10, tf.float32):
                # labels.append(3)
                ta = ta.write(i, 10)
            elif prediction > tf.constant(0, tf.float32):
                ta = ta.write(i, 0)
            i = i + 1
            # index = tf.add(index, one)

        return ta.stack() # labels
        # return tf.strings.as_string(labels), tf.size(logits), logits[0] # logits # class_text


# Load a SavedModel
model_string = CustomConvLSTM()
# Save the image labels as an asset, saved in 'Assets' folder
# You might need to remove the first line, "background"
# Save the model with the re-defined serving_default
tf.saved_model.save(model_string, "./models/pm25/3/")
