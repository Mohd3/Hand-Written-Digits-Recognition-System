import gradio as gr
from tensorflow.keras import models

model = models.load_model('model.h5')

def func(image):
        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255
        prediction = model.predict(image)

        return {str(i): float(prediction[0][i]) for i in range(10)}


ui = gr.Interface(
    fn = func,
    inputs="sketchpad",
    outputs = gr.Label(num_top_classes=3),
    live = True
)

ui.launch(share=True)
    