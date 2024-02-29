from django.shortcuts import render
import numpy as np
from PIL import Image
from keras.models import load_model



def index(request):
    return render(request, 'home.html')


def ResultText(request):
    if request.method == 'POST' and request.FILES['imageInput']:
        img_input = Image.open(request.FILES["imageInput"])
        # img_input.show()
        model = load_model('mnist.h5')
        img = img_input.resize((28, 28))
        img = img.convert('L')
        img = np.array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        y_p = model.predict(img)
        array = (np.reshape(img, (28, 28)) * 255).astype(np.uint8)
        data = Image.fromarray(array, 'L')
        data.save(r'C:\Users\rashi\PycharmProjects\Digit_Recognition\Digit_Recognition\static\images\img.png')

    return render(request, "result.html", {'Entered_text': np.argmax(y_p)})



