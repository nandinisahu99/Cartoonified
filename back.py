from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

def cartoonify_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    #CREATE EDGE MASK----------------------------------
    line_size,blur_value=3,7
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray_blur=cv2.medianBlur(gray,blur_value)
    edge=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_value)

    #REDUCE THE COLOR PALETTE-----------------------------
    k=13
    # Transform the image
    data=np.float32(img).reshape((-1,3))

    # Determine Criteria
    criteria=(cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER,20,0.001)

    # Implementing K-Means
    ret,label,center=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    result=center[label.flatten()]
    result=result.reshape(img.shape)

    #REDUCE THE NOISE---------------------------------------------

    color = cv2.bilateralFilter(result,d=3,sigmaColor=200,sigmaSpace=200)

    #COMBINE EDGE MASK WITH THE QUNTIZING-------------------------
    cartoon = cv2.bitwise_and(color, color, mask=edge)

    # cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    return cartoon


@app.route('/cartoonify', methods=["POST"])
def cartoonify():
    try:
        data = request.get_json()
        # print(data)
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = data['image'].split(",")[1]

        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        cartoonified_image = cartoonify_image(image)

        _, buffer = cv2.imencode('.png', cartoonified_image)
        cartoonified_image_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'cartoonified_image': 'data:image/png;base64,' + cartoonified_image_str})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
