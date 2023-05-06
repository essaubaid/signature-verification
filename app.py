from flask import Flask, request
from extract_signature import extractor

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save('newPic.jpg')
    extractor('newPic.jpg', 'output')
    return 'File saved as newPic.jpg'


if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask

# app = Flask(__name__)


# @app.route('/')
# def index():
#     return 'Hello World'


# if __name__ == '__main__':
#     app.run(debug=True)
