import numpy as np
from flask import Flask, request, jsonify, render_template
from main import recommend_clothing, recommend1_clothing
from flask_bootstrap import Bootstrap
from flask import send_file, make_response
import mysql.connector
import io

app = Flask(__name__)
bootstrap = Bootstrap(app)

db_config = {
    'user': 'root',
    'password': 'password',
    'host': 'localhost',
    'database': 'clothing'
}

def get_image_from_db(image_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    query = "SELECT image FROM images WHERE id = %s"
    cursor.execute(query, (image_id,))

    image = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    # print((image))
    return image

@app.route('/image/<int:image_id>')
def serve_image(image_id):
    image_data = get_image_from_db(image_id)
    if image_data is None:
        return "Image not found", 404

    # Convert binary data to a BytesIO object
    image_io = io.BytesIO(image_data)
    image_io.seek(0)

    # Create a response with the image data
    response = make_response(send_file(image_io, mimetype='image/jpeg'))
    return response


@app.route('/')
def index():
    image_id = 1
    return render_template('homePage.html', image_id=image_id)


@app.route('/submit-form', methods=['POST'])
def submit_form():
    print(f'Request form data: {request.form}')
    if request.method == 'POST':
        body_type = request.form['body_type']
        skin_color = request.form['skin_color']

        # Assuming recommend_clothing and recommend1_clothing return dictionaries
        recommendation1 = recommend_clothing(skin_color, body_type)
        recommendation2 = recommend1_clothing(skin_color, body_type)

        # Convert int32 values to int or float
        recommendation1 = convert_to_serializable(recommendation1)
        recommendation2 = convert_to_serializable(recommendation2)

        # Construct response data
        response_data = {
            'recommendation1': recommendation1,
            'recommendation2': recommendation2
        }

        # Return JSON response with recommendation data
        return jsonify(response_data)

    # Handle errors or redirect as needed
    return jsonify({'error': 'Something went wrong.'})


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: int(value) if isinstance(value, np.int64) else value for key, value in obj.items()}
    # Handle other types if necessary
    return obj


if __name__ == '__main__':
    app.run(debug=True)
