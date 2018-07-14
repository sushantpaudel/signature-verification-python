import flask
import matplotlib.image as mpimg
import numpy as np
from keras.models import load_model

app = flask.Flask(__name__)


def create_model():
    model_inside = load_model("model_saved.h5")
    print("Model created!")
    return model_inside


@app.before_first_request
def startup():
    global model
    model = create_model()
    print("database tables created")


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    prediction = flask.request.get_json()
    print(prediction)
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        path = prediction["image"]
        images = [mpimg.imread(path)]
        x_test = np.array(images)
        preds = model.predict(x_test)[0][0]
        if preds == 0:
            data["predictions"] = "Forged"
        else:
            data["predictions"] = "Genuine"
        data["success"] = True

    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
