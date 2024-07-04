import datetime, io, torch
from PIL import Image
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

@app.route("/", methods=["GET", "POST"])
def home():
    global model
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)

        analysis_results = results.pandas().xyxy[0].to_dict(orient='records')
        for result in analysis_results:
            result['confidence'] = round(result['confidence'] * 100, 2)
            result['bbox'] = [int(result['xmin']), int(result['ymin']), int(result['xmax']), int(result['ymax'])]

        return render_template("result.html", image_name=now_time + ".png", analysis_results=analysis_results)  # Pass image name, not full path

    return render_template("index.html")  # Handle GET requests


if __name__ == "__main__":
    model = torch.hub.load('yolov5', 'custom', path='model/model.pt', source='local')
    app.run(host='0.0.0.0', port=5000)
