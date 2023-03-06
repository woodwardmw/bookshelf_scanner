import os

import modal.aio



stub = modal.aio.AioStub(
    "run-boxes",
    image=modal.Image.debian_slim()
    .run_commands("apt update", "apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6")
    .pip_install(
        "roboflow",
        "easyocr",
        "opencv-python==4.1.2.30",
        "requests_html",
    )
)

@stub.function(gpu='any', secret=modal.Secret.from_name("roboflow"))
async def get_boxes(image):
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace().project("books-9nfs8")
    model = project.version(5).model
    results = model.predict(image, confidence=10, overlap=30).json()

    return results


@stub.function()
async def predict(image):
    print("Starting Boxes")
    results = await get_boxes.call(image)
    print("Finished Boxes")
    return results