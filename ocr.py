# from easyocr import Reader
# import cv2
import modal.aio


stub = modal.aio.AioStub(
    "run_ocr",
    image=modal.Image.debian_slim()
    .run_commands("apt update", "apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6")
    .pip_install(
       "easyocr",
       "opencv-python==4.1.2.30",
    )
)

@stub.function(gpu='any')
async def get_text(image, rotated=0):
    from easyocr import Reader
    import cv2
    if rotated == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    reader = Reader(['en'],gpu=True)

    results = reader.readtext(image)
    
    return results


@stub.function
async def predict(image):
    results = await get_text.call(image)
    return results
