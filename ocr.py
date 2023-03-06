# from easyocr import Reader
# import cv2
import modal.aio


stub = modal.aio.AioStub(
    "run-ocr",
    image=modal.Image.debian_slim()
    .run_commands("apt update", "apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6")
    .pip_install(
       "easyocr",
       "opencv-python==4.1.2.30",
    )
)

volume = modal.SharedVolume().persist("ocr_model_vol")
CACHE_PATH = "/root/model_cache"

@stub.function(gpu='any', shared_volumes={CACHE_PATH: volume})
async def get_text(image, rotated=0):
    from easyocr import Reader
    import cv2
    if rotated == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotated == -1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotated == 2:
        image = cv2.rotate(image, cv2.ROTATE_180)
    reader = Reader(['en'],gpu=True, model_storage_directory=CACHE_PATH)

    results = reader.readtext(image)
    
    return results


@stub.function
async def predict(image):
    print("Starting OCR")
    results = await get_text.call(image)
    best_results = [result for result in results if len(result[1]) * result[2] ** 2 > 0.9]
    print(results)
    print(sum([result[2] for result in best_results]) / len(results))
    results = await get_text.call(image, rotated=1)
    best_results = [result for result in results if len(result[1]) * result[2] ** 2 > 0.9]
    print(results)
    print(sum([result[2] for result in best_results]) / len(results))
    results = await get_text.call(image, rotated=-1)
    best_results = [result for result in results if len(result[1]) * result[2] ** 2 > 0.9]
    print(results)
    print(sum([result[2] for result in best_results]) / len(results))
    results = await get_text.call(image, rotated=2)
    best_results = [result for result in results if len(result[1]) * result[2] ** 2 > 0.9]
    print(results)
    print(sum([result[2] for result in best_results]) / len(results))

    # Convert the coordinates so they're all on the same system
    # Calculate the average confidence for each orientation
    # Keep the two adjacent orientations that are best. You don't want an orientation and its opposite, since upside-down is fairly high scoring but wrong.

    print("Finished OCR")
    return results
