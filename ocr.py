# from easyocr import Reader
# import cv2
import modal.aio
import asyncio


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
    height, width, _ = image.shape
    if rotated == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotated == 3:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotated == 2:
        image = cv2.rotate(image, cv2.ROTATE_180)
    reader = Reader(['en'],gpu=True, model_storage_directory=CACHE_PATH)

    results = reader.readtext(image)

    if rotated == 1:
        results = [(
                    ([
                        [width - result[0][3][1], result[0][3][0]],
                        [width - result[0][0][1], result[0][0][0]],
                        [width - result[0][1][1], result[0][1][0]], 
                        [width - result[0][2][1], result[0][2][0]],
        ]),
                    result[1],
                    result[2])
                       for result in results]
    if rotated == 3:
        results = [(
                    ([
                        [result[0][1][1], height - result[0][1][0]], 
                        [result[0][2][1], height - result[0][2][0]],
                        [result[0][3][1], height - result[0][3][0]],
                        [result[0][0][1], height - result[0][0][0]],
        ]),
                    result[1],
                    result[2])
                       for result in results]
    if rotated == 2:
        results = [(
                    ([
                        [width - result[0][2][0], height - result[0][2][1]],
                        [width - result[0][3][0], height - result[0][3][1]],
                        [width - result[0][0][0], height - result[0][0][1]],
                        [width - result[0][1][0], height - result[0][1][1]],
                        ]),
                    result[1],
                    result[2])
                       for result in results]
    return results


async def process_image(image, rotation):
    print(f'Starting OCR for rotation {rotation}')
    results = await get_text.call(image, rotated=rotation)
    best_results = [result for result in results if len(result[1]) * result[2] ** 2 > 0.9]
    print(f'Finishing OCR for rotation {rotation}')
    return (best_results, sum([result[2] for result in best_results]) / len(results)) if results else ([], 0)


@stub.function()
async def run_process_image(image):
    print("Starting OCR")
    combined_results = {}
    scores = {}

    tasks = []
    for rotation in range(4):
        tasks.append(process_image(image, rotation))

    results = await asyncio.gather(*tasks)

    for rotation, (best_results, score) in enumerate(results):
        combined_results[rotation] = best_results
        scores[rotation] = score
        print(best_results)
        print(score)

    # scores[2] = sum([result[2] for result in best_results]) / len(results)
    # print(sum([result[2] for result in best_results]) / len(results))

    # Take the best scoring rotation and the best one adjacent to it
    highest_key = max(scores, key=scores.get)
    print(highest_key)
    if scores[(highest_key + 1) % 4] > scores[(highest_key + 3) % 4]:
        second_highest_key = (highest_key + 1) % 4
    else:
        second_highest_key = (highest_key + 3) % 4
    print(second_highest_key)
    results = combined_results[highest_key] + combined_results[second_highest_key]
    
    # This needs tidying up and making async!
    # Also, I think we should choose the rotations at the book level, since different books can be rotated differently

    print("Finished OCR")
    return results

@stub.function()
async def predict(image):
    results = await run_process_image.call(image)
    return results
