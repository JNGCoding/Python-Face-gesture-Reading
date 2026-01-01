import cv2
import asyncio
import numpy
from deepface import DeepFace
from typing import Any, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from os import system

# Variables and Initializations
capture: cv2.VideoCapture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not capture.isOpened():
    print("Error: Camera not opened")
    exit(0)
else:
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
print("Camera Initialized!")

# Thread Pool for Face And Expression detection
MultiThreadExecutor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers = 2)
totalTasks: list[asyncio.Task] = []

# Loading all Monkey skins
Monkey_Happy: Optional[cv2.typing.MatLike] = cv2.imread("MonkeyImages/MonkeyHappy.png")
if Monkey_Happy is None:
    print("Failed to load Monkey Happy Image!")
    exit(0)

Monkey_Neutral: Optional[cv2.typing.MatLike] = cv2.imread("MonkeyImages/MonkeyNeutral.png")
if Monkey_Neutral is None:
    print("Failed to load Monkey Neutral Image!")
    exit(0)

Monkey_Sad: Optional[cv2.typing.MatLike] = cv2.imread("MonkeyImages/MonkeySad.png")
if Monkey_Sad is None:
    print("Failed to load Monkey Happy Image!")
    exit(0)

UnknownEmotion: Optional[cv2.typing.MatLike] = cv2.imread("MonkeyImages/UnknownEmotion.png")
if UnknownEmotion is None:
    print("Failed to load Monkey Happy Image!")
    exit(0)

assert isinstance(Monkey_Happy, numpy.ndarray) and isinstance(Monkey_Neutral, numpy.ndarray) and isinstance(Monkey_Sad, numpy.ndarray) and isinstance(UnknownEmotion, numpy.ndarray), "TypeError: Monkey Images extraction failed!"
CurrentEmotion: Optional[cv2.typing.MatLike] = UnknownEmotion

# Functions
async def AnalyzeImage(img: cv2.typing.MatLike) -> List[Dict[str, Any]] | List[List[Dict[str, Any]]] | None:
    global MultiThreadExecutor

    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    try:
        predictions: List[Dict[str, Any]] | List[List[Dict[str, Any]]] = await loop.run_in_executor(MultiThreadExecutor, lambda: DeepFace.analyze(img, actions=["emotion"], detector_backend="opencv"))
        return predictions
    except Exception as e:
        print(e)
        return None

def terminate() -> None:
    global capture

    capture.release()
    cv2.destroyAllWindows()
    exit(0)

# main
async def main() -> None:
    global capture, totalTasks, CurrentEmotion, Monkey_Happy, Monkey_Neutral, Monkey_Sad, UnknownEmotion

    while True:
        # Take Image
        ret, img = capture.read()
        if not ret:
            print("Error: Camera Image Capture failed!!")
            continue

        task: asyncio.Task = asyncio.create_task(AnalyzeImage(img))
        totalTasks.append(task)

        done, pending = await asyncio.wait(totalTasks, timeout = 0, return_when = asyncio.FIRST_COMPLETED)
        for d in done:
            predictions = d.result()
            if predictions and isinstance(predictions, List):
                system("cls")

                # Getting the max value
                max_emotion_name: str = ""
                max_emotion_value: float = 0
                emotions: dict[str, float] = predictions[0]["emotion"]

                for name, val in emotions.items():
                    if max_emotion_value < val:
                        max_emotion_value = val
                        max_emotion_name = name

                available_emotions: dict[str, cv2.typing.MatLike | None] = {
                    "happy": Monkey_Happy,
                    "neutral": Monkey_Neutral,
                    "sad": Monkey_Sad
                }

                if max_emotion_name in available_emotions:
                    CurrentEmotion = available_emotions[max_emotion_name]
                else:
                    CurrentEmotion = UnknownEmotion

                print(f"Emotion: {max_emotion_name}, Prediction: {max_emotion_value}")
            else:
                print("Unknown Exception, Dict not fouund:", predictions)
                CurrentEmotion = UnknownEmotion
                

            totalTasks.remove(d)

        # Show the emotion
        cv2.imshow("Avatar", CurrentEmotion) #type:ignore

        # Exit if we detect 'q'
        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("Avatar", cv2.WND_PROP_VISIBLE) < 1:
            break

# Run main
if __name__ == "__main__":
    asyncio.run(main())
    terminate()
