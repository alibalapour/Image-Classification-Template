import argparse
import datetime
from fastapi import FastAPI, UploadFile, File
import PIL.Image as Image
import uvicorn
import uuid
from model import ResNetInterface

app = FastAPI()


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', default='./model.pth', type=str)
    parser.add_argument('--input-size', default=32, type=int)
    parser.add_argument('--model-name', default="ResNet", type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num-classes', default=10, type=int)

    arguments = parser.parse_args()
    return arguments


args = get_parser()
interface = ResNetInterface(device=args.device, input_size=args.input_size, model_path=args.model_path)


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    img_path = f"{file.filename}"
    with open(img_path, "wb") as f:
        f.write(contents)

    img = Image.open(img_path)
    p_label = interface.inference(img)
    return {"predicted class": p_label, 'date': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}


if __name__ == '__main__':
    uvicorn.run("app:app", reload=True)
