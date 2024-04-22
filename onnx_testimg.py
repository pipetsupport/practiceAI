from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("super_resolution.onnx")

img = Image.open("./testimg/cat.jpg")

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

ort_inputs = {ort_session.get_inputs()[0].name: img_y.detach().numpy()}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]


img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# PyTorch 버전의 후처리 과정 코드를 이용해 결과 이미지 만들기
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# 이미지를 저장하고 모바일 기기에서의 결과 이미지와 비교하기
final_img.save("./testimg/cat_superres_with_ort.jpg")