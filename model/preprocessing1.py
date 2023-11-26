import torchvision.transforms as T

preprocessing_func = T.Compose(
    [
        T.Resize((150, 150)),
        T.ToTensor()
    ]
)

def preprocess(img):
    return preprocessing_func(img)