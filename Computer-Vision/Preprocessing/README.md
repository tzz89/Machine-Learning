## Libraries
1. Albumentations
2. Torchvision



## Torchvision.transform.v2
In torchvision, there are many prebuilt augmentations in torchvision.transform.v2 as well as torchvision.transform.functional that does not have rng. we can create our own class and overwrite __call__ function to use with V2.compose

```python
from torchvision.transforms.functional import adjust_brightness

def build_transform():
    transform = v2.Compose(
        [
            v2.RandomApply([v2.RandomErasing(scale=0.06)]),
            v2.RandomApply([RandomBrightness(0.1)]),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]
    )

    return transform

```
To convert torch tensor back to PILLOW image, we can use torchvision.transforms.functional to_pil_image(tensor)