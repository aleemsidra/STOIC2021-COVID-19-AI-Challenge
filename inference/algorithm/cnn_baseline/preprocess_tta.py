
import SimpleITK as sitk
import numpy as np 
import torch
from torchvision import transforms
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.augmentations.crops.transforms as albut

tensorv2 = ToTensorV2()


def apply_transform(images, transform):
    if transform is None:
        return images
    if isinstance(transform, albu.ReplayCompose):
        results = []
        result = transform(image=images[0])
        results.append(result['image'])
        replay = result['replay']
        for image in images[1:]:
            results.append(transform.replay(replay, image=image)['image'])
        return results
    return [transform.apply(image=image)['image'] for image in images]


def apply_tta_custom(images, transform_dict):

    # print('transform ', transform_dict)
    # print('images ', [images[0].shape, len(images)])
    images = [i.permute(1,2,0).numpy() for i in images]
    # print('images ', [images[0].shape, len(images)])

    results = []

    transform = transform_dict['transform']
    kwargs = transform_dict['params']
    for image in images:
        if isinstance(transform, albu.Compose):
            out = image
            for i in transform.transforms:
                # print(['i', i, out.shape, type(out[0,0,0]), out.min(), out.max()])
                # out = out
                out = i.apply(img=out, **kwargs)
            
        else:
            out = transform.apply(img=image, **kwargs)
        out = tensorv2.apply(out)
        results.append(out)
        # asd
    return results

test_transform = [
    albu.Resize(256, 256),
    # albu.CenterCrop(224, 224),
    albu.Normalize(mean=0.5, std=0.5),
    ToTensorV2(),
]

test_transform = albu.ReplayCompose(test_transform)


def tta_first(input_image: sitk.Image):

    output_list =[]
    w = 1500
    l= -600

    #sub sample
    # input_image = sitk.ReadImage(img_name + ".mha")
    # print('input_image.shape ', input_image.GetSize())
    #slice = int(np.round(input_image.GetSize()[2]/33))
    slice_ = int(np.floor(input_image.GetSize()[2]/33))
    sub_sample = input_image[:, :, 0:input_image.GetDepth():slice_]
    sub_sample = sitk.GetArrayFromImage(sub_sample) 
    sub_sample = sub_sample[:32, :, :]
    # print(sub_sample.shape)
    # windowing
    x = l + w/2
    y = l - w/2
    sub_sample[sub_sample > x] = x
    sub_sample[sub_sample < y] = y
    chanel = (sub_sample - np.min(sub_sample))/(np.max(sub_sample) - np.min(sub_sample))
    # print('chanel ', [chanel.shape, chanel.min(), chanel.max()])
    # asd
    chanel = chanel*255
    chanel = chanel.astype(np.uint8)
    for img in range(chanel.shape[0]):
        img
        image =np.asarray(chanel[img,:,:])
        # print('image ', image.shape)
        image = np.expand_dims(image, 2)
        image = np.repeat(image, 3, axis=2)
        # print('image,shape ', image.shape)
        output_list.append(image)

    output_list = apply_transform(output_list, test_transform)
    return output_list

def preprocess_tta(output_list, transform

               ) -> torch.Tensor:
   
    output_list = apply_tta_custom(output_list, transform)
    image = torch.stack(output_list)
    # print(image.shape)
    return image