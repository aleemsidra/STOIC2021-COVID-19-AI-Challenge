from typing import Dict
from pathlib import Path
import SimpleITK
import torch
import os
import json

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, device
from algorithm.cnn_baseline.preprocess import preprocess
from algorithm.cnn_baseline.preprocess_tta import preprocess_tta, tta_first

from algorithm.cnn_baseline.trainers.example_model import ExampleModel
from algorithm.cnn_baseline.utils.config import process_config
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.augmentations.crops.transforms as albut


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/"),
            file_filters={'/input/images/ct/':'*.mha'}
        )

        # load model

        print('inside process getcwd ', os.getcwd())
        config_path = "algorithm/cnn_baseline/configs/config.json"
        
        config = process_config(config_path)
        self.config = config

        

        self.model = ExampleModel(config)
        self.model = self.model.to(device)
        self.model.load()
        self.model = self.model.eval()

        config['pretrained_file'] = f'mobile_{2}.pth.tar'

        self.model1 = ExampleModel(config)
        self.model1 = self.model1.to(device)
        self.model1.load()
        self.model1 = self.model1.eval()

        config['pretrained_file'] = f'mobile_{3}.pth.tar'

        self.model2 = ExampleModel(config)
        self.model2 = self.model2.to(device)
        self.model2.load()
        self.model2 = self.model2.eval()

        config['pretrained_file'] = f'mobile_{4}.pth.tar'

        self.model3 = ExampleModel(config)
        self.model3 = self.model3.to(device)
        self.model3.load()
        self.model3 = self.model3.eval()

        config['pretrained_file'] = f'mobile_{5}.pth.tar'

        self.model4 = ExampleModel(config)
        self.model4 = self.model4.to(device)
        self.model4.load()
        self.model4 = self.model4.eval()



        mob_model_list = [self.model, self.model1, self.model2, self.model3, self.model4]


        print('loading r18 models')
        config_path = "algorithm/cnn_baseline/configs/config_resnet.json"
        
        config = process_config(config_path)
        self.config_resnet = config

        

        self.model_r = ExampleModel(config)
        self.model_r = self.model_r.to(device)
        self.model_r.load()
        self.model_r = self.model_r.eval()

        config['pretrained_file'] = f'resnet_{2}.pth.tar'

        self.model1_r = ExampleModel(config)
        self.model1_r = self.model1_r.to(device)
        self.model1_r.load()
        self.model1_r = self.model1_r.eval()

        config['pretrained_file'] = f'resnet_{3}.pth.tar'

        self.model2_r = ExampleModel(config)
        self.model2_r = self.model2_r.to(device)
        self.model2_r.load()
        self.model2_r = self.model2_r.eval()

        config['pretrained_file'] = f'resnet_{4}.pth.tar'

        self.model3_r = ExampleModel(config)
        self.model3_r = self.model3_r.to(device)
        self.model3_r.load()
        self.model3_r = self.model3_r.eval()

        config['pretrained_file'] = f'resnet_{5}.pth.tar'

        self.model4_r = ExampleModel(config)
        self.model4_r = self.model4_r.to(device)
        self.model4_r.load()
        self.model4_r = self.model4_r.eval()

        res_model_list = [self.model_r, self.model1_r, self.model2_r, self.model3_r, self.model4_r]

        self.model_list = mob_model_list + res_model_list

        # self.model_list = [self.model, self.model1]


        rotate = albu.Compose([albu.CenterCrop(224, 224),albu.SafeRotate(limit=20, border_mode=0, always_apply=True, p=1)])


        h,w = 224,224
        h_crop, w_crop = 166, 166
        crops4 = [
                albu.Compose([albut.Crop(x_min=0, y_min=0, x_max=h_crop, y_max=w_crop), albu.Resize(224,224)]),
                albu.Compose([albut.Crop(x_min=h-h_crop, y_min=w-w_crop, x_max=h, y_max=w),albu.Resize(224,224)]),
                albu.Compose([albut.Crop(x_min=h-h_crop, y_min=0, x_max=h, y_max=w_crop),albu.Resize(224,224)]),
                albu.Compose([albut.Crop(x_min=0, y_min=w-w_crop, x_max=h_crop, y_max=w),albu.Resize(224,224)]),
        ]

        # transform_dict =
        tta_horizontalflip = [{'transform':albu.HorizontalFlip(always_apply=True, p=1), 'params':{}}]
        tta_rotate = [{'transform':rotate, 'params':{'angle':i}} for i in (-5,-10,+5)]

        tta_center = [{'transform':albu.CenterCrop(224,224), 'params':{}}]

        tta_crop = [{'transform':crop, 'params':{}} for crop in crops4]

        self.tta_transforms = tta_crop + tta_center + tta_rotate
        # self.tta_transforms = tta_crop + tta_center


        # self.tta_transforms = None

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        input_list = tta_first(input_image)

        ensemble_outs = []
        for i in range(len(self.model_list)):

            output = self.predict_tta(input_list, model_number=i)
            ensemble_outs.append(output)

        ensemble_probs = torch.stack(ensemble_outs)
        # print('ensemble_probs ', ensemble_probs.shape)
        # print(ensemble_probs)

        output = ensemble_probs.mean(0)

        # print(probs_tta)
        # asd
        # print('here')
        # print('inside output ', output)

        return {
            COVID_OUTPUT_NAME: output[1].item(),
            SEVERE_OUTPUT_NAME: output[0].item()
        }


    # def predict(self, *, input_image: SimpleITK.Image) -> Dict:
    #     # pre-processing
    #     input_image = preprocess(input_image)
    #     input_image = input_image.cuda()

    #     # run model
    #     input_image = input_image.unsqueeze(0)
    #     with torch.no_grad():
    #         output = torch.sigmoid(self.model(input_image))
    #     # print('here')
    #     # print('inside output ', output)

    #     return {
    #         COVID_OUTPUT_NAME: output[1].item(),
    #         SEVERE_OUTPUT_NAME: output[0].item()
    #     }

    def predict_tta(self, input_list, model_number):

        model = self.model_list[model_number]

        # print('model on cuda ', next(model.parameters()).is_cuda)

        tta_outs = []
        
        input_image = torch.stack(input_list)
        input_image = input_image.cuda()
        input_image = input_image.unsqueeze(0)
        # print('input_image', input_image.shape)
        # asd
        with torch.no_grad():
            output = torch.sigmoid(model(input_image))
        tta_outs.append(output)

        if self.tta_transforms is not None:
            for transform in self.tta_transforms:
                input_image = preprocess_tta(input_list, transform)
                input_image = input_image.cuda()

                # run model
                input_image = input_image.unsqueeze(0)
                with torch.no_grad():
                    output = torch.sigmoid(model(input_image))
                tta_outs.append(output)

        # print('tta outs ', [len(tta_outs), tta_outs[0].shape])

        probs_tta = torch.stack(tta_outs)
        # print('probs_tta', probs_tta.shape)
        # asd
        output = probs_tta.mean(0)

        return output


if __name__ == "__main__":
    StoicAlgorithm().process()
    a = json.load(open('/output/probability-covid-19.json', 'rb'))
    b = json.load(open('/output/probability-severe-covid-19.json', 'rb'))
    # print('a and b ')
    # print(a)
    # print(b)
    # print('inside ')
    # print(os.path.listdir('/output/'))
