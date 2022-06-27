
import shutil
from train import do_learning
import os
from preprocess import preprocess_dir
import pandas as pd

DATA_DIR = "/input/"
ARTIFACT_DIR = "/output/"

SCRATCH_DIR = "/scratch/"

if __name__ == "__main__":
    # Substitute do_learning for your training function.
    # It is recommended to write artifacts (e.g. model weights) to ARTIFACT_DIR during training.


    suffixes_mob= ['mobile_1','mobile_2','mobile_3','mobile_4','mobile_5']
    suffixes_res= ['resnet_1','resnet_2','resnet_3','resnet_4','resnet_5']

    suffixes = suffixes_mob + suffixes_res
    # suffixes = suffixes_res

    # artifacts = do_learning(DATA_DIR, ARTIFACT_DIR)

    mha_dir = os.path.join(DATA_DIR, 'data/mha/')
    reference_file = os.path.join(DATA_DIR, 'metadata/reference.csv')

    preprocess_dir(mha_dir, SCRATCH_DIR)
    shutil.copy(reference_file, SCRATCH_DIR)

    # do splits

    ref = pd.read_csv(os.path.join(SCRATCH_DIR, 'reference.csv'))

    for suffix in suffixes:

        ref_sampled = ref.sample(frac=1)
        n = len(ref_sampled)
        train_n = int(0.85*n)

        train_df = ref_sampled.iloc[:train_n]
        val_df = ref_sampled.iloc[train_n:]


        print('First patient of train df ', train_df.iloc[0])
        print('len of train, val', [len(train_df), len(val_df)])

        a = set(train_df.PatientID.to_list())
        b = set(val_df.PatientID.to_list())

        print('a,b, intersection ', [len(list(a)),len(list(b)),len(list(a.intersection(b)))])
        # asd

        train_df.to_csv(os.path.join(SCRATCH_DIR, f'train_{suffix}.csv'),index=False)

        val_df.to_csv(os.path.join(SCRATCH_DIR, f'val_{suffix}.csv'),index=False)

        config = f'./configs/config_{suffix}.json'


        artifacts = do_learning(config, suffix)

        print('artifacts ', artifacts)
        shutil.copy(artifacts[0], os.path.join(ARTIFACT_DIR, suffix+'.pth.tar'))

        # When the learning has completed, any artifacts should have been saved to ARTIFACT_DIR.
        # Alternatively, you can copy artifacts to ARTIFACT_DIR after the learning has completed:


        print("Training completed for suffix ", suffix)

    print("Training completed")