
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


    suffixes= ['1','2','3','4','5']
    # artifacts = do_learning(DATA_DIR, ARTIFACT_DIR)


    preprocess_dir(DATA_DIR, SCRATCH_DIR)
    shutil.copy(os.path.join(DATA_DIR, 'reference.csv'), SCRATCH_DIR)

    # do splits

    ref = pd.read_csv(os.path.join(SCRATCH_DIR, 'reference.csv'))

    for suffix in suffixes:

        ref_sampled = ref.sample(frac=1)
        n = len(ref_sampled)
        train_n = int(0.8*n)

        train_df = ref_sampled.iloc[:train_n]
        val_df = ref_sampled.iloc[train_n:]

        print('First patient of train df ', train_df.iloc[0])

        train_df.to_csv(os.path.join(SCRATCH_DIR, f'train_{suffix}.csv'),index=False)

        train_df.to_csv(os.path.join(SCRATCH_DIR, f'val_{suffix}.csv'),index=False)

        config = f'./configs/config_mobile_{suffix}.json'


        artifacts = do_learning(config, suffix)

        print('artifacts ', artifacts)
        shutil.copy(artifacts[0], os.path.join(ARTIFACT_DIR, suffix+'.pth.tar'))

        # When the learning has completed, any artifacts should have been saved to ARTIFACT_DIR.
        # Alternatively, you can copy artifacts to ARTIFACT_DIR after the learning has completed:


        print("Training completed for suffix ", suffix)

    print("Training completed")