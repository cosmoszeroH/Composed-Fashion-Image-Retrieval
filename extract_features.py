import pickle
from typing import Union

import PIL
import PIL.Image

import clip
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils import FashionIQDataset, targetpad_transform, CIRRDataset #, data_path
from src.utils import collate_fn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_and_save_index_features(dataset: Union[CIRRDataset, FashionIQDataset], clip_model: nn.Module,
                                    feature_dim: int, file_name: str):
    """
    Extract CIRR or fashionIQ features (with the respective names) from a dataset object and save them into a file
    which can be used by the server
    :param dataset: dataset where extract the features
    :param clip_model: clip model used to extract the features
    :param feature_dim: feature dimensionality
    :param file_name: name used to save the features
    """
    val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8, pin_memory=True,
                            collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim), dtype=torch.float16).to(device)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")

    # iterate over the dataset object
    for names, images in tqdm(val_loader):
        images = images.to(device)

        # extract and concatenate features and names
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)

    # save the extracted features
    data_path = r'C:\Users\CPS125\OneDrive\Documents\Dataset\fashionIQ_dataset\demo_feature_2'
    torch.save(index_features, f"{data_path}\\{file_name}_index_features.pt")
    with open(f'{data_path}\\{file_name}_index_names.pkl', 'wb+') as f:
        pickle.dump(index_names, f)


def main():
    clip_model, _ = clip.load('RN50x4')
    input_dim = clip_model.visual.input_resolution
    preprocess = targetpad_transform(1.25, input_dim)
    feature_dim = clip_model.visual.output_dim
    # clip_state_dict = torch.load(r'D:\Specialization\Project\Clip4Cir\FashionIQ\RN50x4_fullft\fiq_clip_RN50x4_fullft.pt', map_location=torch.device('cpu'))
    # clip_model.load_state_dict(clip_state_dict["CLIP"])

    dress_type = 'toptee'
    set_type = 'test'
    classic_dataset = FashionIQDataset(set_type, [dress_type], 'classic', preprocess)
    extract_and_save_index_features(classic_dataset, clip_model, feature_dim, f'{dress_type}_{set_type}')


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()