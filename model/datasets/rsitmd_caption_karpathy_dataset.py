from .base_dataset import BaseDataset


class RSITMDCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        if split == "train":
            names = ["rsitmd_caption_karpathy_train"]
        elif split == "val":
            names = ["rsitmd_caption_karpathy_test"]
        elif split == "test":
            names = ["rsitmd_caption_karpathy_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
