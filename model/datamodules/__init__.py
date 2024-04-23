
from .rsicd_caption_karpathy_datamodule import RSICDCaptionKarpathyDataModule
from .rsitmd_caption_karpathy_datamodule import RSITMDCaptionKarpathyDataModule

_datamodules = {
    "rsicd":RSICDCaptionKarpathyDataModule,
    "rsitmd":RSITMDCaptionKarpathyDataModule,
}