from dataclasses import dataclass
from typing import Any


@dataclass
class MetaDataHolder:
    metadata: Any  # array of 4, element = different size array
    idx2item: Any  # list of 4, element = dict
    item2idx: Any  # list of 4, element = dict
    item_size: Any  # list of 4, element = size of other element
    similarities: Any  # array of 4, element = different size NxN array
    feats: Any  # array of 4, element = different size array
