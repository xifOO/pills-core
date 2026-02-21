import pandas as pd
from pills_core.descriptors.base import Descriptor


class DatasetDescriptor(Descriptor[pd.DataFrame]): ...
