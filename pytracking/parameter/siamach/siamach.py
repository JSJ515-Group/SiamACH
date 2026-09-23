from ltr.train_settings.siamach.siamach_tracker_settings import get_tracker_settings
from ltr.models.tracking.siamach import siamach_tracker
from pytracking.utils import TrackerParams
import torch

def parameters():
    params = TrackerParams()
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.checkpoint = ''
    params.settings = get_tracker_settings()
    params.device = torch.device("cuda:0")
    params.net = siamach_tracker(params.settings)

    return params
