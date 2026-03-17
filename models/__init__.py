import sys
import os

# Add Show-o2 to path so we can import its models
SHOWO2_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Show-o', 'show-o2')
if SHOWO2_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(SHOWO2_PATH))

from models import Showo2Qwen2_5, omni_attn_mask_naive, WanVAE
from models.misc import (
    prepare_gen_input, get_text_tokenizer, velocity_prediction,
    next_token_prediction
)
from models.lr_schedulers import get_scheduler

from .showo2_spy_wrapper import Showo2SpyWrapper
