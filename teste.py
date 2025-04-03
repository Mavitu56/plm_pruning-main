from dataclasses import dataclass
from typing import Dict

from syne_tune.config_space import Domain, Categorical
from syne_tune.optimizer.baselines import (
    RandomSearch,
    MOREA,
    NSGA2,
    MORandomScalarizationBayesOpt,
    MOASHA,
)

from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)

from local_search import LS

from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertForMultipleChoice,
)

from model_wrapper.mask import mask_bert
from search_spaces import (
    SmallSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
    MediumSearchSpace,
)

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import logging
from copy import deepcopy
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers import StochasticSearcher
from syne_tune.config_space import Domain

import transformers

from model_wrapper.mask import (
    mask_bert,
    mask_gpt,
    mask_gpt_neox,
    mask_roberta,
)

import logging
import sys
import time

from dataclasses import dataclass, field

import torch
import datasets
import evaluate
import numpy as np
from syne_tune.report import Reporter
from torch.optim import AdamW

import transformers
from transformers import (
    AutoConfig,
    get_scheduler,
    HfArgumentParser,
    TrainingArguments,
)

from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG
from data_wrapper.task_data import GLUE_TASK_INFO
from estimate_efficency import compute_parameters
from model_data import get_model_data
from train_supernet import model_types

import json
import logging
import os

from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoModelForSequenceClassification

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint
from syne_tune.experiments import load_experiment

from baselines import MethodArguments, methods

import json

import os
import time
import logging
import sys

from dataclasses import dataclass, field

import numpy as np
import torch
import datasets
import transformers

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from evaluate import load
from functools import partial

from whittle.search import multi_objective_search

from estimate_efficency import compute_parameters
from data_wrapper.task_data import GLUE_TASK_INFO
from search_spaces import (
    SmallSearchSpace,
    MediumSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
)
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG
from model_data import get_model_data
from train_supernet import model_types

import numpy as np
import torch

from syne_tune.config_space import randint, choice, Domain, ordinal

import os
import time
import json

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import evaluate

from tqdm.auto import tqdm

from torch.optim import AdamW

from transformers import (
    AutoConfig,
    get_scheduler,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from whittle.sampling import RandomSampler
from whittle.training_strategies import (
    RandomLinearStrategy,
    RandomStrategy,
    SandwichStrategy,
    StandardStrategy,
)

from search_spaces import (
    FullSearchSpace,
    SmallSearchSpace,
    LayerSearchSpace,
    MediumSearchSpace,
)
from data_wrapper.task_data import GLUE_TASK_INFO
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG
from bert import (
    SuperNetBertForMultipleChoiceSMALL,
    SuperNetBertForMultipleChoiceMEDIUM,
    SuperNetBertForMultipleChoiceLAYER,
    SuperNetBertForMultipleChoiceLARGE,
    SuperNetBertForSequenceClassificationSMALL,
    SuperNetBertForSequenceClassificationMEDIUM,
    SuperNetBertForSequenceClassificationLAYER,
    SuperNetBertForSequenceClassificationLARGE,
)
from roberta import (
    SuperNetRobertaForMultipleChoiceSMALL,
    SuperNetRobertaForMultipleChoiceMEDIUM,
    SuperNetRobertaForMultipleChoiceLAYER,
    SuperNetRobertaForMultipleChoiceLARGE,
    SuperNetRobertaForSequenceClassificationSMALL,
    SuperNetRobertaForSequenceClassificationMEDIUM,
    SuperNetRobertaForSequenceClassificationLAYER,
    SuperNetRobertaForSequenceClassificationLARGE,
)