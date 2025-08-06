from .base import *
from .randomizations import *
from .observations import *
from .rewards import *
from .terminations import *
from .commands import *
from .action import *
from .addons import *

def get_obj_by_class(mapping, obj_class):
    return {
        k: v for k, v in mapping.items() 
        if isinstance(v, type) and issubclass(v, obj_class)
    }

RAND_FUNCS = get_obj_by_class(vars(randomizations), randomizations.Randomization)
ADDONS = get_obj_by_class(vars(addons), addons.AddOn)