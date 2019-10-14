import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import sklearn
import re
import feather
import datetime
import math
import dateutil
import itertools
import datetime as dt
from functools import partial
from functools import reduce



# Color names in AtRemote data
colors = [('K', 'BK'), 'Y', 'M', 'C']
# Normalized color names
colors_norm = ['K', 'Y', 'M', 'C']
toner_replace_rate = [f"Toner.{c}.replaced.rate" for c in colors]
color_display = ['k', 'y', 'm', 'c']


# Black is variously K or BK in AtRemote field names, so allow colors to be a list or tuple of names
def colorToRegex(c):
    if isinstance(c, (list, tuple)):
        return f"({'|'.join(c)})"
    else:
        return c
    
def colorToString(c):
    if isinstance(c, (list, tuple)):
        return c[0]
    else:
        return c

# Find fields matching regex template, e.g. '.*Current\.Toner\.%s\.(?!previous)', where the short color name (e.g. K) is substituted for %s.
# Likewise name_template specifies a template for a label (e.g. 'Toner.%s')
# Return a dictionary mapping labels to field names
# If colors is false, use templates without substitution
def findFields(names, regex_template, name_template, colors=colors, take_first=False, allow_missing=False):
    res = {}
    if not colors:
        colors = [None] 
    for color in colors:
        if isinstance(regex_template, list):
            regex_templates = regex_template
        else:
            regex_templates = [regex_template]
        if color is None:
            regexes = regex_templates
        else:
            regexes = [r % colorToRegex(color) for r in regex_templates]
        regex = "|".join(regexes)
        matching = [n for n in names if re.match(regex, n)]
        if not matching:
            if not allow_missing:
                raise RuntimeError(f"""No match for '{regex}''""")
        else:
            if len(matching)>2:
                if not take_first:
                    raise RuntimeError(f"Unexpected multiple matches: {matching}")
            # Human-friendly toner replacement stat names that we will make consistent between models
            if color is None:
                res[name_template] = matching[0]
            else:
                res[name_template % colorToString(color)] = matching[0]
    return res
