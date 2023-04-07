import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 501141319

def solution(p: float, x: np.array) -> tuple:
    alpha = 1 - p 
    scale = np.sqrt(np.var(x)) / np.sqrt(len(x))
    return scale * norm.ppf(1 - alpha / 2)/np.sqrt(47), \
           scale * norm.ppf(alpha / 2)/np.sqrt(47)
