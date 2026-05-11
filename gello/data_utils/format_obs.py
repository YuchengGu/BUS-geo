import datetime
import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    frame = dict(obs)
    frame["control"] = np.asarray(action)  # add action to saved frame

    # make folder if it doesn't exist
    folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder / (timestamp.isoformat() + ".pkl")

    with open(recorded_file, "wb") as f:
        pickle.dump(frame, f)
