"""Qlib handlers adapted for WRDS SP500 dump (no $vwap field)."""

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL


class Alpha158SP500(Alpha158):
    """Built-in Alpha158 with VWAP removed because our dump has no $vwap field."""

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)
