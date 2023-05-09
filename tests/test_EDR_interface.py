import math
import sys

import numpy as np
import pandas as pd
import xarray as xr
from shapely import wkt

from polytope.datacube.datacube_request_tree import DatacubeRequestTree
from polytope.datacube.xarray import XArrayDatacube
from polytope.engine.hullslicer import HullSlicer
from polytope.polytope import Polytope, Request
from polytope.shapes import (
    Box,
    ConvexPolytope,
    Disk,
    PathSegment,
    Polygon,
    Select,
    Span,
    Union,
)

class TestEDRInterface:
    def setup_method(self, method):
        # Create a dataarray with 3 labelled axes using different index types
        dims = np.random.randn(3, 6, 129)
        array = xr.Dataset(
            data_vars=dict(param=(["date", "step", "level"], dims)),
            coords={
                "date": pd.date_range("2000-01-01", "2000-01-03", 3),
                "step": [0, 3, 6, 9, 12, 15],
                "level": range(1, 130),
            },
        )
        self.xarraydatacube = XArrayDatacube(array)
        self.slicer = HullSlicer()
        self.API = Polytope(datacube=array, engine=self.slicer)

    # Testing different shapes

    def test_2D_box(self):
        request = Request(Box(["step", "level"], [3, 10], [6, 11]), Select("date", ["2000-01-01"]))
        result = self.API.retrieve(request)
        assert len(result.leaves) == 4

    def test_EDR_point(self):
        p1 = wkt.loads('POINT(0 51.48)')
        #request = Request(Select(["step", "level"], [p1.x, p1.y]))
        assert p1.x == 0
        assert p1.y == 51.48
