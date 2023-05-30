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
        dims = np.random.randn(3, 200, 200)
        array = xr.Dataset(
            data_vars=dict(param=(["date", "lat", "long"], dims)),
            coords={
                "date": pd.date_range("2000-01-01", "2000-01-03", 3),
                "lat": range(0,200),
                "long": range(0, 200),
            },
        )
        self.xarraydatacube = XArrayDatacube(array)
        self.slicer = HullSlicer()
        self.API = Polytope(datacube=array, engine=self.slicer)

    # Testing different shapes

    def test_2D_box(self):
        request = Request(Box(["lat", "long"], [3, 10], [6, 11]), Select("date", ["2000-01-01"]))
        result = self.API.retrieve(request)
        print(result)
        assert len(result.leaves) == 8

    def test_EDR_point(self):
        p1 = wkt.loads('POINT(0 51)')
        request = Request(Select("date", ["2000-01-03"]), Select("lat", [p1.x]), Select("long", [p1.y]))
        result = self.API.retrieve(request)
        assert p1.x == 0
        assert p1.y == 51

    def test_EDR_radius(self):
        p1 = wkt.loads('POINT(0 51)')
        radius = 3
        request = Request(Disk(["lat", "long"], [p1.x, p1.y], [radius, radius]), Select("date", ["2000-01-01"]))
        result = self.API.retrieve(request)
        assert len(result.leaves) == 18

    #def test_EDR_polygon(self):
