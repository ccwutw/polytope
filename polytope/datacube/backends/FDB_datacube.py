import math
import os
from copy import deepcopy

from ...utility.combinatorics import unique, validate_axes
from .datacube import Datacube, DatacubePath, IndexTree, configure_datacube_axis

os.environ["DYLD_LIBRARY_PATH"] = "/Users/male/build/fdb-bundle/lib"
os.environ["FDB_HOME"] = "/Users/male/git/fdb-home"
import pyfdb  # noqa: E402

# TODO: currently, because the date and time are strings, the data will be treated as an unsliceable axis...


def glue(path):
    return {"t": 0}


def update_fdb_dataarray(fdb_dataarray):
    new_dict = {}
    for key, values in fdb_dataarray.items():
        if key in ["levelist", "param", "step"]:
            new_values = []
            for val in values:
                new_values.append(int(val))
            new_dict[key] = new_values
        else:
            new_dict[key] = values
    new_dict["values"] = [0.0]
    return new_dict


class FDBDatacube(Datacube):
    def __init__(self, config={}, axis_options={}):
        # Need to get the cyclic options and grid options from somewhere
        self.axis_options = axis_options
        self.grid_mapper = None
        self.axis_counter = 0
        self._axes = {}
        self.blocked_axes = []
        self.transformation = {}
        self.fake_axes = []
        self.complete_axes = []
        partial_request = config
        # Find values in the level 3 FDB datacube
        # Will be in the form of a dictionary? {axis_name:values_available, ...}
        fdb = pyfdb.FDB()
        fdb_dataarray = fdb.axes(partial_request).as_dict()
        dataarray = update_fdb_dataarray(fdb_dataarray)
        self.dataarray = dataarray

        for name, values in dataarray.items():
            values.sort()
            options = axis_options.get(name, {})
            configure_datacube_axis(options, name, values, self)
            self.complete_axes.append(name)

    def get(self, requests: IndexTree):
        for r in requests.leaves:
            path = r.flatten()
            path = self.remap_path(path)
            if len(path.items()) == self.axis_counter:
                if self.grid_mapper is not None:
                    first_axis = self.grid_mapper._mapped_axes[0]
                    first_val = path[first_axis]
                    second_axis = self.grid_mapper._mapped_axes[1]
                    second_val = path[second_axis]
                    path.pop(first_axis, None)
                    path.pop(second_axis, None)
                    # need to remap the lat, lon in path to dataarray index
                    unmapped_idx = self.grid_mapper.unmap(first_val, second_val)
                    path[self.grid_mapper._base_axis] = unmapped_idx
                    # Ask FDB what values it has on the path
                    subxarray = glue(path)
                    key = list(subxarray.keys())[0]
                    value = subxarray[key]
                    r.result = (key, value)
                else:
                    # if we have no grid map, still need to assign values
                    subxarray = glue(path)
                    # value = subxarray.item()
                    # key = subxarray.name
                    key = list(subxarray.keys())[0]
                    value = subxarray[key]
                    r.result = (key, value)
            else:
                r.remove_branch()

    def get_mapper(self, axis):
        return self._axes[axis]

    def remap_path(self, path: DatacubePath):
        for key in path:
            value = path[key]
            path[key] = self._axes[key].remap_val_to_axis_range(value)
        return path

    def _look_up_datacube(self, search_ranges, search_ranges_offset, indexes, axis, first_val):
        idx_between = []
        for i in range(len(search_ranges)):
            print(search_ranges[i])
            r = search_ranges[i]
            offset = search_ranges_offset[i]
            low = r[0]
            up = r[1]
            if axis.name in self.transformation.keys():
                axis_transforms = self.transformation[axis.name]
                temp_indexes = deepcopy(indexes)
                for transform in axis_transforms:
                    print(low)
                    print(up)
                    print(first_val)
                    print(axis)
                    print(offset)
                    print(temp_indexes)
                    (offset, temp_indexes) = transform._find_transformed_indices_between(
                        axis, self, temp_indexes, low, up, first_val, offset
                    )
                    print(offset)
                indexes_between = temp_indexes
            else:
                indexes_between = self._find_indexes_between(axis, indexes, low, up)
            # Now the indexes_between are values on the cyclic range so need to remap them to their original
            # values before returning them
            for j in range(len(indexes_between)):
                if offset is None:
                    indexes_between[j] = indexes_between[j]
                else:
                    indexes_between[j] = round(indexes_between[j] + offset, int(-math.log10(axis.tol)))
                idx_between.append(indexes_between[j])
        return idx_between

    # def _look_up_datacube(self, search_ranges, search_ranges_offset, indexes, axis, first_val):
    #     idx_between = []
    #     for i in range(len(search_ranges)):
    #         r = search_ranges[i]
    #         offset = search_ranges_offset[i]
    #         low = r[0]
    #         up = r[1]

    #         if self.grid_mapper is not None:
    #             first_axis = self.grid_mapper._mapped_axes[0]
    #             second_axis = self.grid_mapper._mapped_axes[1]
    #             if axis.name == first_axis:
    #                 indexes_between = self.grid_mapper.map_first_axis(low, up)
    #             elif axis.name == second_axis:
    #                 indexes_between = self.grid_mapper.map_second_axis(first_val, low, up)
    #             else:
    #                 indexes_between = [i for i in indexes if low <= i <= up]
    #         else:
    #             indexes_between = [i for i in indexes if low <= i <= up]

    #         # Now the indexes_between are values on the cyclic range so need to remap them to their original
    #         # values before returning them
    #         for j in range(len(indexes_between)):
    #             if offset is None:
    #                 indexes_between[j] = indexes_between[j]
    #             else:
    #                 indexes_between[j] = round(indexes_between[j] + offset, int(-math.log10(axis.tol)))

    #             idx_between.append(indexes_between[j])
    #     return idx_between

    def fit_path_to_original_datacube(self, path):
        path = self.remap_path(path)
        first_val = None
        unmap_path = {}
        considered_axes = []
        changed_type_path = {}
        for axis_name in self.transformation.keys():
            axis_transforms = self.transformation[axis_name]
            for transform in axis_transforms:
                (path, temp_first_val, considered_axes, unmap_path, changed_type_path) = transform._adjust_path(
                    path, considered_axes, unmap_path, changed_type_path
                )
                if temp_first_val:
                    first_val = temp_first_val
        # for key in path.keys():
        #     if self.dataarray[key].dims == ():
        #         path.pop(key)
        return (path, first_val, considered_axes, unmap_path, changed_type_path)

    # def get_indices(self, path: DatacubePath, axis, lower, upper):
    #     path = self.remap_path(path)
    #     first_val = None
    #     if self.grid_mapper is not None:
    #         first_axis = self.grid_mapper._mapped_axes[0]
    #         first_val = path.get(first_axis, None)
    #         second_axis = self.grid_mapper._mapped_axes[1]
    #         path.pop(first_axis, None)
    #         path.pop(second_axis, None)
    #         if axis.name == first_axis:
    #             indexes = []
    #         elif axis.name == second_axis:
    #             indexes = []
    #         else:
    #             indexes = self.dataarray[axis.name]
    #     else:
    #         indexes = self.dataarray[axis.name]

    #     # Here, we do a cyclic remapping so we look up on the right existing values in the cyclic range on the datacube
    #     search_ranges = axis.remap([lower, upper])
    #     original_search_ranges = axis.to_intervals([lower, upper])

    #     # Find the offsets for each interval in the requested range, which we will need later
    #     search_ranges_offset = []
    #     for r in original_search_ranges:
    #         offset = axis.offset(r)
    #         search_ranges_offset.append(offset)

    #     # Look up the values in the datacube for each cyclic interval range
    #     idx_between = self._look_up_datacube(search_ranges, search_ranges_offset, indexes, axis, first_val)

    #     # Remove duplicates even if difference of the order of the axis tolerance
    #     if offset is not None:
    #         # Note that we can only do unique if not dealing with time values
    #         idx_between = unique(idx_between)

    #     return idx_between

    def get_indices(self, path: DatacubePath, axis, lower, upper):
        # NEW VERSION OF THIS METHOD
        (path, first_val, considered_axes, unmap_path, changed_type_path) = self.fit_path_to_original_datacube(path)

        subarray = self.dataarray
        # subarray = self.dataarray.sel(path, method="nearest")
        # subarray = subarray.sel(unmap_path)
        # subarray = subarray.sel(changed_type_path)
        # Get the indexes of the axis we want to query
        # XArray does not support branching, so no need to use label, we just take the next axis
        if axis.name in self.transformation.keys():
            axis_transforms = self.transformation[axis.name]
            # This bool will help us decide for which axes we need to calculate the indexes again or not
            # in case there are multiple relevant transformations for an axis
            already_has_indexes = False
            for transform in axis_transforms:
                # TODO: here, instead of creating the indices, would be better to create the standard datacube axes and
                # then succesively map them to what they should be
                indexes = transform._find_transformed_axis_indices(self, axis, subarray, already_has_indexes)
                already_has_indexes = True
        else:
            indexes = self.datacube_natural_indexes(axis, subarray)
        # Here, we do a cyclic remapping so we look up on the right existing values in the cyclic range on the datacube
        search_ranges = axis.remap([lower, upper])
        original_search_ranges = axis.to_intervals([lower, upper])
        # Find the offsets for each interval in the requested range, which we will need later
        search_ranges_offset = []
        for r in original_search_ranges:
            offset = axis.offset(r)
            search_ranges_offset.append(offset)
        # Look up the values in the datacube for each cyclic interval range
        idx_between = self._look_up_datacube(search_ranges, search_ranges_offset, indexes, axis, first_val)
        # Remove duplicates even if difference of the order of the axis tolerance
        if offset is not None:
            # Note that we can only do unique if not dealing with time values
            idx_between = unique(idx_between)
        return idx_between
    
    def datacube_natural_indexes(self, axis, subarray):
        indexes = subarray[axis.name]
        return indexes

    def has_index(self, path: DatacubePath, axis, index):
        # when we want to obtain the value of an unsliceable axis, need to check the values does exist in the datacube
        subarray_vals = self.dataarray[axis.name]
        return index in subarray_vals

    @property
    def axes(self):
        return self._axes

    def validate(self, axes):
        return validate_axes(self.axes, axes)

    def ax_vals(self, name):
        for _name, values in self.dataarray.items():
            if _name == name:
                return values

    def _find_indexes_between(self, axis, indexes, low, up):
        print("INSIDE FIND INDEXES")
        print(indexes)
        print(low)
        print(up)
        return [i for i in indexes if low <= i <= up]
