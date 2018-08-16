"""Bare bones object oriented wrapper around grib_api.
 
.. codeauthor:: Florian Rathgeber <florian.rathgeber@ecmwf.int>
"""
 
from gribapi import *
 
 
class GribKeysIterator(object):
    "Iterator for the keys in a GRIB message"
 
    def __init__(self, gid, namespace=None):
        self.it = grib_keys_iterator_new(gid, namespace)
 
    def __iter__(self):
        return self
 
    def next(self):
        if not grib_keys_iterator_next(self.it):
            raise StopIteration()
        return grib_keys_iterator_get_name(self.it)
 
 
class GribIterator(object):
    "Iterator for the keys and values in a GRIB message"
 
    def __init__(self, gid, namespace=None):
        self.gid = gid
        self.it = grib_keys_iterator_new(gid, namespace)
 
    def __iter__(self):
        return self
 
    def next(self):
        if not grib_keys_iterator_next(self.it):
            raise StopIteration()
        key = grib_keys_iterator_get_name(self.it)
        size = grib_get_size(self.gid, key)
        if size > 1:
            val = grib_get_array(self.gid, key)
        else:
            val = grib_get(self.gid, key)
        return key, val
 
 
class Grib(object):
    """A GRIB message initialised either from a file object or file name
 
    :param fname: file name or file object
    """
 
    def __init__(self, fname):
        self.f = open(fname) if isinstance(fname, str) else fname
        self.gid = grib_new_from_file(self.f)
 
    def __del__(self):
        grib_release(self.gid)
        self.f.close()
 
    def __iter__(self):
        return GribIterator(self, self.gid)
 
    def __getitem__(self, key):
        size = grib_get_size(self.gid, key)
        if size > 1:
            return grib_get_array(self.gid, key)
        else:
            return grib_get(self.gid, key)
 
    def __setitem__(self, key, val):
        size = grib_get_size(self.gid, key)
        if size > 1:
            grib_set_array(self.gid, key, val)
        else:
            grib_set(self.gid, key, val)
 
    @property
    def keys(self):
        "A tuple of all the keys contained in the message."
        return tuple(GribKeysIterator(self.gid))
 
    @property
    def values(self):
        "Contents of the ``values`` key as :class:`numpy.ndarray`."
        return grib_get_values(self.gid)
 
    def iterkeys(self, namespace=None):
        """An iterator of all the keys contained in the message.
 
        :param namespace: (optional) restrict keys to namespace"""
        return GribKeysIterator(self.gid, namespace)