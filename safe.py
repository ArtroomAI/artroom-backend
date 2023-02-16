# this code is adapted from the script contributed by anon from /h/

import _codecs
import collections
import pickle
import re
import sys
import traceback
import zipfile

import numpy
import torch

# PyTorch 1.13 and later have _TypedStorage renamed to TypedStorage
TypedStorage = torch.storage.TypedStorage if hasattr(torch.storage, 'TypedStorage') else torch.storage._TypedStorage


def encode(*args):
    out = _codecs.encode(*args)
    return out


class RestrictedUnpickler(pickle.Unpickler):
    extra_handler = None

    def persistent_load(self, saved_id):
        assert saved_id[0] == 'storage'
        try:
            t = TypedStorage()
            return t
        except:
            return None

    def find_class(self, module, name):
        if self.extra_handler is not None:
            res = self.extra_handler(module, name)
            if res is not None:
                return res

        if module == 'collections' and name == 'OrderedDict':
            return getattr(collections, name)
        if module == 'torch._utils' and name in ['_rebuild_tensor_v2', '_rebuild_parameter']:
            return getattr(torch._utils, name)
        if module == 'torch' and name in ['FloatStorage', 'HalfStorage', 'IntStorage', 'LongStorage', 'DoubleStorage',
                                          'ByteStorage']:
            return getattr(torch, name)
        if module == 'torch.nn.modules.container' and name in ['ParameterDict']:
            return getattr(torch.nn.modules.container, name)
        if module == 'numpy.core.multiarray' and name == 'scalar':
            return numpy.core.multiarray.scalar
        if module == 'numpy' and name == 'dtype':
            return numpy.dtype
        if module == '_codecs' and name == 'encode':
            return encode
        if module == "pytorch_lightning.callbacks" and name == 'model_checkpoint':
            import pytorch_lightning.callbacks
            return pytorch_lightning.callbacks.model_checkpoint
        if module == "pytorch_lightning.callbacks.model_checkpoint" and name == 'ModelCheckpoint':
            import pytorch_lightning.callbacks.model_checkpoint
            return pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
        if module == "__builtin__" and name == 'set':
            return set

        # Forbid everything else.
        raise Exception(f"global '{module}/{name}' is forbidden")


disallowed_patterns = ['.exe', '.bat', '.com', '.cmd', '.inf', '.ipa', '.osx', '.pif', '.runwsh']
disallowed_patterns = disallowed_patterns + [x.upper() for x in disallowed_patterns]


def check_zip_filenames(filename, names):
    for name in names:
        for pattern in disallowed_patterns:
            if pattern in name:
                raise Exception(f"bad file inside {filename}: {name}")


def check_pt(filename, extra_handler):
    try:
        # new pytorch format is a zip file
        with zipfile.ZipFile(filename) as z:
            check_zip_filenames(filename, z.namelist())

            if "pkl" in z.namelist()[0]:
                with z.open(z.namelist()[0]) as file:
                    unpickler = RestrictedUnpickler(file)
                    unpickler.extra_handler = extra_handler
                    unpickler.load()
            else:
                raise Exception(f"Expected {z.namelist()[0]} to be .pkl file")

    except zipfile.BadZipfile:

        # if it's not a zip file, it's an old pytorch format, with five objects written to pickle
        with open(filename, "rb") as file:
            unpickler = RestrictedUnpickler(file)
            unpickler.extra_handler = extra_handler
            for i in range(5):
                unpickler.load()


def load(filename, *args, **kwargs):
    return load_with_extra(filename, *args, **kwargs)


def load_with_extra(filename, extra_handler=None, *args, **kwargs):
    """
    this functon is intended to be used by extensions that want to load models with
    some extra classes in them that the usual unpickler would find suspicious.
    Use the extra_handler argument to specify a function that takes module and field name as text,
    and returns that field's value:
    ```python
    def extra(module, name):
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        return None
    safe.load_with_extra('model.pt', extra_handler=extra)
    ```
    The alternative to this is just to use safe.unsafe_torch_load('model.pt'), which as the name implies is
    definitely unsafe.
    """
    try:
        check_pt(filename, extra_handler)

    except pickle.UnpicklingError:
        print(f"Error verifying pickled file from {filename}:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"-----> !!!! The file is most likely corrupted !!!! <-----", file=sys.stderr)
        return None

    except Exception as e:
        print(f"Caught {e} error verifying pickled file from {filename}:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"\nThe file may be malicious, so the program is not going to read it.", file=sys.stderr)
        return None

    return torch.load(filename, map_location="cpu", *args, **kwargs)
