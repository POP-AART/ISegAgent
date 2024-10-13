import argparse
from argparse import Namespace


def set_nested_attr(namespace, keys, value):
    for key in keys[:-1]:
        if not hasattr(namespace, key):
            setattr(namespace, key, Namespace())
        namespace = getattr(namespace, key)
    setattr(namespace, keys[-1], value)


def convert_to_nested_namespace(args):
    nested_args = Namespace()
    for arg, value in vars(args).items():
        keys = arg.split('.')
        set_nested_attr(nested_args, keys, value)
    return nested_args

# TODO: useless, might be deleted