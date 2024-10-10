import json


def parse_complex_args_or_load_json_config(args, json_config_key="config_path"):
    """
    Converts the given args object to a dictionary and parses complex type args by dot notation, or loads a configuration from a json file.
    If the given args has a value for the json_config_key field then the returned args dict will be loaded from the json file. Otherwise,
    the given args object will be converted to a dictionary, and parsed for complex type args by dot notation and returned.
    :param args: Arguments object as parsed using ArgumentParser or a dictionary of arguments.
    :param json_config_key: Name of the optional field that contains the path to load json config from.
    :return: Dictionary of arguments, converted from the given args or loaded from a json file.
    """
    args_dict = args if isinstance(args, dict) else args.__dict__
    if json_config_key not in args_dict or not args_dict[json_config_key]:
        return parse_complex_args(args)

    with open(args_dict[json_config_key]) as f:
        return json.load(f)


def parse_complex_args(args):
    """
    Converts the given args object to a dictionary and parses complex type args by dot notation.
    :param args: Arguments object as parsed using ArgumentParser or a dictionary of arguments.
    :return: Dictionary of arguments, converted from the given args and loading complex types by dot notation.
    """
    args_dict = args if isinstance(args, dict) else args.__dict__

    parsed_args = {}
    for key, value in args_dict.items():
        if not isinstance(key, str) or "." not in key:
            parsed_args[key] = value
        else:
            __update_with_parsed_dot_notation_arg(parsed_args, key, value)

    return parsed_args


def __update_with_parsed_dot_notation_arg(parsed_args, key, value):
    """
    Updates the given arguments dictionary with the complex object field value that is described by the key by dot notation.
    :param parsed_args: Dictionary of arguments to be updated.
    :param key: String key of an argument that contains dot notation.
    :param value: Value of argument.
    """
    key_parts = key.split(".")
    current_dict = parsed_args
    for i, key_part in enumerate(key_parts):
        if i == len(key_parts) - 1:
            current_dict[key_part] = value
            return

        if key_part not in current_dict:
            current_dict[key_part] = {}
        current_dict = current_dict[key_part]
