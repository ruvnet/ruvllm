import re


def ensure_unique_parameters(text):
    names = re.findall(r"<parameter=([^>]+)>", text)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tool parameters are ambiguous")
