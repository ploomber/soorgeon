from functools import reduce
from isort import place_module


# NOTE: we use this in find_inputs_and_outputs and ImportParser, maybe
# move the functionality to a class so we only compute it once
def from_imports(tree):
    # build a defined-name -> import-statement-code mapping. Note that
    # the same code may appear more than once if it defines more than one name
    # e.g. from package import a, b, c
    imports = [{
        name.value: import_.get_code().rstrip()
        for name in import_.get_defined_names()
    } for import_ in tree.iter_imports()]

    if imports:
        imports = reduce(lambda x, y: {**x, **y}, imports)
    else:
        imports = {}

    return imports


def packages_used(tree):
    """
    Return a list of the packages used, correcting for some packages whose
    module name does not match the PyPI package (e.g., sklearn -> scikit-learn)
    """
    pkg_name = {
        'sklearn': 'scikit-learn',
    }

    def extract_name(import_):
        if import_.type == 'name':
            return import_.value
        elif import_.type in {'dotted_name', 'dotted_as_name'}:
            return import_.children[0].value

        second = import_.children[1]

        if second.type in {'dotted_name', 'dotted_as_name'}:
            return extract_name(second.children[0])
        else:
            return second.value

    def extract_pkg_name(import_):
        name = extract_name(import_)

        return (pkg_name.get(name, name)
                if place_module(name) == 'THIRDPARTY' else None)

    pkgs = [extract_pkg_name(import_) for import_ in tree.iter_imports()]

    return sorted([name for name in set(pkgs) if name is not None])


def from_def_and_class(tree):
    fns = {
        fn.name.value: fn.get_code().rstrip()
        for fn in tree.iter_funcdefs()
    }

    classes = {
        class_.name.value: class_.get_code().rstrip()
        for class_ in tree.iter_classdefs()
    }

    return {**fns, **classes}


def find_defined_names(tree):
    return {**from_imports(tree), **from_def_and_class(tree)}
