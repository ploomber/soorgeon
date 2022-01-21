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

    Returns None if fails to parse them
    """
    pkg_name = {
        'sklearn': 'scikit-learn',
    }

    def flatten(elements):
        return [i for sub in elements for i in sub]

    def extract_names(import_):
        if import_.type == 'name':
            return [import_.value]
        elif import_.type in {'dotted_name', 'dotted_as_name'}:
            return [import_.children[0].value]

        second = import_.children[1]

        if second.type in {'dotted_name', 'dotted_as_name'}:
            return extract_names(second.children[0])
        elif second.type == 'dotted_as_names':
            # import a as something, b as another
            return flatten([
                extract_names(node.children[0])
                for i, node in enumerate(second.children) if i % 2 == 0
            ])
        else:
            return [second.value]

    pkgs = flatten([extract_names(import_) for import_ in tree.iter_imports()])

    # replace using pkg_name mapping and ignore standard lib
    pkgs_final = [
        pkg_name.get(name, name) for name in pkgs
        if place_module(name) == 'THIRDPARTY'
    ]

    # remove duplicates and sort
    return sorted(set(pkgs_final))


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
