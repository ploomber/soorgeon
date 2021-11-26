from functools import reduce


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
