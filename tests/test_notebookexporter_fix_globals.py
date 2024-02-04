import textwrap
import jupytext
from soorgeon.export import NotebookExporter
from soorgeon import export
from ploomber.spec import DAGSpec


def _read(nb_str):
    return jupytext.reads(nb_str, fmt='py:light')


def _reformat(code):
    return textwrap.dedent(code)[:-1]


def test_simple_fix_1_global():
    nb = _reformat(
        """
        # ## section1

        def fn():
            return global1
        global1 = 9
        fn()
        a = fn()
        print(a + fn())
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(global1)' in fixed_code
    assert 'a = fn(global1)' in fixed_code
    assert 'print(a + fn(global1)' in fixed_code

    expected_code = _reformat(
        """
        def fn(global1):
            return global1
        global1 = 9
        fn(global1)
        a = fn(global1)
        print(a + fn(global1))
        """)
    assert expected_code == fixed_code


def test_fix_1_global_1_positional_0_default_0_starArg_0_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(a):
            return(global1 + a)
        global1 = 8
        a = 8
        fn(a)
        x = 8
        y = fn(x)
        print(y + fn(8))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(a, global1):' in fixed_code
    assert 'y = fn(x, global1)' in fixed_code
    assert 'print(y + fn(8, global1))' in fixed_code

    expected_code = _reformat(
        """
        def fn(a, global1):
            return(global1 + a)
        global1 = 8
        a = 8
        fn(a, global1)
        x = 8
        y = fn(x, global1)
        print(y + fn(8, global1))
        """)
    assert expected_code == fixed_code


def test_fix_1_global_0_positional_1_default_0_starArg_0_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(kw=9):
            return(global1 + kw)
        global1 = 9
        fn(kw=8)
        a = 9
        b = fn(kw=a)
        print(b + fn(kw=a))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(global1, kw=9):' in fixed_code
    assert 'fn(global1, kw=8)' in fixed_code
    assert 'b = fn(global1, kw=a)' in fixed_code
    assert 'print(b + fn(global1, kw=a))' in fixed_code
    expected_code = _reformat(
        """
        def fn(global1, kw=9):
            return(global1 + kw)
        global1 = 9
        fn(global1, kw=8)
        a = 9
        b = fn(global1, kw=a)
        print(b + fn(global1, kw=a))
        """)
    assert expected_code == fixed_code


def test_fix_1_global_0_positional_0_default_1_starArg_0_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(*arg):
            return global1 + arg[0]
        global1 = 8
        fn(8)
        fn(8, 8)
        alist = [8, 8]
        a = fn(*alist)
        print(a + fn(*alist))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(global1, *arg):' in fixed_code
    assert 'fn(global1, 8)' in fixed_code
    assert 'fn(global1, 8, 8)' in fixed_code
    assert 'a = fn(global1, *alist)' in fixed_code
    assert 'print(a + fn(global1, *alist))' in fixed_code
    expected_code = _reformat(
        """
        def fn(global1, *arg):
            return global1 + arg[0]
        global1 = 8
        fn(global1, 8)
        fn(global1, 8, 8)
        alist = [8, 8]
        a = fn(global1, *alist)
        print(a + fn(global1, *alist))
        """)
    assert expected_code == fixed_code


def test_fix_1_global_0_positional_0_default_0_starArg_1_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(**kwarg):
            print(global1 + kwarg['kw1'])
        global1 = 8
        adict = dict([('kw1', 8), ('kw2', 8)])
        fn(**adict)
        a = fn(kw1=8, kw2=8)
        print(a + fn(kw1=8, kw2=8))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(global1, **kwarg):' in fixed_code
    assert 'fn(global1, **adict)' in fixed_code
    assert 'a = fn(global1, kw1=8, kw2=8)' in fixed_code
    assert 'print(a + fn(global1, kw1=8, kw2=8))' in fixed_code

    expected_code = _reformat(
        """
        def fn(global1, **kwarg):
            print(global1 + kwarg['kw1'])
        global1 = 8
        adict = dict([('kw1', 8), ('kw2', 8)])
        fn(global1, **adict)
        a = fn(global1, kw1=8, kw2=8)
        print(a + fn(global1, kw1=8, kw2=8))
        """)
    assert expected_code == fixed_code


def test_fix_1_global_1_positional_1_default_0_starArg_0_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(a, kw=9):
            print(global1 + a + kw)
        global1 = 9
        a = 9
        fn(a, kw=8)
        x = 9
        b = fn(x, kw=x)
        print(b + fn(x, kw=x))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(a, global1, kw=9):' in fixed_code
    assert 'fn(a, global1, kw=8)' in fixed_code
    assert 'b = fn(x, global1, kw=x)' in fixed_code
    assert 'print(b + fn(x, global1, kw=x))' in fixed_code

    expected_code = _reformat(
        """
        def fn(a, global1, kw=9):
            print(global1 + a + kw)
        global1 = 9
        a = 9
        fn(a, global1, kw=8)
        x = 9
        b = fn(x, global1, kw=x)
        print(b + fn(x, global1, kw=x))
        """)
    assert expected_code == fixed_code


def test_fix_1_global_1_positional_0_default_1_starArg_0_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(a, *arg):
            return(global1 + a + arg[0])
        global1 = 9
        a = 9
        alist = [9]
        fn(a, *alist)
        y = 9
        b = fn(y, 9, 9)
        print(b + fn(y, 9, 9))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(a, global1, *arg):' in fixed_code
    assert 'fn(a, global1, *alist)' in fixed_code
    assert 'b = fn(y, global1, 9, 9)' in fixed_code
    assert 'print(b + fn(y, global1, 9, 9))' in fixed_code

    expected_code = _reformat(
        """
        def fn(a, global1, *arg):
            return(global1 + a + arg[0])
        global1 = 9
        a = 9
        alist = [9]
        fn(a, global1, *alist)
        y = 9
        b = fn(y, global1, 9, 9)
        print(b + fn(y, global1, 9, 9))
        """)
    assert expected_code == fixed_code


def test_fix_1_global_1_positional_0_default_0_starArg_1_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(a, **kwargs):
            print(global1 + a + kwargs['kw1'])
        global1 = 9
        a = 9
        fn(a, kw1=9, kw2=9)
        x = 9
        adict = dict([('kw1', 9), ('kw2', 9)])
        b = fn(x, **adict)
        print(b + fn(x, **adict))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(a, global1, **kwargs):' in fixed_code
    assert 'fn(a, global1, kw1=9, kw2=9)' in fixed_code
    assert 'b = fn(x, global1, **adict)' in fixed_code
    assert 'print(b + fn(x, global1, **adict))' in fixed_code

    expected_code = _reformat(
        """
        def fn(a, global1, **kwargs):
            print(global1 + a + kwargs['kw1'])
        global1 = 9
        a = 9
        fn(a, global1, kw1=9, kw2=9)
        x = 9
        adict = dict([('kw1', 9), ('kw2', 9)])
        b = fn(x, global1, **adict)
        print(b + fn(x, global1, **adict))
        """)
    assert expected_code == fixed_code


def test_simple_fix_2_globals():
    nb = _reformat(
        """
        # ## section1

        def fn():
            return(global1 + global2)
        global1 = 9
        global2 = 9
        fn()
        a = fn()
        print(a + fn())
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(global1, global2)' in fixed_code
    assert 'a = fn(global1, global2)' in fixed_code
    assert 'print(a + fn(global1, global2))' in fixed_code

    expected_code = _reformat(
        """
        def fn(global1, global2):
            return(global1 + global2)
        global1 = 9
        global2 = 9
        fn(global1, global2)
        a = fn(global1, global2)
        print(a + fn(global1, global2))
        """)
    assert expected_code == fixed_code


def test_fix_2_globals_2_positional_0_default_0_starArg_0_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(a, b):
            return(a + b + global1 + global2)
        global1 = 9
        global2 = 9
        a = 9
        b = 9
        fn(a, b)
        x = 9
        y = 9
        z = fn(x, y)
        print(a + fn(x, y))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(a, b, global1, global2)' in fixed_code
    assert 'fn(a, b, global1, global2)' in fixed_code
    assert 'z = fn(x, y, global1, global2)' in fixed_code
    assert 'print(a + fn(x, y, global1, global2))' in fixed_code

    expected_code = _reformat(
        """
        def fn(a, b, global1, global2):
            return(a + b + global1 + global2)
        global1 = 9
        global2 = 9
        a = 9
        b = 9
        fn(a, b, global1, global2)
        x = 9
        y = 9
        z = fn(x, y, global1, global2)
        print(a + fn(x, y, global1, global2))
        """)
    assert expected_code == fixed_code


def test_fix_2_globals_2_positional_2_default_0_starArg_0_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(a, b, d=9, e=9):
            return(a + b + global1 + global2 + d + e)
        global1 = 9
        global2 = 9
        a = 9
        b = 9
        fn(a, b, d=8, e=8)
        x = 1
        y = 1
        z = fn(x, y, d=8, e=8)
        print(a + fn(x, y, d=8, e=8))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(a, b, global1, global2, d=9, e=9)' in fixed_code
    assert 'fn(a, b, global1, global2, d=8, e=8)' in fixed_code
    assert 'z = fn(x, y, global1, global2, d=8, e=8)' in fixed_code
    assert 'print(a + fn(x, y, global1, global2, d=8, e=8))' in fixed_code

    expected_code = _reformat(
        """
        def fn(a, b, global1, global2, d=9, e=9):
            return(a + b + global1 + global2 + d + e)
        global1 = 9
        global2 = 9
        a = 9
        b = 9
        fn(a, b, global1, global2, d=8, e=8)
        x = 1
        y = 1
        z = fn(x, y, global1, global2, d=8, e=8)
        print(a + fn(x, y, global1, global2, d=8, e=8))
        """)
    assert expected_code == fixed_code


def test_fix_2_globals_2_positional_0_default_1_starArg_0_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(a, b, *arg):
            return(a + b + global1 + global2 + arg[0])
        global1 = 9
        global2 = 9
        a = 9
        b = 9
        alist = [9]
        fn(a, b, *alist)
        x = 9
        y = 9
        z = fn(x, y, 9, 9)
        print(a + fn(x, y, 9, 9))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(a, b, global1, global2, *arg):' in fixed_code
    assert 'fn(a, b, global1, global2, *alist)' in fixed_code
    assert 'z = fn(x, y, global1, global2, 9, 9)' in fixed_code
    assert 'print(a + fn(x, y, global1, global2, 9, 9))' in fixed_code

    expected_code = _reformat(
        """
        def fn(a, b, global1, global2, *arg):
            return(a + b + global1 + global2 + arg[0])
        global1 = 9
        global2 = 9
        a = 9
        b = 9
        alist = [9]
        fn(a, b, global1, global2, *alist)
        x = 9
        y = 9
        z = fn(x, y, global1, global2, 9, 9)
        print(a + fn(x, y, global1, global2, 9, 9))
        """)
    assert expected_code == fixed_code


def test_fix_2_globals_2_positional_0_default_0_starArg_1_starKWarg():
    nb = _reformat(
        """
        # ## section1

        def fn(a, b, **kwargs):
            return(a + b + global1 + global2 + kwargs['kw1'])
        global1 = 9
        global2 = 9
        a = 9
        b = 9
        fn(a, b, kw1=9, kw2=9)
        x = 9
        y = 9
        adict = dict([('kw1', 9), ('kw2', 9)])
        z = fn(x, y, **adict)
        print(a + fn(x, y, **adict))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()
    assert 'def fn(a, b, global1, global2, **kwargs):' in fixed_code
    assert 'fn(a, b, global1, global2, kw1=9, kw2=9)' in fixed_code
    assert 'z = fn(x, y, global1, global2, **adict)' in fixed_code
    assert 'print(a + fn(x, y, global1, global2, **adict))' in fixed_code

    expected_code = _reformat(
        """
        def fn(a, b, global1, global2, **kwargs):
            return(a + b + global1 + global2 + kwargs['kw1'])
        global1 = 9
        global2 = 9
        a = 9
        b = 9
        fn(a, b, global1, global2, kw1=9, kw2=9)
        x = 9
        y = 9
        adict = dict([('kw1', 9), ('kw2', 9)])
        z = fn(x, y, global1, global2, **adict)
        print(a + fn(x, y, global1, global2, **adict))
        """)
    assert expected_code == fixed_code


def test_fix_3_globals_3_positional_0_default_1_starArg_1_starKWarg(tmp_empty):
    nb = _reformat(
        """
        # ## section1

        def fn(a, b, c, *args, **kwargs):
            return(a + b + c
                   + global1 + global2 + global3
                   + args[0] + kwargs['kw1'])
        global1 = global2 = global3 = 9
        a = b = c = 9
        alist = [9]
        fn(a, b, c, *alist, kw1=9, kw2=9)
        x = y = z = 9
        adict = dict([('kw1', 9), ('kw2', 9)])
        d = fn(x, y, z, 9, 9, **adict)
        print(a + fn(x, y, z, 9, 9, **adict))
        """)
    fixed_code = NotebookExporter(_read(nb))._get_code()

    fndef = 'def fn(a, b, c, global1, global2, global3, *args, **kwargs):'
    assert fndef in fixed_code

    call1 = 'fn(a, b, c, global1, global2, global3, *alist, kw1=9, kw2=9)'
    assert call1 in fixed_code

    call2 = 'd = fn(x, y, z, global1, global2, global3, 9, 9, **adict)'
    assert call2 in fixed_code

    call3 = 'print(a + fn(x, y, z, global1, global2, global3, 9, 9, **adict))'
    assert call3 in fixed_code

    expected_code = _reformat(
        """
        def fn(a, b, c, global1, global2, global3, *args, **kwargs):
            return(a + b + c
                   + global1 + global2 + global3
                   + args[0] + kwargs['kw1'])
        global1 = global2 = global3 = 9
        a = b = c = 9
        alist = [9]
        fn(a, b, c, global1, global2, global3, *alist, kw1=9, kw2=9)
        x = y = z = 9
        adict = dict([('kw1', 9), ('kw2', 9)])
        d = fn(x, y, z, global1, global2, global3, 9, 9, **adict)
        print(a + fn(x, y, z, global1, global2, global3, 9, 9, **adict))
        """)
    assert expected_code == fixed_code

    # sanity check: would DAG be able to build?
    export.from_nb(_read(nb))
    dag = DAGSpec('pipeline.yaml').to_dag()
    dag.build()
    source = str(dag['section1'].source)
    assert expected_code in source


def test_fix_complex_nb(tmp_empty):
    nb = _reformat(
        """
        # ## section1

        # comment 1
        def fn1():
            return(global_1)

        def fn2(a): # comment 2
            return(a + fn1())

        def fn3(a, b):
            return(fn2(a) + b)

        def fn4(a, b):
            return(fn3(a, b) + global_2)

        def fn5(a, b, kw1=1):
            return(fn4(a, b) + kw1)

        global_1 = 1 # used inside a func without declaring it as a param
        global_2 = 1
        a = 1
        b = 2

        # # ## section2

        x = 8; y = 8

        def fn6(x, y, kw1=8, kw2=8):
            return(fn5(x, y, kw1=kw1))

        def fn7(x, y, *args, kw1=8, kw2=8):
            return(fn6(x, y, kw1=kw1, kw2=kw2))

        def fn8(x, y, *args, kw1=8, kw2=8, **kwargs):
            return(fn7(x, y, 8, kw1=kw1, kw2=kw2))


        # comment 3
        def fn9(x, y, *args, kw1=8, kw2=8, **kwargs):
            return(fn8(x, y, *args, kw1=kw1, kw2=kw2) + global_3)

        # no call funcs w * in another func - creo q es mucho para este PR

        global_3 = 1
        alist = [8] # comment 4

        print(1 + 1 + fn9(x, y, *alist, kw1=8, kw2=8, kw3=8))
        """)

    fixed_code = NotebookExporter(_read(nb))._get_code()

    assert 'comment 1' in fixed_code
    assert 'comment 2' in fixed_code
    assert 'comment 3' in fixed_code
    assert 'comment 4' in fixed_code

    assert 'def fn1(global_1):' in fixed_code
    assert 'def fn2(a, global_1):' in fixed_code
    assert 'return(a + fn1(global_1))' in fixed_code
    assert 'def fn3(a, b, global_1):' in fixed_code
    assert 'return(fn2(a, global_1) + b)' in fixed_code

    # 1st need global_2, but after fixing fn3(), now it also needs global_1
    assert 'def fn4(a, b, global_2, global_1):' in fixed_code
    assert 'return(fn3(a, b, global_1) + global_2)' in fixed_code
    assert 'def fn5(a, b, global_2, global_1, kw1=1):' in fixed_code
    assert 'return(fn4(a, b, global_2, global_1) + kw1)' in fixed_code
    assert 'def fn6(x, y, global_2, global_1, kw1=8, kw2=8):' in fixed_code
    assert 'return(fn5(x, y, global_2, global_1, kw1=kw1))' in fixed_code

    fn7def = 'def fn7(x, y, global_2, global_1, *args, kw1=8, kw2=8):'
    assert fn7def in fixed_code

    fn6call = 'return(fn6(x, y, global_2, global_1, kw1=kw1, kw2=kw2))'
    assert fn6call in fixed_code

    fn8def = ('def fn8(x, y, global_2, global_1,'
              ' *args, kw1=8, kw2=8, **kwargs):')
    assert fn8def in fixed_code

    fn7call = ('return(fn7(x, y, global_2,'
               ' global_1, 8, kw1=kw1, kw2=kw2))')
    assert fn7call in fixed_code

    fn9def = ('def fn9(x, y, global_3, global_2,'
              ' global_1, *args, kw1=8, kw2=8, **kwargs):')
    assert fn9def in fixed_code

    fn8call = ('return(fn8(x, y, global_2, global_1,'
               ' *args, kw1=kw1, kw2=kw2) + global_3)')
    assert fn8call in fixed_code

    fn9call = ('print(1 + 1 + fn9(x, y, global_3, global_2,'
               ' global_1, *alist, kw1=8, kw2=8, kw3=8))')
    assert fn9call in fixed_code

    # sanity check: would DAG be able to build?
    export.from_nb(_read(nb))
    dag = DAGSpec('pipeline.yaml').to_dag()
    dag.build()
    source = str(dag['section1'].source)
    assert 'print(1 + 1 + fn9(x, y, global_3' in source


# Failing to detect global in class method.
#    In export.py, find_inputs_and_outputs does not
#        detect class methods that use globals
#    Is fixing this a priority ?
#    see: https://github.com/ploomber/soorgeon/issues/26
# def test_fix_1_global_in_class_method():
#     nb = _reformat(
#         """
#         # ## section1

#         class C:
#             def amethod(self):
#                 return global1
#         global1 = 9
#         c = C()
#         c.amethod()
#         print(c.amethod())
#         """)
#     fixed_code = NotebookExporter(_read(nb))._get_code()
#     assert 'def amethod(self, global1):' in fixed_code # fail
