# Functions with global variables

Currently, Soorgeon does not support functions that use variables defined out of its local scope, for example:

```python
z = 1

def my_function(x, y):
    result = x + y + z # z's value is not a function argument or a local variable!
    return result
```

If you attempt to use `soorgeon refactor` with a notebook that has a function like that, you'll see the following error:

```
Looks like the following functions are using global variables, this is unsupported. Please add all missing arguments.

Function 'my_function' uses variables 'z'
```

To fix it, add the offending variables as arguments:

```python
# add z as argument
def my_function(x, y, z):
    result = x + y + z
    return result
```

And modify any calls to that function:

```python
# before
my_function(x, y)

# after
my_function(x, y, z)
```