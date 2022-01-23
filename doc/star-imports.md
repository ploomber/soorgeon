# Fixing star imports

Star imports are not supported. Here's an example:

```python
from math import *
```

To fix it, replace the `*` with the elements you're using.

Say your code looks like this:

```python
from math import *

result = log(e)
```

You can fix it by changing the `*` for `log, e`:

```python
from math import log, e

result = log(e)
```
