# metrecs

metrics for recommendation systems

![alt text](images/frozen.jpeg)

# getting started

```python
ref = np.array([["a", "a", "b", "c"],
                     ["a", "c", "b", "c"]])

own = np.array(["a", "a", "b", "c"])

import numpy as np
from metrecs import metrics

metrics.fragmentation.fragmentation(pop_recs, own_recs)
```
