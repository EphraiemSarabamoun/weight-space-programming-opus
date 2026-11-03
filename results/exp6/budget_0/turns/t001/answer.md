```python
import os
print(NAMES)
print(SHAPES)
print(saved())
for root, dirs, files in os.walk('/work'):
    for f in files:
        p = os.path.join(root, f)
        print(p, os.path.getsize(p))
```