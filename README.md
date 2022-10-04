### Update 4 October 2022
The analysis functions used in the Jupyter Notebooks were moved from separated files (i.e., `onset_functions.py`, `onset_widgets.py`, `read_swaves.py`) 
to the package [SEPpy](https://github.com/serpentine-h2020/SEPpy) ([PyPI entry](https://pypi.org/project/seppy/)). 
In the course of this, all .py files except `inf_inj_time.py` became obsolote, but are kept for backwards compatibility. 

Please install the latest version of SEPpy with `pip install seppy` and change the imports in the Notebooks from:
  ``` python
  from onset_functions import *
  import onset_widgets as w
  ```
  to:
  ``` python
  from seppy.tools import Event
  import seppy.tools.widgets as w
  import datetime, os
  ```
