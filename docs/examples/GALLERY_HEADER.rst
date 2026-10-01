Case examples
=============

Each page asks one question about a PV and battery system and answers it with
a stored BREOS run. The configuration is shown on the page, the numbers in the
text are read from the stored results, and every page says which BREOS
version produced them.

The results were computed once by ``tools/regenerate_gallery_results.py``
from the configuration files in ``configs/examples/`` and the PVGIS weather
committed in the repository. The documentation build only loads and plots
them, so it needs no network and simulates nothing. To run a case yourself,
use the command shown on its page.

The packaged cost presets, tariffs and load profile make these runs
reproducible; they are not advice for a real project. Put your own prices,
load and weather in their place.

For short, task-focused instructions without a stored run, see the
:doc:`how-to guides </how-to/index>`.
