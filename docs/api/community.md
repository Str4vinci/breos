# Community facade

{py:class}`~breos.Community` is the entry point for collective self-consumption
beside the single-site [App facade](app.md). It accepts a configuration
mapping, runs `simulate()`, and exposes typed accounts with `result()` and a
copy of the first year's Wh frame with `settlement()`. Both accessors raise
`RuntimeError` before simulation. See the
[community guide](../getting-started/community.md) for the configuration and
[settlement and results](community-settlement.md) for every field.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.Community
```

The result type is available from `breos.community.results`. Like App's
result, it has no `summary()` or `to_dict()` method; unlike App's plain dict,
it contains pandas account tables. The root `breos` package exports only
`Community`, since App has no exported result class either.

```{eval-rst}
.. autoclass:: breos.community.results.CommunityResult
   :members:

.. autoclass:: breos.community.money.ParticipantAccount
   :members:

.. autoclass:: breos.community.money.AssetAccount
   :members:
```
