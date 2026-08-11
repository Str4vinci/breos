# Appendix

Modules that are re-exported from the `breos` namespace but aren't part of
the primary puzzle-piece surface — utilities, I/O helpers, model constants,
and research-validation modules.

Some article-scoped and report helpers on these modules are scheduled for
removal in 0.6.0. See [Deprecations for 0.6.0](../deprecations.md) for the
complete inventory and migration guidance.

The deprecated `polysun_degradation` name denotes an independent,
documentation-derived comparison approximation. It is not Polysun source code
or a validated reproduction of the Polysun product.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :recursive:

   breos.utils
   breos.io
   breos.constants
   breos.polysun_degradation
```
