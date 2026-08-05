# Models, assumptions, and data

BREOS combines weather, PV production, load, battery dispatch and degradation,
economics, and emissions in one simulation. This section collects the pages
that explain the modeled boundaries and the assumptions you should record for
a reproducible study.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} PV production
:link: ../api/pv
:link-type: doc

Plane-of-array irradiance, module performance, losses, arrays, tracking, and
inverter conversion.
:::

:::{grid-item-card} Energy balance
:link: ../api/energy-balance
:link-type: doc

DC-coupled dispatch, inverter boundaries, battery routing, conservation, and
reported KPIs.
:::

:::{grid-item-card} Battery models
:link: ../api/degradation-models
:link-type: doc

Native and BLAST degradation models, selection rules, provenance, and known
limitations.
:::

:::{grid-item-card} Required inputs
:link: ../getting-started/inputs
:link-type: doc

Weather, load, component, tariff, emissions, and reproducibility inputs for a
real study.
:::

:::{grid-item-card} Data resources
:link: ../resources
:link-type: doc

Official weather, PV module, solar-resource, and load-profile sources.
:::

:::{grid-item-card} Load profile data
:link: ../legal/load-profile-data
:link-type: doc

Bundled and external residential load profiles, expected file formats,
official sources, and redistribution boundaries.
:::

::::

## Reading model documentation

The [guides](../getting-started/index.md) explain how to run a study. The
[API reference](../api/index.md) documents callable Python interfaces. Pages
in this section instead answer questions such as “what physical boundary is
being modeled?”, “which assumptions affect this result?”, and “what evidence
supports this model choice?”.
