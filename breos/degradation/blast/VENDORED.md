# BLAST-Lite vendoring manifest

BREOS includes a transformed subset of BLAST-Lite 1.1.0 under the upstream
BSD-3-Clause license. The source is pinned to NREL/BLAST-Lite commit
`d789e00bca60f628de640745c18eb724b07358bd` (`Correct Tesla Model 3 data source
in README`). GitHub now redirects that repository to
`NatLabRockies/BLAST-Lite`.

The hashes below are SHA-256 hashes of file contents. They distinguish the
pinned input from the transformed BREOS result; only the legal files remain
byte-for-byte identical.

| Upstream path at `d789e00` | Upstream SHA-256 | BREOS path | BREOS SHA-256 | Transform |
|---|---|---|---|---|
| `LICENSE` | `5c608631ff4a0bd83b9768e1f38d8e317ea67c878192eba6222bc5856dae47ea` | `breos/degradation/blast/LICENSE` | `5c608631ff4a0bd83b9768e1f38d8e317ea67c878192eba6222bc5856dae47ea` | L |
| `NOTICE` | `58f1af21812d1b59fb979738e1d886235e182c94a35fb5bd1482cd77f9f7a96a` | `breos/degradation/blast/NOTICE` | `58f1af21812d1b59fb979738e1d886235e182c94a35fb5bd1482cd77f9f7a96a` | L |
| `blast/__init__.py` | `6ab38125c18e1578df64f1dd6703677855c80d715caa8b5793a08fca73069e60` | `breos/degradation/blast/__init__.py` | `c56c9d8cfc98bc102f37f90faecd24774843f349bf851c5c366616f354299239` | P |
| `blast/models/degradation_model.py` | `65a1c013a0b9cc91ff5881e195df6c1723fe05b8b84a6bb0b457969696ec10ab` | `breos/degradation/blast/degradation_model.py` | `170fd2c25029fcc02d08098f27847fa4dbecc47ab6dd45cf774b8a3274e6b347` | B, I, N, F |
| `blast/utils/functions.py` | `2e23fb7cb609ebad34fdca8d2b8d2a75a8b5021c95c3be0c0543a554907fc323` | `breos/degradation/blast/functions.py` | `a6d156c880f206bae1e9422d217fe44dda7388091356552d5ebaadb7b3e34ffd` | E, F |
| `blast/utils/rainflow.py` | `12ee9c0d48877ae7cd13465b4d1a092ef07c9629b72b2b2cb6bd76f705cdfff8` | `breos/degradation/blast/rainflow.py` | `4eacc7d9be9d86eaaa0104c428ee4f76febb8180b078aae20c7f5fb0d0a7f420` | R, F |
| `blast/models/__init__.py` | `5dfc819435065cefdebdbe4763ee00a9c73269ed45e7781e66d18953e2c7df89` | `breos/degradation/blast/models/__init__.py` | `91fec32d1b52ad1a8d4f19b4278131ff2bf33ce34ef5db505bd83716af945203` | R, I, F |
| `blast/models/_available_models.py` | `bf00a095048b83450a8a2056b872c345023a30bae27cbd4d1c61c03498d70971` | `breos/degradation/blast/models/_available_models.py` | `573306835b2471014ca631cd5a26bf7570386c38c590872b82384767369f8399` | R, F |
| `blast/models/lfp_gr_250AhPrismatic_2019.py` | `b145c45a5a7cbb76a4a3d6c0a76b85f127f1a45a5570ecf4b5284bf089bdacf8` | `breos/degradation/blast/models/lfp_gr_250AhPrismatic_2019.py` | `b5a8ed5e73d2753a89ee59a3db6072833afea83a0cfe11982773ec79f11c1544` | R, I, N, F |
| `blast/models/lfp_gr_SonyMurata3Ah_2018.py` | `3d952907b120b23584d6cd7b7644f5e7d288ed1f5f5d92443fec89cbe7d67349` | `breos/degradation/blast/models/lfp_gr_SonyMurata3Ah_2018.py` | `c2e0a364ba3c14602294a3ad7705fecc8dd151f249a33f47f8b5b06a5d7e6b66` | R, I, N, F |
| `blast/models/lmo_gr_NissanLeaf66Ah_2ndLife_2020.py` | `bf5a832f3eb0ecec9bfb48efb8440b90a6f90e628903edfa50fd9cc5d684c689` | `breos/degradation/blast/models/lmo_gr_NissanLeaf66Ah_2ndLife_2020.py` | `f951d902d3815de4cf39fa8e2f6c3bed3cdcbf4403ceb6515c35c5cd10d6b9ef` | R, I, N, F |
| `blast/models/nca_gr_Panasonic3Ah_2018.py` | `1522e76b850ec111c51e87620fe18e63d95da8d485bc1dd5df0fc7522f3d355e` | `breos/degradation/blast/models/nca_gr_Panasonic3Ah_2018.py` | `485f42a5d5cf74ae686e1ffd08a8f6948325ecfbe42fbdf3ce6a615d88a43fbd` | R, I, N, F |
| `blast/models/nca_grsi_SonyMurata2p5Ah_2023.py` | `29da0052d51fbf9b120ff5ea9d95c506aa68013f718120d40b32bd11cb0782ce` | `breos/degradation/blast/models/nca_grsi_SonyMurata2p5Ah_2023.py` | `8157da59d48bd8fb7cfa82720b0ec2e525f838e218ed0b52ed78df282848a005` | R, I, N, F |
| `blast/models/nmc111_gr_Kokam75Ah_2017.py` | `5f6e8e59409aa17d8ca1902553ec344fed6ac0ce9261be3be19cc70e35fde6d3` | `breos/degradation/blast/models/nmc111_gr_Kokam75Ah_2017.py` | `3d5808884a0d2d38cb5a46e12caab25c34b6662ec0390eee382cbd4e134d1408` | R, I, N, F |
| `blast/models/nmc111_gr_Sanyo2Ah_2014.py` | `87aa8a235425f4de2dea38fa45e537715c7c24d878f76a0de50786ac05023887` | `breos/degradation/blast/models/nmc111_gr_Sanyo2Ah_2014.py` | `2a2364adf34d84b1e3c95595a6c2c494f7d8b2a27440f65b7cc0d62a7c7f9703` | R, I, N, F |
| `blast/models/nmc622_gr_DENSO50Ah_2021.py` | `248b24edcb0d2f1cc750012c0fe44c7121b785628ca30a94eef264769bc551f1` | `breos/degradation/blast/models/nmc622_gr_DENSO50Ah_2021.py` | `e27b799253d82e02469c5d250403f1902eddffb5bfcc9fedb2659af13da3bfaa` | R, I, N, F |
| `blast/models/nmc811_grSi_LGM50_5Ah_2021.py` | `0e9f8d2b09d96078f779f0f3462418ab31d44f2018506fb723a15ef4c91cec79` | `breos/degradation/blast/models/nmc811_grSi_LGM50_5Ah_2021.py` | `40bb169ba0d16c7cd13064fbeaef05354cb7d918758d6551e5067672dca0437c` | R, I, N, F |
| `blast/models/nmc811_grSi_LGMJ1_4Ah_2020.py` | `ce41da67f90d51b985902e695e97c13bb13f317e4dde53c9eb22da557f06e2f7` | `breos/degradation/blast/models/nmc811_grSi_LGMJ1_4Ah_2020.py` | `908f19b9c25a66ef348b65cebd79cd774288e3ac09f03d811ee5c86f407c8c8e` | R, I, N, F |
| `blast/models/nmc_gr_50Ah_B1_2020.py` | `84b0c35771b43dcb35179f92d7ad554daf839915f0cac6d2301054534cecfe25` | `breos/degradation/blast/models/nmc_gr_50Ah_B1_2020.py` | `4e919ff1ae7e8b88ce44fb6ca9ffcd838a286f94fe70319dfba238341a9d9af4` | R, I, N, F |
| `blast/models/nmc_gr_50Ah_B2_2020.py` | `f0f92af27620f0cca47866aa19fd8a774f5d6c0f76b54c8a797f1db207dbbadd` | `breos/degradation/blast/models/nmc_gr_50Ah_B2_2020.py` | `4ff5b458ae688c454213243131e250a36b2a251a6beea84355d8a7930d3b88d8` | R, I, N, F |
| `blast/models/nmc_gr_75Ah_A_2019.py` | `ae651c021852e1c1440a1b0b326facf60c502b3d9352c35e2f1e77de9792b7ab` | `breos/degradation/blast/models/nmc_gr_75Ah_A_2019.py` | `7909f9d1f2ff7c276175945fc70ceb28a91e2a5204c98c14cb7bbd8e2eb32320` | R, I, N, F |
| `blast/models/nmc_lto_10Ah_2020.py` | `f08b825e2ea1a0bf5beed552719e02804185b0566c7666f6d6f76ee7111bd1f9` | `breos/degradation/blast/models/nmc_lto_10Ah_2020.py` | `4dab506eee4d937ffe89aa22a94b810ffdac8d18325dfc5612fa7c02d45c5a34` | R, I, N, F |

## Transform key

- **L — legal exact copy:** copied without modification.
- **P — package marker:** replaced the upstream package imports with a BREOS
  package docstring. The model exports remain in the nested `models` package.
- **R — relocation:** moved into the `breos.degradation.blast` namespace.
- **I — import rewrite:** changed imports to package-relative BREOS locations.
- **N — NumPy 2 compatibility:** replaced `np.trapz` calls with the numerically
  equivalent `np.trapezoid` where present.
- **E — extraction:** copied only `rescale_soc` from the upstream utility file;
  NSRDB, geocoding, demo, and other helpers with heavy optional dependencies
  were not vendored.
- **B — base-class dependency trim:** removed runtime pandas and matplotlib
  imports. The DataFrame branch uses the same column/value protocol without a
  hard pandas type check, and its annotation is dependency-neutral.
- **F — repository formatting:** applied Ruff/Black-compatible formatting,
  including whitespace, quote, and line-wrap changes.

No scientific coefficient, equation, or calibration value was intentionally
changed. The transformed models are checked against fixed all-model golden
trajectories in `tests/test_blast_vendoring.py`.

## Derived parity fixture

The fixture is not upstream BLAST-Lite source. It was generated from the
untransformed model behavior in the local preparation commit
`b12e8f377a4b8d93901b54300acbbe2a1f987b95` and copied without modification:

| Preparation path at `b12e8f3` | Preparation SHA-256 | BREOS path | BREOS SHA-256 |
|---|---|---|---|
| `tests/fixtures/breos_vendoring/blast_golden_soh_100d.json` | `30789eaaf6e27ab92a06e8d69032b069de5fbc7a80011bcee78e18e7bca5624e` | `tests/fixtures/blast/blast_golden_soh_100d.json` | `30789eaaf6e27ab92a06e8d69032b069de5fbc7a80011bcee78e18e7bca5624e` |

The BREOS adapter and runner integration (`breos/degradation/engine.py`,
`breos/battery.py`, and `breos/runners/app.py`) are original BREOS code and are
outside this vendored-source table.
