# Data Manifest

Public Numerai Tournament snapshot downloaded through NumerAPI on 2026-08-10 during round 1329. Remote object paths were under `20260807/v5.3/`. Local data are deliberately outside Git at `/media/titus/big/tmp/numerai-v5.3-competitive/`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `features.json` | 386,826 | `27de6b598b0d479415dba8062d050fc190469776f9c905feea9ee3f2bdda3631` |
| `train.parquet` | 3,296,841,026 | `bae773ddd7eea6ed07c55d87b882cc061901abbb724972b2a630381417c328f8` |
| `validation.parquet` | 5,601,474,091 | `62bb9a587ecdcb5f3095809de276da381a803b699e905a82b962d2e4d35295c0` |
| `train_benchmark_models.parquet` | 65,008,010 | `5a8729941c481abe95236920663e1d3cb1140407cc743e3a7f71afbf78f80248` |
| `validation_benchmark_models.parquet` | 126,282,139 | `e771f395d689cee948435656af64bf789ea67c4421a5bc7cd568f83e85faf7d2` |

Observed contract: train has 2,746,268 rows and eras 0001–0574; validation has 4,107,040 rows and eras 0575–1231. Generic `target` is resolved through era 1218 and exactly equals `target_ender_60`; eras 1219–1231 currently have no generic target. Benchmark columns are `v53_lgbm_ender20` and `v53_lgbm_ender60`.
