from numerai_competitive.search_space import generate_search


def test_search_is_symmetric_and_deterministic():
    first, second = generate_search(n=40), generate_search(n=40)
    assert first == second
    adamw = {x["config_id"]: x for x in first if x["arm"] == "adamw"}
    spectral = {x["config_id"]: x for x in first if x["arm"] == "spectral"}
    assert len(adamw) == len(spectral) == 40
    spectral_only = {"rank", "decay", "filter_strength", "filter_warmup",
                     "filter_update_every", "filter_mode"}
    for key in adamw:
        left = {k: v for k, v in adamw[key].items() if k != "arm"}
        right = {k: v for k, v in spectral[key].items()
                 if k not in spectral_only and k != "arm"}
        assert left == right
