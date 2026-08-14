# Spectral optimizer Numerai presentation

This 30-slide Beamer deck summarizes experiments 001–008, including the corrected
parameter-space implementation, low- and high-rank studies, overparameterized
training trajectories, the bounded no-leakage Numerai protocol, the audited
outer result, leaderboard context, limitations, and proposed follow-up work.

Rebuild from the repository root:

```bash
python3 presentation/generate_figures.py
cd presentation
tectonic spectral_numerai_results.tex
```

The committed PDF is `spectral_numerai_results.pdf`. Data under `data/` are
copies of the immutable audited MATS artifacts; `vendor/` contains plots from
earlier experiments used to explain the research chronology.
