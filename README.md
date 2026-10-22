# DANDI reanalysis examples

Full output of the three example reanalyses on our SfN 2026 poster (PSTR339.15, "Conversational AI agents for converting neurophysiology data to NWB and reanalyzing the DANDI Archive").

Each folder is one run of Claude Code (Claude Opus 5.5) with the [analyzing-dandi-datasets](https://github.com/catalystneuro/claude-skills/tree/main/analyzing-dandi-datasets), [using-pynapple](https://github.com/pynapple-org/claude-skills) and [using-nemos](https://github.com/catalystneuro/claude-skills/tree/main/using-nemos) skills, copied here under `.claude/skills/`. Each run started from a single plain-language question. We ran each question several times and kept the run shown on the poster. Everything in a folder (scripts, figures, results, README and executed notebook) was written by the agent; the poster panels are figures from these folders.

| Folder | Question | Dandiset | Poster figure |
|---|---|---|---|
| [spatial](spatial) | How does an animal know where it is located in space? | [DANDI:000582](https://dandiarchive.org/dandiset/000582) (Sargolini et al., 2006) | `figures/05_example_cells.png` |
| [reward](reward) | How does an animal learn what comes before a reward? | [DANDI:001632](https://dandiarchive.org/dandiset/001632) (Burke et al., 2026) | `figures/04_peth_by_session.png` |
| [vision](vision) | How does an animal's brain turn light into a picture of the world? | [DANDI:000021](https://dandiarchive.org/dandiset/000021) (Siegle et al., 2021) | `figures/04_orientation_tuning.png` |

Start with each folder's README and executed notebook.
