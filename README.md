# MAVEpolish

Quality control and reconstruction for Multiplexed Assays of Variant Effect (MAVE) data.

MAVEpolish uses dictionary learning to reconstruct variant effect maps (VEMs), providing quality metrics and polished score estimates for every variant.

## Installation

Requires Python 3.9+.

```
git clone https://github.com/gkudla/mavepolish.git
cd mavepolish
pip install .
```

This installs `mavepolish`, `to_vem`, and `mavepolish-web` commands that work in any new terminal window.

<details>
<summary>Optional: install in a virtual environment</summary>

If you prefer to keep MAVEpolish isolated from your system Python:

```
cd mavepolish
python3 -m venv venv
source venv/bin/activate   # run this each time you open a new terminal
pip install .
```

</details>

<details>
<summary>For developers</summary>

Use an editable install so code changes take effect immediately:

```
pip install -e .
```

</details>

## Usage

### Web app

```
mavepolish-web
```

Then open http://localhost:8051

### Command line

Convert raw MAVE data to VEM format:

```
to_vem -i my_data.csv
```

Run quality control analysis:

```
mavepolish -t my_data.VEM.tsv -e my_data.VEM.tsv
```

Use the pretrained model for faster analysis:

```
mavepolish -m pretrained_model.pkl -e my_data.VEM.tsv
```

Run `mavepolish --help` and `to_vem --help` for all options.
