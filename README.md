# MAVEpolish

Quality control and reconstruction for Multiplexed Assays of Variant Effect (MAVE) data.

MAVEpolish uses dictionary learning to reconstruct variant effect maps (VEMs), providing quality metrics and polished score estimates for every variant.

## Installation

```
git clone https://github.com/gkudla/mavepolish.git
cd mavepolish
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

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
