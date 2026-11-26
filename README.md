# HiLoNet

Code for *HiLoNet: A Cross-Domain Wireless Sensing Framework for Human Activity Recognition in Smart Homes* (ICONIP 2026). [Paper](https://doi.org/10.1007/978-981-92-8113-8_23)

HiLoNet recognizes human activities from radio-frequency signals. It classifies the 55 activities of the [XRF55](https://aiotgroup.github.io/XRF55/) dataset using WiFi, RFID or mmWave data, with a separate model for each modality.

Each model converts its input to the frequency domain and learns a soft mask `M(f)` over the frequency bins. The masked spectrum `M(f)` goes to a Transformer and the remainder `1 - M(f)` goes to a CNN. The two streams are then combined with learned per-class weights.

## Setup

Tested on Python 3.10 with PyTorch 2.7.1. An NVIDIA GPU is recommended; without one, the scripts run on the CPU.

```bash
pip install -r requirements.txt
```

## Data

Download [XRF55](https://www.kaggle.com/xrfdataset/xrf55) and arrange it as follows, next to the repository folder, with the commands below run from the repository root. To use other locations, edit `RAW_DATA_ROOT` and `SPLIT_OUTPUT_ROOT` at the top of `split_train_test.py`, then re-run it; every script reads them from there.

```
../Datasets/XRF55/
├── Scene1/
│   ├── WiFi/      XX_XX_XX.npy    # shape (270, 1000)
│   ├── RFID/      XX_XX_XX.npy    # shape (23, 1000)
│   └── mmWave/    XX_XX_XX.npy    # shape (17, 256, 128)
├── Scene2/
├── Scene3/
└── Scene4/
```

Files are named `subject_activity_sample.npy`. Then create the splits and label files:

```bash
python split_train_test.py              # default split: symlinks the data into dataset/XRF_dataset/ and writes label files
python split_train_test.py --mode 1     # another split (see the table below)
```

All four scenes are then loaded through `dataset/XRF_dataset/` and its label files. Keep `../Datasets/XRF55/` in place afterwards, because those links point to it.

Scene 1 is split by subject, in order of subject id, using the splits from the paper. Choose one with `--mode`:

| `--mode` | Scene 1 split | Used in the paper for |
|---|---|---|
| `0` (default) | 21/9 subjects: the last 9 are the test set; for the first 21, the first 14 samples of each activity train and the rest validate | Tables 2 and 3 |
| `1` | 20/10 subjects: first 20 train, last 10 validate | Fold 1, Tables 4 and 5 |
| `2` | 20/10 subjects: last 20 train, first 10 validate | Fold 2, Tables 4 and 5 |
| `3` | 20/10 subjects: 11th to 20th validate, all others train | Fold 3, Tables 4 and 5 |

Scenes 2, 3 and 4 are used only for testing. Each run replaces the previous split and writes its label files, so train and evaluate one split before creating the next. Results are kept per split in `result/params/split<mode>/` and `result/eval/split<mode>/`, so different splits don't overwrite each other.

## Training

```bash
python train_hiloNet.py                               # all three modalities, one after another
python train_hiloNet.py --modality wifi --gpu 1       # one modality on a chosen GPU
python train_hiloNet.py --help                        # all options
```

Models always train on scene 1, using the split created above. Checkpoints are saved to `result/params/split<mode>/<model_name>/`, and retraining on the same split overwrites them.

The defaults are 200 epochs, batch size 32, and AdamW with a learning rate of 0.0001 and weight decay of 0.001. Training also uses a cosine learning-rate schedule, gradient clipping at a norm of 1.0, and a mask regularization weight (`λ`) of 0.01.

## Evaluation

```bash
python eval_hiloNet.py                                # all three modalities on all scenes
python eval_hiloNet.py --modality rfid --scene 3      # one modality on scene 3 test data (also 1, 2 or 4)
python eval_hiloNet.py --scene val                    # scene 1 validation data only
```

Evaluation uses the checkpoints from the active split, meaning the one most recently created by `split_train_test.py`. By default, each model is evaluated on scene 1 validation data, scene 1 test subjects (split mode 0 only) and the test data of scenes 2, 3 and 4. The run prints the range of the learned mask and a table of accuracy, precision, recall and F1 for each scene, followed by the same table for soft voting (averaged class probabilities) over every combination of the evaluated modalities. It saves a summary and a confusion matrix to `result/eval/split<mode>/<model_name>/<scene>_<data>/`, for example `wifi_hilonet_SMR/scene2_test_data` or `ensemble_wifi_rfid/scene2_test_data`.

Scene 1's validation data also selects the best checkpoint during training, so results on it are optimistic. This includes modes 1, 2 and 3, where the validation subjects are the only unseen scene 1 subjects: those modes have no separate test set, so the default run skips it and `--scene 1` fails with an error about an empty label file.

Missing or corrupt data files are replaced with zeros and logged as warnings. Check for these warnings before relying on a result.

## Repository layout

| File | Purpose |
|---|---|
| `train_hiloNet.py` | Models and training for all three modalities |
| `eval_hiloNet.py` | Evaluation of the trained models |
| `XRFDataset.py` | Dataset loader |
| `split_train_test.py` | Creates the train, validation and test splits, and sets the data paths |
| `generate_txt.py` | Creates the label files (run automatically by `split_train_test.py`) |
| `check_dataset.py` | Optional dataset checks; see the usage notes at the top of the file |

## Citation

If you use this code, please cite:

> Ullah, H.A., Zhou, Y., Sheng, Q.Z. (2027). HiLoNet: A Cross-Domain Wireless Sensing Framework for Human Activity Recognition in Smart Homes. In: Bhatti, A., et al. *Neural Information Processing*. ICONIP 2026. Communications in Computer and Information Science, vol 3199. Springer, Singapore. https://doi.org/10.1007/978-981-92-8113-8_23

```bibtex
@inproceedings{ullah2027hilonet,
  title={HiLoNet: A Cross-Domain Wireless Sensing Framework for Human Activity Recognition in Smart Homes},
  author={Ullah, Hadiqa Aman and Zhou, Yipeng and Sheng, Quan Z.},
  editor={Bhatti, A. and others},
  booktitle={Neural Information Processing (ICONIP 2026)},
  series={Communications in Computer and Information Science},
  volume={3199},
  publisher={Springer},
  address={Singapore},
  year={2027},
  doi={10.1007/978-981-92-8113-8_23}
}
```

This work uses the XRF55 dataset. Please also cite:

```bibtex
@article{wang2024xrf55,
  title={XRF55: A Radio Frequency Dataset for Human Indoor Action Analysis},
  author={Wang, Fei and Lv, Yizhe and Zhu, Mengdie and Ding, Han and Han, Jinsong},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  issue={1},
  volume={8},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

## License

MIT. See [LICENSE](LICENSE). This project builds on the [XRF55 code](https://github.com/aiotgroup/XRF55-repo) by aiotgroup, also under the MIT License.
