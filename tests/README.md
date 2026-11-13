# Real radiograph references

The runner and full-network tests use the CC0 chest radiograph by Mikael
Häggström from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg).
They read `~/.cache/medmlx/test-data/chest_xray.jpg` by default;
`MEDMLX_TEST_XRAY` selects another local copy. File-format cases convert the
same radiograph to the relevant raster or DICOM encoding.

To regenerate the runner references, use the pinned FleXray source and a
Torch 2.6.0 environment with its inference dependencies and pydicom:

```sh
python scripts/record_runner_reference.py --source PATH_TO_FLEXRAY \
  --torch-python PATH_TO_TORCH_PYTHON
python scripts/record_tta_reference.py --source PATH_TO_FLEXRAY \
  --torch-python PATH_TO_TORCH_PYTHON
```

Image-bearing arrays are written to `work/real-reference`, outside Git;
`MEDMLX_TEST_REFERENCES` selects another local reference directory.
Augmentation inputs use resized and reflected views of the same radiograph;
the constant and singleton normalization cases repeat a sourced pixel.
