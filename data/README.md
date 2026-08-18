# Data layout

Large, licensed, and generated arrays are excluded from Git. Local execution
uses the following stage-aware layout:

```text
data/
├── gom/
│   ├── raw/
│   ├── synthetic/{syn_v001_clean,syn_v002_fieldcal_noiseRMO}/
│   ├── datasets/ds_v001_syn2d_ang3_p50x100_s10x25_sliceHoldout/
│   ├── attributes/
│   └── usable/
├── sleipner/
│   ├── raw/{horizons,offsets,wells}/
│   ├── datasets/ds_v001_field2d_ang3_p50x100_s10x25_blockHoldout/
│   ├── attributes/{horizons_mapped,dip,rgt}/
│   └── usable/{stacks,velocity,angles}/
├── avo/s01/bundles/
└── s01data/
    ├── raw/{well,horizon,seismic}/
    ├── usable/<version>/
    ├── attributes/<version>/
    ├── derived/<version>/
    ├── synthetic/<version>/
    ├── datasets/<version>/
    └── bundles/<version>/
```

`sage_avo.data.DataLayout` keeps the authorized raw-data root read-only while
placing all generated products under this local tree. Copy
`configs/paths.example.yaml` to the ignored `configs/paths.yaml` to configure
the external S01 raw folder and this repository's writable data root.
