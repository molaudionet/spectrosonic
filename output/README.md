# Generated Audio Files

This directory contains audio files generated from real spectroscopic data.

## Files

- `ubiquitin_hsqc_demo.wav` - NMR protein structure
- `lysozyme_cd_spectrum.wav` - CD spectrum
- `lysozyme_cd_structure.wav` - Secondary structure signature
- `bsa_ftir_demo.wav` - IR spectrum
- `insulin_multimodal.wav` - Multi-modal (NMR+IR+MS)
- `multimodal_demo.wav` - Synthetic multi-modal demo

## Usage

Play in any audio player or extract embeddings:

```python
from scipy.io import wavfile

sr, audio = wavfile.read('audio/ubiquitin_hsqc_demo.wav')
# Process with Wav2Vec 2.0...
```
