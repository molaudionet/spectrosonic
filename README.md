# Molecular AI Project - Spectroscopic Sonification

Audio-based molecular AI using spectroscopic data (NMR, IR, CD, MS) for proteins, DNA, and RNA.

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas scipy librosa transformers torch

# 2. Run demo
cd demos
python demo_standalone.py

# 3. Listen to generated audio
# Audio files in: ../output/audio/
```

## 📁 Project Structure

```
molecular_ai_project/
├── README.md                 # This file
├── docs/                     # Documentation
│   ├── BIOMOLECULAR_INTEGRATION.md
│   ├── BIOMOLECULAR_QUICKSTART.md
│   ├── MACROMOLECULAR_SONIFICATION.md
│   ├── SPECTROSCOPY_BIOMOLECULES.md
│   ├── SPECTROSCOPY_REAL_WORLD.md
│   ├── DATASETS_COMPLETE_SUMMARY.md
│   └── DATASETS_QUICK_REFERENCE.md
├── src/                      # Source code
│   ├── biomolecular_sonification.py
│   ├── spectroscopy_to_audio.py
│   └── audio_featurizer_patched.py
├── demos/                    # Demo scripts
│   ├── demo_standalone.py
│   └── demo_real_spectroscopy_data.py
├── data/                     # Spectroscopic datasets
│   ├── nmr/
│   │   ├── ubiquitin_hsqc.csv
│   │   └── dna_dodecamer_imino.csv
│   ├── cd/
│   │   └── lysozyme_cd.csv
│   ├── ir/
│   │   └── bsa_ftir.csv
│   └── ms/
│       └── insulin_esi_ms.csv
└── output/                   # Generated files
    └── audio/
        ├── ubiquitin_hsqc_demo.wav
        ├── lysozyme_cd_spectrum.wav
        ├── lysozyme_cd_structure.wav
        ├── bsa_ftir_demo.wav
        ├── insulin_multimodal.wav
        └── multimodal_demo.wav
```

## 📊 Datasets

**5 real spectroscopic datasets** from published sources:

1. **Ubiquitin NMR** (BMRB 6457) - 73 residues, 1H-15N HSQC
2. **Lysozyme CD** (PCDDB) - α-helix signature
3. **BSA FTIR** - IR amide bands
4. **DNA Dodecamer** - B-DNA structure
5. **Insulin MS** - ESI mass spectrum

## 🎵 Audio Files

**7 generated audio files** ready to use:
- Play in any audio player
- Extract Wav2Vec 2.0 embeddings
- Use for ML prediction

## 📚 Documentation

- **Quick Start:** `docs/BIOMOLECULAR_QUICKSTART.md`
- **Technical:** `docs/SPECTROSCOPY_BIOMOLECULES.md`
- **Real-world usage:** `docs/SPECTROSCOPY_REAL_WORLD.md`
- **Complete guide:** `docs/DATASETS_COMPLETE_SUMMARY.md`

## 🚀 Usage

### Run Demo
```bash
cd demos
python demo_standalone.py
```

### Extract Embeddings
```python
from scipy.io import wavfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load audio
sr, audio = wavfile.read('../output/audio/ubiquitin_hsqc_demo.wav')
audio = audio.astype(float) / 32767.0

# Extract embeddings
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # (768,)
```

## ✅ Features

- ✅ Real spectroscopic data (BMRB, PCDDB, literature)
- ✅ Audio conversion (NMR, CD, IR, MS → WAV)
- ✅ Wav2Vec 2.0 integration
- ✅ Multi-modal fusion
- ✅ Complete documentation
- ✅ Working demos

## 📖 References

- Zhou & Zhou (2026) - Molecular Sonification paper
- BMRB: http://www.bmrb.wisc.edu
- PCDDB: https://pcddb.cryst.bbk.ac.uk
- PRIDE: https://www.ebi.ac.uk/pride/

---

**Ready to use!** Start with `demos/demo_standalone.py`
