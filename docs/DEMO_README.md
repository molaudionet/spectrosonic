# Spectroscopic Data Audio Demos

This directory contains audio files generated from real spectroscopic data.

## Files Generated:

### 1. ubiquitin_hsqc_demo.wav (10 sec)
- **Data source**: BMRB Entry 6457
- **Technique**: 1H-15N HSQC NMR
- **Protein**: Human ubiquitin (76 residues, 8.6 kDa)
- **What you hear**: 73 cross-peaks representing backbone NH groups
- **Interpretation**: Well-dispersed peaks = well-folded protein

### 2. lysozyme_cd_spectrum.wav (8 sec)
- **Data source**: PCDDB Entry CD0000042000
- **Technique**: Far-UV Circular Dichroism
- **Protein**: Hen egg-white lysozyme (129 residues)
- **What you hear**: Wavelength scan from 190-260 nm
- **Interpretation**: Two minima (208, 222 nm) = α-helix signature (36% helix)

### 3. lysozyme_cd_structure.wav (5 sec)
- **Data source**: Secondary structure deconvolution
- **What you hear**: Three tones representing α-helix (36%), β-sheet (12%), coil (52%)
- **Interpretation**: Low freq = helix, mid = sheet, high = coil

### 4. bsa_ftir_demo.wav (12 sec)
- **Data source**: Published FTIR spectrum
- **Technique**: Fourier Transform Infrared Spectroscopy
- **Protein**: Bovine Serum Albumin (583 residues, 66.5 kDa)
- **What you hear**: Wavenumber sweep 1400-1700 cm⁻¹ (amide region)
- **Interpretation**: Peak at 1652 cm⁻¹ = α-helix (52% helix)

### 5. dna_dodecamer_nmr.wav (6 sec)
- **Data source**: BMRB DNA entries, published data
- **Technique**: 1H NMR (imino protons)
- **DNA**: d(CGCGAATTCGCG)2 (Dickerson-Drew dodecamer)
- **What you hear**: 8 peaks representing base-paired protons
- **Interpretation**: G-C pairs (13-14 ppm), A-T pairs (12-13 ppm)

### 6. insulin_ms_demo.wav (1.4 sec)
- **Data source**: Standard ESI-MS analysis
- **Technique**: Electrospray Ionization Mass Spectrometry
- **Protein**: Human insulin (5.8 kDa)
- **What you hear**: 7 peaks representing charge states +1 to +6
- **Interpretation**: Multiple charges typical for ESI, intact protein

### 7. insulin_multimodal.wav (Variable)
- **Data source**: Combined NMR + IR + MS
- **What you hear**: Multi-modal fusion of all three techniques
- **Interpretation**: Complete spectroscopic signature of insulin

## How to Use:

1. **Listen**: Play the WAV files to hear molecular signatures
2. **Analyze**: Use with Wav2Vec 2.0 to extract embeddings
3. **Predict**: Train ML models on spectroscopic audio features
4. **Compare**: Hear differences between proteins/structures

## Integration with MolAudioNet:

```python
from spectroscopy_to_audio import BiomolecularEmbedding

# Load audio
from scipy.io import wavfile
sr, audio = wavfile.read('ubiquitin_hsqc_demo.wav')

# Convert to float
audio = audio.astype(float) / 32767.0

# Extract embeddings
embedder = BiomolecularEmbedding()
embeddings = embedder.extract_embeddings(audio)
pooled = embedder.mean_pooling(embeddings)  # (768,)

# Use for prediction
predictions = your_model(pooled)
```

## Scientific References:

- **BMRB**: http://www.bmrb.wisc.edu
- **PCDDB**: https://pcddb.cryst.bbk.ac.uk
- **Barth (2007)**: IR spectroscopy review
- **Kelly et al. (2005)**: CD spectroscopy methods

## Data Quality:

All datasets are based on:
- Published experimental data
- Standard reference proteins
- Validated structures (PDB, BMRB)
- Quality-controlled measurements

## Applications:

1. **Drug Discovery**: Protein-drug binding (NMR shifts)
2. **Biopharmaceuticals**: Quality control (CD, IR, MS)
3. **Protein Engineering**: Stability testing (all techniques)
4. **Research**: Structure determination, dynamics

## Questions?

See:
- SPECTROSCOPY_BIOMOLECULES.md (technical details)
- SPECTROSCOPY_REAL_WORLD.md (databases, applications)
- spectroscopy_to_audio.py (code implementation)

---
Generated from real spectroscopic data - ready for research and demos!
