"""
Standalone Spectroscopy to Audio Demo - Auto-Finding Version
Automatically searches for CSV files in common locations!

Requirements: numpy, pandas, scipy
Install: pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
from scipy.io import wavfile
import os
import glob

# ==================== FILE FINDER ====================

def find_csv_file(filename, search_dirs=None):
    """
    Automatically find CSV file in common locations
    
    Searches in:
    1. Current directory
    2. Parent directory
    3. spectroscopy_data/ subdirectories
    4. ../spectroscopy_data/ subdirectories
    5. Any subdirectory up to 3 levels deep
    """
    if search_dirs is None:
        search_dirs = [
            '.',                                    # Current dir
            '..',                                   # Parent dir
            'spectroscopy_data',                    # Data dir
            '../spectroscopy_data',                 # Parent data dir
            'spec/demo/spectroscopy_data',          # Old structure
            '../spec/demo/spectroscopy_data',       # Old structure parent
        ]
    
    # Also search subdirectories
    for root, dirs, files in os.walk('.', followlinks=False):
        if root.count(os.sep) <= 3:  # Max 3 levels deep
            search_dirs.append(root)
    
    # Try to find file
    for search_dir in search_dirs:
        # Direct path
        filepath = os.path.join(search_dir, filename)
        if os.path.exists(filepath):
            return filepath
        
        # In subdirectories (nmr, cd, ir, ms)
        for subdir in ['nmr', 'cd', 'ir', 'ms', '']:
            filepath = os.path.join(search_dir, subdir, filename)
            if os.path.exists(filepath):
                return filepath
    
    return None

# ==================== AUDIO GENERATION FUNCTIONS ====================

def generate_tone(frequency, duration, sample_rate=16000, amplitude=0.5):
    """Generate pure sine tone"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_harmony(frequencies, duration, sample_rate=16000, amplitudes=None):
    """Generate harmonic series"""
    if amplitudes is None:
        amplitudes = [1.0 / (i + 1) for i in range(len(frequencies))]
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    for freq, amp in zip(frequencies, amplitudes):
        audio += amp * np.sin(2 * np.pi * freq * t)
    
    return audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

# ==================== NMR TO AUDIO ====================

def nmr_hsqc_to_audio(h_shifts, n_shifts, intensities, duration=10.0, sample_rate=16000):
    """Convert 2D HSQC to audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    # Map shifts to frequencies
    h_freqs = 200 + (h_shifts / 12.0) * 800  # 1H: 200-1000 Hz
    n_freqs = 1000 + (n_shifts / 140.0) * 1000  # 15N: 1000-2000 Hz
    
    # Each cross-peak is a chord
    for h_freq, n_freq, intensity in zip(h_freqs, n_freqs, intensities):
        chord = intensity * (
            0.6 * np.sin(2 * np.pi * h_freq * t) +
            0.4 * np.sin(2 * np.pi * n_freq * t)
        )
        audio += chord
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

def nmr_spectrum_to_audio(chemical_shifts, intensities, duration=5.0, sample_rate=16000):
    """Convert 1D NMR spectrum to audio"""
    norm_shifts = chemical_shifts / 12.0
    frequencies = 200 + norm_shifts * 1800
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    for freq, intensity in zip(frequencies, intensities):
        audio += intensity * np.sin(2 * np.pi * freq * t)
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

# ==================== CD TO AUDIO ====================

def cd_spectrum_to_audio(wavelengths, ellipticity, duration=8.0, sample_rate=16000):
    """Convert CD spectrum to audio"""
    norm_wl = (250 - wavelengths) / 60
    norm_wl = np.clip(norm_wl, 0, 1)
    frequencies = 200 + norm_wl * 1800
    
    amplitudes = np.abs(ellipticity)
    if np.max(amplitudes) > 0:
        amplitudes = amplitudes / np.max(amplitudes)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    n_points = len(wavelengths)
    samples_per_point = len(t) // n_points
    
    for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
        start = i * samples_per_point
        end = start + samples_per_point
        if end > len(t):
            end = len(t)
        
        segment = amp * np.sin(2 * np.pi * freq * t[start:end])
        audio[start:end] = segment
    
    return audio

def cd_structure_signature(helix_frac, sheet_frac, coil_frac, duration=3.0, sample_rate=16000):
    """Generate audio for secondary structure"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    helix_audio = helix_frac * np.sin(2 * np.pi * 300 * t)
    sheet_audio = sheet_frac * np.sin(2 * np.pi * 600 * t)
    coil_audio = coil_frac * np.sin(2 * np.pi * 1200 * t)
    
    audio = helix_audio + sheet_audio + coil_audio
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

# ==================== IR TO AUDIO ====================

def ir_spectrum_to_audio(wavenumbers, absorbance, duration=10.0, sample_rate=16000):
    """Convert IR spectrum to audio"""
    norm_wn = (wavenumbers - wavenumbers.min()) / (wavenumbers.max() - wavenumbers.min())
    frequencies = 200 + norm_wn * 1800
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    n_points = len(wavenumbers)
    samples_per_point = len(t) // n_points
    
    for i, (freq, intensity) in enumerate(zip(frequencies, absorbance)):
        start = i * samples_per_point
        end = start + samples_per_point
        if end > len(t):
            end = len(t)
        
        segment = intensity * np.sin(2 * np.pi * freq * t[start:end])
        audio[start:end] = segment
    
    return audio

# ==================== MS TO AUDIO ====================

def ms_spectrum_to_audio(mz_values, intensities, duration_per_peak=0.1, sample_rate=16000):
    """Convert mass spectrum to audio"""
    log_mz = np.log10(np.clip(mz_values, 100, 10000))
    
    min_log = np.log10(100)
    max_log = np.log10(10000)
    norm_log = (log_mz - min_log) / (max_log - min_log)
    frequencies = 200 + norm_log * 1800
    
    total_duration = len(mz_values) * duration_per_peak
    t = np.linspace(0, total_duration, int(sample_rate * total_duration))
    audio = np.zeros_like(t)
    
    for i, (freq, intensity) in enumerate(zip(frequencies, intensities)):
        peak_samples = int(sample_rate * duration_per_peak)
        start = i * peak_samples
        end = start + peak_samples
        
        if end > len(audio):
            break
        
        peak_t = np.linspace(0, duration_per_peak, peak_samples)
        peak = intensity * np.sin(2 * np.pi * freq * peak_t)
        
        envelope = np.exp(-peak_t / 0.02)
        audio[start:end] += peak * envelope
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

# ==================== DEMO FUNCTIONS ====================

def demo_ubiquitin_nmr():
    """Load and convert ubiquitin NMR data"""
    print("="*60)
    print("1. UBIQUITIN NMR (1H-15N HSQC)")
    print("="*60)
    
    # Try to find file
    filepath = find_csv_file('ubiquitin_hsqc.csv')
    
    if filepath is None:
        print("✗ File not found: ubiquitin_hsqc.csv")
        print("  Searched common locations")
        print("  Skipping...")
        print()
        return None
    
    print(f"✓ Found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, comment='#')
        df = df[df['H_ppm'] != '---']
        df = df.reset_index(drop=True)
        
        h_shifts = df['H_ppm'].astype(float).values
        n_shifts = df['N_ppm'].astype(float).values
        intensities = df['Intensity'].astype(float).values
        
        print(f"✓ Loaded {len(h_shifts)} residues")
        print(f"  1H range: {h_shifts.min():.2f} - {h_shifts.max():.2f} ppm")
        print(f"  15N range: {n_shifts.min():.2f} - {n_shifts.max():.2f} ppm")
        
        audio = nmr_hsqc_to_audio(h_shifts, n_shifts, intensities, duration=10.0)
        
        wavfile.write('ubiquitin_hsqc_demo.wav', 16000, (audio * 32767).astype(np.int16))
        print(f"✓ Saved: ubiquitin_hsqc_demo.wav (10 sec)")
        print()
        
        return audio
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        print("  Skipping...")
        print()
        return None

def demo_lysozyme_cd():
    """Load and convert lysozyme CD data"""
    print("="*60)
    print("2. LYSOZYME CD SPECTRUM")
    print("="*60)
    
    filepath = find_csv_file('lysozyme_cd.csv')
    
    if filepath is None:
        print("✗ File not found: lysozyme_cd.csv")
        print("  Skipping...")
        print()
        return None, None
    
    print(f"✓ Found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, comment='#')
        
        wavelengths = df['Wavelength_nm'].values
        ellipticity = df['MRE'].values
        
        print(f"✓ Loaded CD spectrum")
        print(f"  Wavelength: {wavelengths.min():.0f} - {wavelengths.max():.0f} nm")
        print(f"  → α-helix signature (36% helix)")
        
        audio_spectrum = cd_spectrum_to_audio(wavelengths, ellipticity, duration=8.0)
        wavfile.write('lysozyme_cd_spectrum.wav', 16000, (audio_spectrum * 32767).astype(np.int16))
        print(f"✓ Saved: lysozyme_cd_spectrum.wav (8 sec)")
        
        audio_struct = cd_structure_signature(0.36, 0.12, 0.52, duration=5.0)
        wavfile.write('lysozyme_cd_structure.wav', 16000, (audio_struct * 32767).astype(np.int16))
        print(f"✓ Saved: lysozyme_cd_structure.wav (5 sec)")
        print()
        
        return audio_spectrum, audio_struct
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        print("  Skipping...")
        print()
        return None, None

def demo_bsa_ir():
    """Load and convert BSA FTIR data"""
    print("="*60)
    print("3. BSA FTIR SPECTRUM")
    print("="*60)
    
    filepath = find_csv_file('bsa_ftir.csv')
    
    if filepath is None:
        print("✗ File not found: bsa_ftir.csv")
        print("  Skipping...")
        print()
        return None
    
    print(f"✓ Found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, comment='#')
        
        wavenumbers = df['Wavenumber'].values
        absorbance = df['Absorbance'].values
        
        print(f"✓ Loaded IR spectrum")
        print(f"  Wavenumber: {wavenumbers.min():.0f} - {wavenumbers.max():.0f} cm⁻¹")
        print(f"  Amide I peak → α-helix signature")
        
        audio = ir_spectrum_to_audio(wavenumbers, absorbance, duration=12.0)
        
        wavfile.write('bsa_ftir_demo.wav', 16000, (audio * 32767).astype(np.int16))
        print(f"✓ Saved: bsa_ftir_demo.wav (12 sec)")
        print()
        
        return audio
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        print("  Skipping...")
        print()
        return None

def demo_dna_nmr():
    """Load and convert DNA NMR data"""
    print("="*60)
    print("4. DNA DODECAMER NMR")
    print("="*60)
    
    filepath = find_csv_file('dna_dodecamer_imino.csv')
    
    if filepath is None:
        print("✗ File not found: dna_dodecamer_imino.csv")
        print("  Skipping...")
        print()
        return None
    
    print(f"✓ Found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, comment='#')
        
        shifts = df['Chemical_Shift'].values
        intensities = df['Intensity'].values
        
        print(f"✓ Loaded DNA NMR spectrum")
        print(f"  {len(shifts)} imino proton peaks")
        print(f"  Chemical shifts: {shifts.min():.2f} - {shifts.max():.2f} ppm")
        print(f"  → Classic B-DNA structure")
        
        audio = nmr_spectrum_to_audio(shifts, intensities, duration=6.0)
        
        wavfile.write('dna_dodecamer_nmr.wav', 16000, (audio * 32767).astype(np.int16))
        print(f"✓ Saved: dna_dodecamer_nmr.wav (6 sec)")
        print()
        
        return audio
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        print("  Skipping...")
        print()
        return None

def demo_insulin_ms():
    """Load and convert insulin MS data"""
    print("="*60)
    print("5. INSULIN MASS SPECTRUM")
    print("="*60)
    
    filepath = find_csv_file('insulin_esi_ms.csv')
    
    if filepath is None:
        print("✗ File not found: insulin_esi_ms.csv")
        print("  Skipping...")
        print()
        return None
    
    print(f"✓ Found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, comment='#')
        
        charge_states = df.iloc[:7]
        mz_values = charge_states['m/z'].values
        intensities = charge_states['Intensity'].values
        
        print(f"✓ Loaded mass spectrum")
        print(f"  {len(mz_values)} charge states")
        print(f"  m/z range: {mz_values.min():.1f} - {mz_values.max():.1f}")
        print(f"  Human insulin: 5806.5 Da")
        
        audio = ms_spectrum_to_audio(mz_values, intensities, duration_per_peak=0.2)
        
        wavfile.write('insulin_ms_demo.wav', 16000, (audio * 32767).astype(np.int16))
        print(f"✓ Saved: insulin_ms_demo.wav ({len(audio)/16000:.1f} sec)")
        print()
        
        return audio
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        print("  Skipping...")
        print()
        return None

def create_simple_multimodal():
    """Create simple multimodal demo"""
    print("="*60)
    print("6. MULTIMODAL DEMO (Synthetic)")
    print("="*60)
    
    nmr_shifts = np.array([8.2, 7.8, 7.5, 8.0, 7.2])
    nmr_int = np.array([1.0, 0.9, 0.8, 0.95, 0.7])
    nmr_audio = nmr_spectrum_to_audio(nmr_shifts, nmr_int, duration=3.0)
    
    ir_wn = np.linspace(1200, 1800, 100)
    ir_abs = np.exp(-((ir_wn - 1656)**2) / 200)
    ir_audio = ir_spectrum_to_audio(ir_wn, ir_abs, duration=3.0)
    
    ms_mz = np.array([1000, 2000, 3000])
    ms_int = np.array([0.8, 1.0, 0.6])
    ms_audio = ms_spectrum_to_audio(ms_mz, ms_int, duration_per_peak=0.3)
    
    max_len = max(len(nmr_audio), len(ir_audio), len(ms_audio))
    nmr_padded = np.pad(nmr_audio, (0, max_len - len(nmr_audio)))
    ir_padded = np.pad(ir_audio, (0, max_len - len(ir_audio)))
    ms_padded = np.pad(ms_audio, (0, max_len - len(ms_audio)))
    
    fused = (nmr_padded + ir_padded + ms_padded) / 3.0
    
    if np.max(np.abs(fused)) > 0:
        fused = fused / np.max(np.abs(fused))
    
    wavfile.write('multimodal_demo.wav', 16000, (fused * 32767).astype(np.int16))
    print(f"✓ Created synthetic multimodal demo")
    print(f"✓ Saved: multimodal_demo.wav ({len(fused)/16000:.1f} sec)")
    print(f"  Combines: NMR + IR + MS")
    print()
    
    return fused

# ==================== MAIN ====================

def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("SPECTROSCOPIC DATA → AUDIO CONVERSION")
    print("Auto-Finding Demo (Searches for CSV files)")
    print("="*60 + "\n")
    
    # Run demos
    ub_audio = demo_ubiquitin_nmr()
    lyz_spectrum, lyz_struct = demo_lysozyme_cd()
    bsa_audio = demo_bsa_ir()
    dna_audio = demo_dna_nmr()
    ins_audio = demo_insulin_ms()
    multi_audio = create_simple_multimodal()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nAudio files generated:")
    
    files_created = []
    for fname in ['ubiquitin_hsqc_demo.wav', 'lysozyme_cd_spectrum.wav', 
                  'lysozyme_cd_structure.wav', 'bsa_ftir_demo.wav',
                  'dna_dodecamer_nmr.wav', 'insulin_ms_demo.wav',
                  'multimodal_demo.wav']:
        if os.path.exists(fname):
            size = os.path.getsize(fname) / 1024
            files_created.append(f"  ✓ {fname} ({size:.0f} KB)")
    
    for f in files_created:
        print(f)
    
    if len(files_created) == 0:
        print("  ⚠ No files created (CSV files not found)")
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("\nCSV files not found. Please ensure they are in:")
        print("  1. Current directory")
        print("  2. spectroscopy_data/ subdirectories")
        print("  3. ../spectroscopy_data/ subdirectories")
        print("\nExpected files:")
        print("  - ubiquitin_hsqc.csv")
        print("  - lysozyme_cd.csv")
        print("  - bsa_ftir.csv")
        print("  - dna_dodecamer_imino.csv")
        print("  - insulin_esi_ms.csv")
    else:
        print("\nNext steps:")
        print("  1. Play the WAV files in any audio player")
        print("  2. Extract Wav2Vec embeddings (see README.md)")
        print("  3. Use for protein function prediction!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
