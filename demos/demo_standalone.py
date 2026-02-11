"""
Standalone Spectroscopy to Audio Demo
All code included - no external dependencies needed!

Just run: python demo_standalone.py

Requirements: numpy, pandas, scipy
Install: pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
from scipy.io import wavfile
import os

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
    """
    Convert 2D HSQC (protein backbone) to audio
    Each cross-peak becomes a chord
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    # Map shifts to frequencies
    h_freqs = 200 + (h_shifts / 12.0) * 800  # 1H: 200-1000 Hz
    n_freqs = 1000 + (n_shifts / 140.0) * 1000  # 15N: 1000-2000 Hz
    
    # Each cross-peak is a chord (H + N frequencies)
    for h_freq, n_freq, intensity in zip(h_freqs, n_freqs, intensities):
        # Generate chord
        chord = intensity * (
            0.6 * np.sin(2 * np.pi * h_freq * t) +
            0.4 * np.sin(2 * np.pi * n_freq * t)
        )
        audio += chord
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

def nmr_spectrum_to_audio(chemical_shifts, intensities, duration=5.0, sample_rate=16000):
    """Convert 1D NMR spectrum to audio"""
    # Map chemical shifts to frequencies (0-12 ppm → 200-2000 Hz)
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
    # Map wavelength to frequency (inverse - higher wavelength → lower freq)
    norm_wl = (250 - wavelengths) / 60  # 190-250 nm range
    norm_wl = np.clip(norm_wl, 0, 1)
    frequencies = 200 + norm_wl * 1800
    
    # Use absolute value of ellipticity for amplitude
    amplitudes = np.abs(ellipticity)
    if np.max(amplitudes) > 0:
        amplitudes = amplitudes / np.max(amplitudes)
    
    # Generate audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    # Sweep through spectrum
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
    """Generate audio signature for secondary structure composition"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Different structures → different frequency signatures
    helix_audio = helix_frac * np.sin(2 * np.pi * 300 * t)  # α-helix: Low
    sheet_audio = sheet_frac * np.sin(2 * np.pi * 600 * t)  # β-sheet: Mid
    coil_audio = coil_frac * np.sin(2 * np.pi * 1200 * t)   # Coil: High
    
    audio = helix_audio + sheet_audio + coil_audio
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

# ==================== IR TO AUDIO ====================

def ir_spectrum_to_audio(wavenumbers, absorbance, duration=10.0, sample_rate=16000):
    """Convert IR spectrum to audio by sweeping through frequencies"""
    # Map wavenumbers to frequencies
    norm_wn = (wavenumbers - wavenumbers.min()) / (wavenumbers.max() - wavenumbers.min())
    frequencies = 200 + norm_wn * 1800
    
    # Generate audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    # Divide into time segments
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
    # Log scale for m/z
    log_mz = np.log10(np.clip(mz_values, 100, 10000))
    
    # Map to frequencies
    min_log = np.log10(100)
    max_log = np.log10(10000)
    norm_log = (log_mz - min_log) / (max_log - min_log)
    frequencies = 200 + norm_log * 1800
    
    # Generate audio
    total_duration = len(mz_values) * duration_per_peak
    t = np.linspace(0, total_duration, int(sample_rate * total_duration))
    audio = np.zeros_like(t)
    
    # Each peak as impulse
    for i, (freq, intensity) in enumerate(zip(frequencies, intensities)):
        peak_samples = int(sample_rate * duration_per_peak)
        start = i * peak_samples
        end = start + peak_samples
        
        if end > len(audio):
            break
        
        # Generate peak
        peak_t = np.linspace(0, duration_per_peak, peak_samples)
        peak = intensity * np.sin(2 * np.pi * freq * peak_t)
        
        # Fast decay envelope
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
    
    try:
        # Load data
        df = pd.read_csv('ubiquitin_hsqc.csv', comment='#')
        
        # Remove prolines
        df = df[df['H_ppm'] != '---']
        df = df.reset_index(drop=True)
        
        # Extract data
        h_shifts = df['H_ppm'].astype(float).values
        n_shifts = df['N_ppm'].astype(float).values
        intensities = df['Intensity'].astype(float).values
        
        print(f"✓ Loaded {len(h_shifts)} residues")
        print(f"  1H range: {h_shifts.min():.2f} - {h_shifts.max():.2f} ppm")
        print(f"  15N range: {n_shifts.min():.2f} - {n_shifts.max():.2f} ppm")
        
        # Convert to audio
        audio = nmr_hsqc_to_audio(h_shifts, n_shifts, intensities, duration=10.0)
        
        # Save
        wavfile.write('ubiquitin_hsqc_demo.wav', 16000, (audio * 32767).astype(np.int16))
        print(f"✓ Saved: ubiquitin_hsqc_demo.wav (10 sec)")
        print()
        
        return audio
        
    except FileNotFoundError:
        print("✗ File not found: ubiquitin_hsqc.csv")
        print("  Skipping...")
        print()
        return None

def demo_lysozyme_cd():
    """Load and convert lysozyme CD data"""
    print("="*60)
    print("2. LYSOZYME CD SPECTRUM")
    print("="*60)
    
    try:
        # Load data
        df = pd.read_csv('lysozyme_cd.csv', comment='#')
        
        wavelengths = df['Wavelength_nm'].values
        ellipticity = df['MRE'].values
        
        print(f"✓ Loaded CD spectrum")
        print(f"  Wavelength: {wavelengths.min():.0f} - {wavelengths.max():.0f} nm")
        print(f"  Minima at 208, 222 nm → α-helix signature")
        
        # Convert spectrum to audio
        audio_spectrum = cd_spectrum_to_audio(wavelengths, ellipticity, duration=8.0)
        wavfile.write('lysozyme_cd_spectrum.wav', 16000, (audio_spectrum * 32767).astype(np.int16))
        print(f"✓ Saved: lysozyme_cd_spectrum.wav (8 sec)")
        
        # Structure signature
        audio_struct = cd_structure_signature(0.36, 0.12, 0.52, duration=5.0)
        wavfile.write('lysozyme_cd_structure.wav', 16000, (audio_struct * 32767).astype(np.int16))
        print(f"✓ Saved: lysozyme_cd_structure.wav (5 sec)")
        print()
        
        return audio_spectrum, audio_struct
        
    except FileNotFoundError:
        print("✗ File not found: lysozyme_cd.csv")
        print("  Skipping...")
        print()
        return None, None

def demo_bsa_ir():
    """Load and convert BSA FTIR data"""
    print("="*60)
    print("3. BSA FTIR SPECTRUM")
    print("="*60)
    
    try:
        # Load data
        df = pd.read_csv('bsa_ftir.csv', comment='#')
        
        wavenumbers = df['Wavenumber'].values
        absorbance = df['Absorbance'].values
        
        print(f"✓ Loaded IR spectrum")
        print(f"  Wavenumber: {wavenumbers.min():.0f} - {wavenumbers.max():.0f} cm⁻¹")
        print(f"  Amide I peak at 1652 cm⁻¹ → α-helix")
        
        # Convert to audio
        audio = ir_spectrum_to_audio(wavenumbers, absorbance, duration=12.0)
        
        wavfile.write('bsa_ftir_demo.wav', 16000, (audio * 32767).astype(np.int16))
        print(f"✓ Saved: bsa_ftir_demo.wav (12 sec)")
        print()
        
        return audio
        
    except FileNotFoundError:
        print("✗ File not found: bsa_ftir.csv")
        print("  Skipping...")
        print()
        return None

def demo_dna_nmr():
    """Load and convert DNA NMR data"""
    print("="*60)
    print("4. DNA DODECAMER NMR")
    print("="*60)
    
    try:
        # Load data
        df = pd.read_csv('dna_dodecamer_imino.csv', comment='#')
        
        shifts = df['Chemical_Shift'].values
        intensities = df['Intensity'].values
        
        print(f"✓ Loaded DNA NMR spectrum")
        print(f"  {len(shifts)} imino proton peaks")
        print(f"  Chemical shifts: {shifts.min():.2f} - {shifts.max():.2f} ppm")
        print(f"  → d(CGCGAATTCGCG)2 - Classic B-DNA")
        
        # Convert to audio
        audio = nmr_spectrum_to_audio(shifts, intensities, duration=6.0)
        
        wavfile.write('dna_dodecamer_nmr.wav', 16000, (audio * 32767).astype(np.int16))
        print(f"✓ Saved: dna_dodecamer_nmr.wav (6 sec)")
        print()
        
        return audio
        
    except FileNotFoundError:
        print("✗ File not found: dna_dodecamer_imino.csv")
        print("  Skipping...")
        print()
        return None

def demo_insulin_ms():
    """Load and convert insulin MS data"""
    print("="*60)
    print("5. INSULIN MASS SPECTRUM")
    print("="*60)
    
    try:
        # Load data
        df = pd.read_csv('insulin_esi_ms.csv', comment='#')
        
        # Get charge states
        charge_states = df.iloc[:7]
        mz_values = charge_states['m/z'].values
        intensities = charge_states['Intensity'].values
        
        print(f"✓ Loaded mass spectrum")
        print(f"  {len(mz_values)} charge states")
        print(f"  m/z range: {mz_values.min():.1f} - {mz_values.max():.1f}")
        print(f"  Human insulin: 5806.5 Da")
        
        # Convert to audio
        audio = ms_spectrum_to_audio(mz_values, intensities, duration_per_peak=0.2)
        
        wavfile.write('insulin_ms_demo.wav', 16000, (audio * 32767).astype(np.int16))
        print(f"✓ Saved: insulin_ms_demo.wav ({len(audio)/16000:.1f} sec)")
        print()
        
        return audio
        
    except FileNotFoundError:
        print("✗ File not found: insulin_esi_ms.csv")
        print("  Skipping...")
        print()
        return None

def create_simple_multimodal():
    """Create simple multimodal demo with synthetic data"""
    print("="*60)
    print("6. MULTIMODAL DEMO (Synthetic)")
    print("="*60)
    
    # Create simple synthetic data for each modality
    
    # NMR (5 peaks)
    nmr_shifts = np.array([8.2, 7.8, 7.5, 8.0, 7.2])
    nmr_int = np.array([1.0, 0.9, 0.8, 0.95, 0.7])
    nmr_audio = nmr_spectrum_to_audio(nmr_shifts, nmr_int, duration=3.0)
    
    # IR (simple peak)
    ir_wn = np.linspace(1200, 1800, 100)
    ir_abs = np.exp(-((ir_wn - 1656)**2) / 200)
    ir_audio = ir_spectrum_to_audio(ir_wn, ir_abs, duration=3.0)
    
    # MS (3 charge states)
    ms_mz = np.array([1000, 2000, 3000])
    ms_int = np.array([0.8, 1.0, 0.6])
    ms_audio = ms_spectrum_to_audio(ms_mz, ms_int, duration_per_peak=0.3)
    
    # Make all same length
    max_len = max(len(nmr_audio), len(ir_audio), len(ms_audio))
    nmr_padded = np.pad(nmr_audio, (0, max_len - len(nmr_audio)))
    ir_padded = np.pad(ir_audio, (0, max_len - len(ir_audio)))
    ms_padded = np.pad(ms_audio, (0, max_len - len(ms_audio)))
    
    # Fuse
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
    print("Standalone Demo (No External Dependencies)")
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
            size = os.path.getsize(fname) / 1024  # KB
            files_created.append(f"  ✓ {fname} ({size:.0f} KB)")
    
    for f in files_created:
        print(f)
    
    print("\nNext steps:")
    print("  1. Play the WAV files in any audio player")
    print("  2. Extract Wav2Vec embeddings (see README.md)")
    print("  3. Use for protein function prediction!")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
