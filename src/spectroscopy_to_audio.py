"""
Spectroscopic Data to Audio Conversion for Biomolecules
Extends molecular sonification to include NMR, IR, MS, CD data

Based on: Zhou & Zhou (2026) - "Spectroscopy (NMR, IR, MS + Audio)" input modality
Author: Emily R. Zhou, Charles J. Zhou
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, find_peaks
import librosa
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

# ==================== SPECTROSCOPY TO AUDIO CONVERTERS ====================

class NMRToAudio:
    """
    Convert NMR spectra to audio
    Perfect for proteins, DNA, RNA
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def spectrum_to_audio(self, chemical_shifts: np.ndarray, 
                         intensities: np.ndarray,
                         duration: float = 5.0,
                         ppm_range: Tuple[float, float] = (0, 12)) -> np.ndarray:
        """
        Convert NMR spectrum (chemical shifts) to audio
        
        Parameters:
        -----------
        chemical_shifts : np.ndarray
            Chemical shifts in ppm
        intensities : np.ndarray
            Peak intensities (normalized)
        duration : float
            Audio duration in seconds
        ppm_range : tuple
            (min_ppm, max_ppm) for frequency mapping
        
        Returns:
        --------
        audio : np.ndarray
            Audio waveform
        """
        # Map chemical shifts to audible frequencies
        # 0-12 ppm → 200-2000 Hz (musical range)
        min_ppm, max_ppm = ppm_range
        norm_shifts = (chemical_shifts - min_ppm) / (max_ppm - min_ppm)
        norm_shifts = np.clip(norm_shifts, 0, 1)
        
        frequencies = 200 + norm_shifts * 1800
        
        # Generate audio
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Add each peak as a tone
        for freq, intensity in zip(frequencies, intensities):
            # Generate tone
            tone = intensity * np.sin(2 * np.pi * freq * t)
            
            # Add Lorentzian linewidth (realistic NMR peaks)
            linewidth = 5  # Hz
            envelope = 1 / (1 + ((t - duration/2) * linewidth)**2)
            
            audio += tone * envelope
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def fid_to_audio(self, fid: np.ndarray, 
                     original_rate: float = 50000) -> np.ndarray:
        """
        Convert NMR Free Induction Decay (FID) directly to audio
        FID is already a time-domain signal!
        
        Parameters:
        -----------
        fid : np.ndarray
            Complex FID data
        original_rate : float
            Original NMR sampling rate (Hz)
        
        Returns:
        --------
        audio : np.ndarray
            Audio waveform
        """
        # Take real part of complex FID
        fid_real = np.real(fid)
        
        # Resample to audio rate
        n_samples = int(len(fid_real) * self.sample_rate / original_rate)
        audio = resample(fid_real, n_samples)
        
        # Apply window to prevent clicks
        window = np.hanning(len(audio))
        audio = audio * window
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def hsqc_to_audio(self, h_shifts: np.ndarray, 
                     n_shifts: np.ndarray,
                     intensities: np.ndarray,
                     duration: float = 10.0) -> np.ndarray:
        """
        Convert 2D HSQC (protein backbone) to audio
        Each cross-peak becomes a chord
        
        Parameters:
        -----------
        h_shifts : np.ndarray
            1H chemical shifts
        n_shifts : np.ndarray
            15N chemical shifts  
        intensities : np.ndarray
            Cross-peak intensities
        duration : float
            Audio duration
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
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


class IRToAudio:
    """
    Convert IR/Raman spectra to audio
    Perfect for secondary structure analysis
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def spectrum_to_audio(self, wavenumbers: np.ndarray, 
                         absorbance: np.ndarray,
                         duration: float = 10.0,
                         wn_range: Tuple[float, float] = (400, 4000)) -> np.ndarray:
        """
        Convert IR spectrum to audio by sweeping through frequencies
        
        Parameters:
        -----------
        wavenumbers : np.ndarray
            Wavenumbers in cm⁻¹
        absorbance : np.ndarray
            Absorbance/transmittance values
        duration : float
            Audio duration
        wn_range : tuple
            (min_wn, max_wn) for frequency mapping
        """
        # Map wavenumbers to frequencies
        min_wn, max_wn = wn_range
        norm_wn = (wavenumbers - min_wn) / (max_wn - min_wn)
        norm_wn = np.clip(norm_wn, 0, 1)
        
        frequencies = 200 + norm_wn * 1800  # 200-2000 Hz
        
        # Generate audio (sweep through spectrum)
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Divide into time segments
        n_points = len(wavenumbers)
        samples_per_point = len(t) // n_points
        
        for i, (freq, intensity) in enumerate(zip(frequencies, absorbance)):
            start = i * samples_per_point
            end = start + samples_per_point
            if end > len(t):
                end = len(t)
            
            # Generate segment
            segment = intensity * np.sin(2 * np.pi * freq * t[start:end])
            audio[start:end] = segment
        
        return audio
    
    def amide_bands_to_audio(self, amide_i: Tuple[float, float],
                            amide_ii: Tuple[float, float],
                            duration: float = 5.0) -> np.ndarray:
        """
        Encode protein secondary structure via amide bands
        
        Parameters:
        -----------
        amide_i : tuple
            (center_wavenumber, intensity) for amide I
        amide_ii : tuple
            (center_wavenumber, intensity) for amide II
        duration : float
            Audio duration
        
        Returns:
        --------
        audio : np.ndarray
            Audio signature of secondary structure
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Amide I (1600-1700 cm⁻¹) → lower frequency
        wn_i, int_i = amide_i
        freq_i = 200 + ((wn_i - 1600) / 100) * 400  # 200-600 Hz
        
        # Amide II (1500-1600 cm⁻¹) → higher frequency  
        wn_ii, int_ii = amide_ii
        freq_ii = 600 + ((wn_ii - 1500) / 100) * 400  # 600-1000 Hz
        
        # Generate two-component signal
        audio = (
            int_i * np.sin(2 * np.pi * freq_i * t) +
            int_ii * np.sin(2 * np.pi * freq_ii * t)
        )
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def detect_secondary_structure(self, wavenumbers: np.ndarray,
                                   absorbance: np.ndarray) -> dict:
        """
        Analyze amide I band to determine secondary structure
        
        Returns:
        --------
        structure : dict
            {'helix_frac': float, 'sheet_frac': float, 'coil_frac': float}
        """
        # Focus on amide I region (1600-1700 cm⁻¹)
        mask = (wavenumbers >= 1600) & (wavenumbers <= 1700)
        amide_i_wn = wavenumbers[mask]
        amide_i_abs = absorbance[mask]
        
        # Find peaks
        peaks, properties = find_peaks(amide_i_abs, height=0.1)
        
        structure = {'helix_frac': 0.0, 'sheet_frac': 0.0, 'coil_frac': 0.0}
        
        for peak in peaks:
            peak_wn = amide_i_wn[peak]
            
            if 1650 <= peak_wn <= 1658:
                structure['helix_frac'] += 1
            elif 1625 <= peak_wn <= 1640:
                structure['sheet_frac'] += 1
            elif 1640 <= peak_wn <= 1648:
                structure['coil_frac'] += 1
        
        # Normalize
        total = sum(structure.values())
        if total > 0:
            for key in structure:
                structure[key] /= total
        
        return structure


class MSToAudio:
    """
    Convert mass spectrometry data to audio
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def spectrum_to_audio(self, mz_values: np.ndarray,
                         intensities: np.ndarray,
                         duration_per_peak: float = 0.1) -> np.ndarray:
        """
        Convert mass spectrum to audio
        Each peak becomes a brief tone
        
        Parameters:
        -----------
        mz_values : np.ndarray
            m/z ratios
        intensities : np.ndarray
            Peak intensities
        duration_per_peak : float
            Duration for each peak (seconds)
        """
        # Log scale for m/z (typical range: 100-10000)
        log_mz = np.log10(np.clip(mz_values, 100, 10000))
        
        # Map to frequencies
        min_log = np.log10(100)
        max_log = np.log10(10000)
        norm_log = (log_mz - min_log) / (max_log - min_log)
        frequencies = 200 + norm_log * 1800
        
        # Generate audio
        total_duration = len(mz_values) * duration_per_peak
        t = np.linspace(0, total_duration, int(self.sample_rate * total_duration))
        audio = np.zeros_like(t)
        
        # Each peak as impulse
        for i, (freq, intensity) in enumerate(zip(frequencies, intensities)):
            peak_samples = int(self.sample_rate * duration_per_peak)
            start = i * peak_samples
            end = start + peak_samples
            
            if end > len(audio):
                break
            
            # Generate peak
            peak_t = np.linspace(0, duration_per_peak, peak_samples)
            peak = intensity * np.sin(2 * np.pi * freq * peak_t)
            
            # Fast decay envelope (MS peaks are sharp!)
            envelope = np.exp(-peak_t / 0.02)
            audio[start:end] += peak * envelope
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio


class CDToAudio:
    """
    Convert Circular Dichroism spectra to audio
    Perfect for secondary structure quantification
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def spectrum_to_audio(self, wavelengths: np.ndarray,
                         ellipticity: np.ndarray,
                         duration: float = 8.0) -> np.ndarray:
        """
        Convert CD spectrum to audio
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Wavelengths in nm (190-250 for far-UV)
        ellipticity : np.ndarray
            Molar ellipticity values
        duration : float
            Audio duration
        """
        # Map wavelength to frequency (inverse - higher wavelength → lower freq)
        norm_wl = (250 - wavelengths) / 60  # 190-250 nm range
        norm_wl = np.clip(norm_wl, 0, 1)
        frequencies = 200 + norm_wl * 1800
        
        # Use absolute value of ellipticity for amplitude
        amplitudes = np.abs(ellipticity)
        if np.max(amplitudes) > 0:
            amplitudes = amplitudes / np.max(amplitudes)
        
        # Generate audio
        t = np.linspace(0, duration, int(self.sample_rate * duration))
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
    
    def secondary_structure_signature(self, helix_frac: float,
                                     sheet_frac: float,
                                     coil_frac: float,
                                     duration: float = 3.0) -> np.ndarray:
        """
        Generate audio signature for secondary structure composition
        
        Parameters:
        -----------
        helix_frac : float
            Fraction α-helix (0-1)
        sheet_frac : float
            Fraction β-sheet (0-1)  
        coil_frac : float
            Fraction random coil (0-1)
        duration : float
            Audio duration
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Different structures → different frequency signatures
        # α-helix: Low frequency (stable, ordered)
        helix_audio = helix_frac * np.sin(2 * np.pi * 300 * t)
        
        # β-sheet: Mid frequency  
        sheet_audio = sheet_frac * np.sin(2 * np.pi * 600 * t)
        
        # Random coil: High frequency (disordered)
        coil_audio = coil_frac * np.sin(2 * np.pi * 1200 * t)
        
        # Combine
        audio = helix_audio + sheet_audio + coil_audio
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio


# ==================== INTEGRATION WITH BIOMOLECULAR SONIFICATION ====================

class MultiModalSpectroscopy:
    """
    Combine multiple spectroscopic techniques
    Integrates with ProteinSonification, DNASonification
    """
    
    def __init__(self, sequence: str, molecule_type: str = 'protein'):
        """
        Parameters:
        -----------
        sequence : str
            Protein or nucleic acid sequence
        molecule_type : str
            'protein', 'dna', or 'rna'
        """
        self.sequence = sequence
        self.molecule_type = molecule_type
        
        # Initialize converters
        self.nmr_converter = NMRToAudio()
        self.ir_converter = IRToAudio()
        self.ms_converter = MSToAudio()
        self.cd_converter = CDToAudio()
        
        # Storage for spectroscopic data
        self.nmr_data = None
        self.ir_data = None
        self.ms_data = None
        self.cd_data = None
    
    def add_nmr_spectrum(self, chemical_shifts, intensities):
        """Add NMR spectrum data"""
        self.nmr_data = (chemical_shifts, intensities)
    
    def add_ir_spectrum(self, wavenumbers, absorbance):
        """Add IR spectrum data"""
        self.ir_data = (wavenumbers, absorbance)
    
    def add_ms_spectrum(self, mz_values, intensities):
        """Add mass spectrum data"""
        self.ms_data = (mz_values, intensities)
    
    def add_cd_spectrum(self, wavelengths, ellipticity):
        """Add CD spectrum data"""
        self.cd_data = (wavelengths, ellipticity)
    
    def encode_all_spectra(self) -> dict:
        """
        Generate audio from all available spectroscopic data
        
        Returns:
        --------
        audio_dict : dict
            {'nmr': array, 'ir': array, 'ms': array, 'cd': array}
        """
        audio_dict = {}
        
        if self.nmr_data:
            audio_dict['nmr'] = self.nmr_converter.spectrum_to_audio(*self.nmr_data)
        
        if self.ir_data:
            audio_dict['ir'] = self.ir_converter.spectrum_to_audio(*self.ir_data)
        
        if self.ms_data:
            audio_dict['ms'] = self.ms_converter.spectrum_to_audio(*self.ms_data)
        
        if self.cd_data:
            audio_dict['cd'] = self.cd_converter.spectrum_to_audio(*self.cd_data)
        
        return audio_dict
    
    def fuse_all_modalities(self) -> np.ndarray:
        """
        Fuse all spectroscopic modalities into single audio
        Weighted combination
        """
        audio_dict = self.encode_all_spectra()
        
        if not audio_dict:
            return np.array([])
        
        # Find maximum length
        max_len = max(len(audio) for audio in audio_dict.values())
        
        # Pad all to same length and combine
        fused = np.zeros(max_len)
        n_modalities = len(audio_dict)
        
        for name, audio in audio_dict.items():
            # Pad if needed
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            
            # Add with weight
            fused += audio / n_modalities
        
        # Normalize
        if np.max(np.abs(fused)) > 0:
            fused = fused / np.max(np.abs(fused))
        
        return fused
    
    def save_audio(self, filename: str, audio: Optional[np.ndarray] = None):
        """Save audio to WAV file"""
        if audio is None:
            audio = self.fuse_all_modalities()
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filename, self.nmr_converter.sample_rate, audio_int16)
        print(f"Saved spectroscopic audio to {filename}")


# ==================== EXAMPLE USAGE ====================

def example_ubiquitin_nmr():
    """Example: Ubiquitin NMR spectrum to audio"""
    
    # Simulated 1H-15N HSQC peak list for ubiquitin (76 residues)
    # Real data would come from BMRB
    
    np.random.seed(42)
    n_residues = 76
    
    # 1H chemical shifts (6-10 ppm range for backbone NH)
    h_shifts = np.random.uniform(6, 10, n_residues)
    
    # 15N chemical shifts (105-135 ppm range)
    n_shifts = np.random.uniform(105, 135, n_residues)
    
    # Intensities (all similar for well-folded protein)
    intensities = np.random.uniform(0.7, 1.0, n_residues)
    
    # Convert to audio
    nmr = NMRToAudio()
    audio = nmr.hsqc_to_audio(h_shifts, n_shifts, intensities, duration=10.0)
    
    # Save
    wavfile.write("ubiquitin_hsqc.wav", 16000, (audio * 32767).astype(np.int16))
    print("Saved ubiquitin HSQC audio")
    
    return audio

def example_protein_ir():
    """Example: Protein IR spectrum showing α-helix"""
    
    # Simulated IR spectrum with amide I band at 1656 cm⁻¹ (α-helix)
    wavenumbers = np.linspace(1200, 1800, 600)
    
    # Amide I peak (α-helix at 1656)
    amide_i = np.exp(-((wavenumbers - 1656)**2) / (2 * 10**2))
    
    # Amide II peak (at 1545)
    amide_ii = 0.6 * np.exp(-((wavenumbers - 1545)**2) / (2 * 15**2))
    
    # Combine
    absorbance = amide_i + amide_ii
    
    # Convert to audio
    ir = IRToAudio()
    audio = ir.spectrum_to_audio(wavenumbers, absorbance, duration=8.0)
    
    # Save
    wavfile.write("protein_ir_helix.wav", 16000, (audio * 32767).astype(np.int16))
    print("Saved protein IR audio (α-helix signature)")
    
    # Detect secondary structure
    structure = ir.detect_secondary_structure(wavenumbers, absorbance)
    print(f"Detected structure: {structure}")
    
    return audio

def example_multimodal_insulin():
    """Example: Insulin with NMR + IR + MS data"""
    
    # Insulin A-chain sequence
    sequence = "GIVEQCCTSICSLYQLENYCN"
    
    # Create multimodal object
    insulin = MultiModalSpectroscopy(sequence, molecule_type='protein')
    
    # Add simulated NMR data
    nmr_shifts = np.array([8.2, 7.8, 7.5, 8.0, 7.2])
    nmr_intensities = np.array([1.0, 0.9, 0.8, 0.95, 0.7])
    insulin.add_nmr_spectrum(nmr_shifts, nmr_intensities)
    
    # Add simulated IR data
    wn = np.linspace(1200, 1800, 300)
    absorbance = np.exp(-((wn - 1656)**2) / 200)
    insulin.add_ir_spectrum(wn, absorbance)
    
    # Add simulated MS data (tryptic peptides)
    mz = np.array([500, 750, 1200, 1500, 2000])
    ms_int = np.array([1.0, 0.8, 0.6, 0.9, 0.5])
    insulin.add_ms_spectrum(mz, ms_int)
    
    # Encode all modalities
    audio_dict = insulin.encode_all_spectra()
    print(f"Generated audio for {len(audio_dict)} modalities")
    
    # Fuse and save
    fused = insulin.fuse_all_modalities()
    insulin.save_audio("insulin_multimodal.wav", fused)
    
    return fused, audio_dict

if __name__ == "__main__":
    print("="*60)
    print("Spectroscopic Data to Audio Examples")
    print("="*60)
    
    # NMR example
    print("\n1. UBIQUITIN NMR (HSQC):")
    nmr_audio = example_ubiquitin_nmr()
    
    # IR example
    print("\n2. PROTEIN IR (Secondary Structure):")
    ir_audio = example_protein_ir()
    
    # Multimodal example
    print("\n3. INSULIN MULTIMODAL (NMR + IR + MS):")
    fused, modalities = example_multimodal_insulin()
    
    print("\n" + "="*60)
    print("Complete! Spectroscopic audio files generated.")
    print("="*60)
