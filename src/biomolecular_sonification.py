"""
Biomolecular Sonification - Production Implementation
Extends MolAudioNet to DNA, RNA, and Proteins

Based on: Zhou & Zhou (2026) Molecular Sonification Framework
Patents: US 9,018,506 | US 10,381,108

Author: Emily R. Zhou, Charles J. Zhou
License: MIT (for academic use)
"""

import numpy as np
import librosa
from scipy.io import wavfile
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ==================== CONSTANTS ====================

# Amino acid to frequency mapping (Hz)
# Based on hydrophobicity, charge, and size
AMINO_ACID_FREQS = {
    # Hydrophobic (low frequencies, 200-400 Hz)
    'A': 220,   # Alanine
    'V': 247,   # Valine
    'I': 277,   # Isoleucine
    'L': 294,   # Leucine
    'M': 311,   # Methionine
    'F': 330,   # Phenylalanine
    'W': 349,   # Tryptophan
    'P': 370,   # Proline
    
    # Polar (mid frequencies, 400-600 Hz)
    'S': 392,   # Serine
    'T': 415,   # Threonine
    'C': 440,   # Cysteine (special - disulfide)
    'Y': 466,   # Tyrosine
    'N': 494,   # Asparagine
    'Q': 523,   # Glutamine
    
    # Charged positive (high frequencies, 600-800 Hz)
    'K': 554,   # Lysine
    'R': 587,   # Arginine
    'H': 622,   # Histidine
    
    # Charged negative (very high frequencies, 800-1000 Hz)
    'D': 659,   # Aspartic acid
    'E': 698,   # Glutamic acid
    
    # Special
    'G': 175,   # Glycine (smallest, lowest)
}

# DNA/RNA base to frequency mapping
NUCLEOTIDE_FREQS = {
    # Purines (larger, lower frequencies)
    'A': 440,   # Adenine
    'G': 523,   # Guanine
    
    # Pyrimidines (smaller, higher frequencies)
    'T': 587,   # Thymine (DNA)
    'U': 587,   # Uracil (RNA)
    'C': 659,   # Cytosine
}

# Secondary structure to harmonic signature
SECONDARY_STRUCTURE = {
    'H': [1.0, 0.5, 0.25],      # Alpha helix - consonant harmonics
    'E': [1.0, 0.6, 0.3],       # Beta sheet - slightly different
    'C': [1.0, 0.3, 0.1],       # Coil - minimal harmonics
    'T': [1.0, 0.4, 0.2, 0.1],  # Turn - more complex
}

# ==================== CORE SONIFICATION CLASSES ====================

class BioMolecularSonification:
    """
    Base class for biomolecular sonification
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def generate_tone(self, frequency: float, duration: float, 
                     amplitude: float = 0.5) -> np.ndarray:
        """Generate pure sine tone"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    def generate_harmony(self, frequencies: List[float], duration: float,
                        amplitudes: Optional[List[float]] = None) -> np.ndarray:
        """Generate harmonic series"""
        if amplitudes is None:
            amplitudes = [1.0 / (i + 1) for i in range(len(frequencies))]
            
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        for freq, amp in zip(frequencies, amplitudes):
            audio += amp * np.sin(2 * np.pi * freq * t)
            
        return audio / np.max(np.abs(audio))  # Normalize
    
    def apply_envelope(self, audio: np.ndarray, 
                      attack: float = 0.01, 
                      decay: float = 0.01,
                      sustain: float = 0.7,
                      release: float = 0.1) -> np.ndarray:
        """Apply ADSR envelope"""
        n = len(audio)
        envelope = np.ones(n)
        
        # Attack
        attack_samples = int(attack * n)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        decay_samples = int(decay * n)
        envelope[attack_samples:attack_samples + decay_samples] = \
            np.linspace(1, sustain, decay_samples)
        
        # Release
        release_samples = int(release * n)
        envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)
        
        return audio * envelope

# ==================== PROTEIN SONIFICATION ====================

class ProteinSonification(BioMolecularSonification):
    """
    Hierarchical protein sonification
    Implements multi-scale audio encoding
    """
    
    def __init__(self, sequence: str, structure_pdb: Optional[str] = None,
                 residues_per_second: int = 10, sample_rate: int = 16000):
        """
        Parameters:
        -----------
        sequence : str
            Amino acid sequence (single letter code)
        structure_pdb : str, optional
            Path to PDB file for 3D structure
        residues_per_second : int
            Encoding speed (default: 10 residues/sec)
        """
        super().__init__(sample_rate)
        self.sequence = sequence.upper()
        self.structure_pdb = structure_pdb
        self.residues_per_second = residues_per_second
        self.duration_per_residue = 1.0 / residues_per_second
        
    def encode_primary_structure(self) -> np.ndarray:
        """
        Encode amino acid sequence
        Each residue -> distinct frequency (100ms duration)
        """
        audio_chunks = []
        
        for residue in self.sequence:
            if residue in AMINO_ACID_FREQS:
                freq = AMINO_ACID_FREQS[residue]
                chunk = self.generate_tone(
                    frequency=freq,
                    duration=self.duration_per_residue
                )
                # Apply envelope for smooth transitions
                chunk = self.apply_envelope(chunk)
                audio_chunks.append(chunk)
            else:
                # Unknown residue - silence
                chunk = np.zeros(int(self.sample_rate * self.duration_per_residue))
                audio_chunks.append(chunk)
        
        return np.concatenate(audio_chunks)
    
    def encode_secondary_structure(self, ss_sequence: Optional[str] = None) -> np.ndarray:
        """
        Encode secondary structure as harmonic overlays
        
        Parameters:
        -----------
        ss_sequence : str, optional
            Secondary structure sequence (H=helix, E=sheet, C=coil)
            If None, predict using simple rules
        """
        if ss_sequence is None:
            ss_sequence = self._predict_secondary_structure()
        
        audio_chunks = []
        
        for i, ss_type in enumerate(ss_sequence):
            residue = self.sequence[i]
            base_freq = AMINO_ACID_FREQS.get(residue, 440)
            
            # Get harmonic signature for this SS type
            harmonics = SECONDARY_STRUCTURE.get(ss_type, [1.0])
            
            # Generate harmonic series
            freqs = [base_freq * (j + 1) for j in range(len(harmonics))]
            chunk = self.generate_harmony(
                frequencies=freqs,
                duration=self.duration_per_residue,
                amplitudes=harmonics
            )
            
            audio_chunks.append(chunk)
        
        return np.concatenate(audio_chunks)
    
    def encode_tertiary_structure(self) -> Optional[np.ndarray]:
        """
        Encode 3D fold using low-frequency modulation
        Requires PDB structure
        """
        if self.structure_pdb is None:
            return None
        
        # Parse PDB (simplified - would use BioPython in production)
        # For now, generate placeholder based on sequence properties
        
        # Calculate compactness metric (radius of gyration approximation)
        # Hydrophobic residues tend to cluster -> lower frequency
        hydrophobic_ratio = sum(
            1 for r in self.sequence 
            if r in ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
        ) / len(self.sequence)
        
        # Map to frequency range (20-200 Hz)
        base_freq = 20 + (hydrophobic_ratio * 180)
        
        # Generate low-frequency carrier
        total_duration = len(self.sequence) * self.duration_per_residue
        audio = self.generate_tone(
            frequency=base_freq,
            duration=total_duration,
            amplitude=0.3  # Lower amplitude for background
        )
        
        return audio
    
    def encode_functional_sites(self, sites: Dict[str, List[int]]) -> np.ndarray:
        """
        Highlight functional sites (active sites, binding sites)
        
        Parameters:
        -----------
        sites : dict
            {'site_name': [residue_indices]}
        """
        audio = self.encode_primary_structure()
        
        for site_name, indices in sites.items():
            for idx in indices:
                # Add harmonic emphasis at functional sites
                start_sample = int(idx * self.duration_per_residue * self.sample_rate)
                duration_samples = int(self.duration_per_residue * self.sample_rate)
                
                # Add high-frequency "ping"
                ping = self.generate_tone(
                    frequency=1760,  # A6 - high, noticeable
                    duration=self.duration_per_residue,
                    amplitude=0.3
                )
                
                # Add to main audio
                audio[start_sample:start_sample + duration_samples] += ping
        
        return audio
    
    def fuse_all_levels(self, ss_sequence: Optional[str] = None,
                       functional_sites: Optional[Dict] = None) -> np.ndarray:
        """
        Multi-scale fusion of all encoding levels
        """
        # Primary structure (main signal)
        primary = self.encode_primary_structure()
        
        # Secondary structure (harmonic enrichment)
        secondary = self.encode_secondary_structure(ss_sequence)
        
        # Tertiary structure (low-frequency background)
        tertiary = self.encode_tertiary_structure()
        
        # Start with primary
        fused = primary.copy()
        
        # Add secondary (with weight)
        if secondary is not None and len(secondary) == len(fused):
            fused = 0.6 * fused + 0.4 * secondary
        
        # Add tertiary background
        if tertiary is not None and len(tertiary) == len(fused):
            fused = 0.7 * fused + 0.3 * tertiary
        
        # Highlight functional sites
        if functional_sites is not None:
            site_audio = self.encode_functional_sites(functional_sites)
            if len(site_audio) == len(fused):
                fused = 0.8 * fused + 0.2 * site_audio
        
        # Normalize
        fused = fused / np.max(np.abs(fused))
        
        return fused
    
    def _predict_secondary_structure(self) -> str:
        """
        Simple secondary structure prediction
        (In production, use DSSP or machine learning)
        """
        # Very simple heuristic predictor
        ss = []
        for i, residue in enumerate(self.sequence):
            # Helix-favoring residues
            if residue in ['A', 'E', 'L', 'M']:
                ss.append('H')
            # Sheet-favoring residues
            elif residue in ['V', 'I', 'F', 'Y', 'W']:
                ss.append('E')
            # Others - coil
            else:
                ss.append('C')
        
        return ''.join(ss)
    
    def save_audio(self, filename: str, audio: Optional[np.ndarray] = None):
        """Save audio to WAV file"""
        if audio is None:
            audio = self.fuse_all_levels()
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_int16)
        print(f"Saved protein audio to {filename}")

# ==================== DNA/RNA SONIFICATION ====================

class NucleicAcidSonification(BioMolecularSonification):
    """
    DNA and RNA sonification
    """
    
    def __init__(self, sequence: str, molecule_type: str = 'DNA',
                 bases_per_second: int = 20, sample_rate: int = 16000):
        """
        Parameters:
        -----------
        sequence : str
            Nucleotide sequence (A, T/U, G, C)
        molecule_type : str
            'DNA' or 'RNA'
        bases_per_second : int
            Encoding speed
        """
        super().__init__(sample_rate)
        self.sequence = sequence.upper()
        self.molecule_type = molecule_type.upper()
        self.bases_per_second = bases_per_second
        self.duration_per_base = 1.0 / bases_per_second
        
    def encode_sequence(self) -> np.ndarray:
        """Encode nucleotide sequence"""
        audio_chunks = []
        
        for base in self.sequence:
            if base in NUCLEOTIDE_FREQS:
                freq = NUCLEOTIDE_FREQS[base]
                chunk = self.generate_tone(
                    frequency=freq,
                    duration=self.duration_per_base
                )
                chunk = self.apply_envelope(chunk)
                audio_chunks.append(chunk)
            else:
                # Unknown base - silence
                chunk = np.zeros(int(self.sample_rate * self.duration_per_base))
                audio_chunks.append(chunk)
        
        return np.concatenate(audio_chunks)
    
    def encode_base_pairing(self, structure: Optional[str] = None) -> np.ndarray:
        """
        Encode Watson-Crick pairing
        A-T/U: Consonant interval (perfect fifth)
        G-C: Stronger consonant (perfect fourth) - 3 H-bonds
        """
        if structure is None:
            # Assume all paired for DNA
            structure = self._predict_pairing()
        
        audio_chunks = []
        
        for i, (base, pair_status) in enumerate(zip(self.sequence, structure)):
            base_freq = NUCLEOTIDE_FREQS.get(base, 440)
            
            if pair_status == 'paired':
                # Determine pair type
                if base in ['A', 'T', 'U']:
                    # A-T or A-U: Perfect fifth (3:2 ratio)
                    pair_freq = base_freq * 1.5
                elif base in ['G', 'C']:
                    # G-C: Perfect fourth (4:3 ratio)
                    pair_freq = base_freq * 1.333
                
                # Generate harmony
                chunk = self.generate_harmony(
                    frequencies=[base_freq, pair_freq],
                    duration=self.duration_per_base
                )
            else:
                # Unpaired - single tone
                chunk = self.generate_tone(
                    frequency=base_freq,
                    duration=self.duration_per_base
                )
            
            audio_chunks.append(chunk)
        
        return np.concatenate(audio_chunks)
    
    def encode_helix_structure(self) -> np.ndarray:
        """
        Encode helical twist using phase modulation
        DNA: 10.5 bp per turn
        RNA: varies
        """
        bases_per_turn = 10.5 if self.molecule_type == 'DNA' else 11.0
        
        audio_chunks = []
        
        for i, base in enumerate(self.sequence):
            # Calculate phase based on helical position
            phase = (i / bases_per_turn) * 2 * np.pi
            
            base_freq = NUCLEOTIDE_FREQS.get(base, 440)
            
            # Phase modulation creates "spiraling" effect
            t = np.linspace(0, self.duration_per_base, 
                          int(self.sample_rate * self.duration_per_base))
            audio = np.sin(2 * np.pi * base_freq * t + phase)
            
            audio_chunks.append(audio)
        
        return np.concatenate(audio_chunks)
    
    def encode_codons(self) -> np.ndarray:
        """
        Encode codons (triplets) for mRNA
        Each codon -> chord
        """
        if len(self.sequence) % 3 != 0:
            print("Warning: Sequence length not divisible by 3")
        
        audio_chunks = []
        
        for i in range(0, len(self.sequence) - 2, 3):
            codon = self.sequence[i:i+3]
            
            # Get frequencies for each base
            freqs = [NUCLEOTIDE_FREQS.get(base, 440) for base in codon]
            
            # Generate codon as chord (3x duration of single base)
            chunk = self.generate_harmony(
                frequencies=freqs,
                duration=self.duration_per_base * 3
            )
            
            audio_chunks.append(chunk)
        
        return np.concatenate(audio_chunks)
    
    def fuse_all_levels(self) -> np.ndarray:
        """Multi-level fusion for DNA/RNA"""
        # Base sequence (main signal)
        sequence_audio = self.encode_sequence()
        
        # Helical structure (phase modulation)
        helix_audio = self.encode_helix_structure()
        
        # Fuse
        if len(sequence_audio) == len(helix_audio):
            fused = 0.6 * sequence_audio + 0.4 * helix_audio
        else:
            fused = sequence_audio
        
        # Normalize
        fused = fused / np.max(np.abs(fused))
        
        return fused
    
    def _predict_pairing(self) -> str:
        """
        Simple pairing prediction
        (In production, use RNA folding algorithms)
        """
        # Simplified: assume double helix for DNA, all paired
        if self.molecule_type == 'DNA':
            return 'paired' * len(self.sequence)
        else:
            # RNA - more complex, would use folding algorithm
            return 'paired' * len(self.sequence)
    
    def save_audio(self, filename: str, audio: Optional[np.ndarray] = None):
        """Save audio to WAV file"""
        if audio is None:
            audio = self.fuse_all_levels()
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_int16)
        print(f"Saved {self.molecule_type} audio to {filename}")

# ==================== WAV2VEC 2.0 INTEGRATION ====================

class BiomolecularEmbedding:
    """
    Extract embeddings from biomolecular audio using Wav2Vec 2.0
    Transfer learning from pre-trained voice AI
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        """
        Parameters:
        -----------
        model_name : str
            Hugging Face model name
        """
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        
    def extract_embeddings(self, audio: np.ndarray, 
                          sample_rate: int = 16000) -> np.ndarray:
        """
        Extract Wav2Vec 2.0 embeddings
        
        Returns:
        --------
        embeddings : np.ndarray
            Shape: (time_steps, 768) for base model
        """
        # Prepare input
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Convert to numpy
        embeddings = embeddings.squeeze(0).numpy()
        
        return embeddings
    
    def mean_pooling(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Global mean pooling for fixed-size representation
        
        Returns:
        --------
        pooled : np.ndarray
            Shape: (768,)
        """
        return np.mean(embeddings, axis=0)

# ==================== EXAMPLE USAGE ====================

def example_protein():
    """Example: Insulin sonification"""
    # Insulin A-chain (21 residues)
    insulin_a = "GIVEQCCTSICSLYQLENYCN"
    
    print("Encoding Insulin A-chain...")
    protein = ProteinSonification(
        sequence=insulin_a,
        residues_per_second=10
    )
    
    # Encode with all levels
    audio = protein.fuse_all_levels()
    protein.save_audio("insulin_a_chain.wav", audio)
    
    # Extract embeddings
    print("Extracting Wav2Vec embeddings...")
    embedder = BiomolecularEmbedding()
    embeddings = embedder.extract_embeddings(audio)
    pooled = embedder.mean_pooling(embeddings)
    
    print(f"Audio duration: {len(audio) / 16000:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Pooled embedding: {pooled.shape}")
    
    return audio, pooled

def example_dna():
    """Example: CRISPR guide RNA"""
    # Example guide RNA sequence (20 nt target + scaffold)
    grna = "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCG"
    
    print("Encoding guide RNA...")
    rna = NucleicAcidSonification(
        sequence=grna,
        molecule_type='RNA',
        bases_per_second=20
    )
    
    # Encode
    audio = rna.fuse_all_levels()
    rna.save_audio("guide_rna.wav", audio)
    
    # Extract embeddings
    print("Extracting Wav2Vec embeddings...")
    embedder = BiomolecularEmbedding()
    embeddings = embedder.extract_embeddings(audio)
    pooled = embedder.mean_pooling(embeddings)
    
    print(f"Audio duration: {len(audio) / 16000:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Pooled embedding: {pooled.shape}")
    
    return audio, pooled

if __name__ == "__main__":
    print("="*60)
    print("Biomolecular Sonification Examples")
    print("="*60)
    
    # Protein example
    print("\n1. PROTEIN EXAMPLE:")
    protein_audio, protein_emb = example_protein()
    
    # DNA/RNA example
    print("\n2. RNA EXAMPLE:")
    rna_audio, rna_emb = example_dna()
    
    print("\n" + "="*60)
    print("Complete! Audio files and embeddings generated.")
    print("="*60)
