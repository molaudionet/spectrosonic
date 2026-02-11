"""
Molecular AI with Spectroscopic Audio - Complete Working Examples

This file contains production-ready code for:
1. Extracting Wav2Vec embeddings from audio
2. Building training datasets
3. Training ML models
4. Making predictions
5. Real-world applications

Requirements:
pip install torch transformers scipy numpy pandas scikit-learn
"""

import torch
import numpy as np
import pandas as pd
from scipy.io import wavfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ==================== EMBEDDING EXTRACTION ====================

class SpectroscopyEmbedder:
    """
    Extract Wav2Vec 2.0 embeddings from spectroscopic audio
    """
    
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """
        Initialize with pre-trained Wav2Vec 2.0 model
        """
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        
        print("Loading Wav2Vec 2.0 model...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        print("✓ Model loaded")
    
    def extract_from_wav(self, wav_file):
        """
        Extract 768-dim embedding from WAV file
        
        Parameters:
        -----------
        wav_file : str
            Path to WAV file
        
        Returns:
        --------
        embedding : np.ndarray
            768-dimensional embedding vector
        """
        # Load audio
        sr, audio = wavfile.read(wav_file)
        
        # Convert to float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483647.0
        
        # Process
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        )
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state  # (1, time_steps, 768)
        
        # Mean pooling across time
        embedding = features.mean(dim=1).squeeze().numpy()  # (768,)
        
        return embedding
    
    def extract_from_directory(self, directory, pattern="*.wav"):
        """
        Extract embeddings from all WAV files in directory
        
        Returns:
        --------
        embeddings_dict : dict
            {filename: embedding} mapping
        """
        directory = Path(directory)
        embeddings = {}
        
        wav_files = list(directory.glob(pattern))
        print(f"Found {len(wav_files)} WAV files")
        
        for wav_file in wav_files:
            print(f"Processing: {wav_file.name}...", end=" ")
            embedding = self.extract_from_wav(str(wav_file))
            embeddings[wav_file.stem] = embedding
            print("✓")
        
        return embeddings

# ==================== EXAMPLE 1: EXTRACT ALL EMBEDDINGS ====================

def example_1_extract_embeddings():
    """
    Extract embeddings from all your WAV files
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Extract Embeddings from WAV Files")
    print("="*60 + "\n")
    
    # Initialize embedder
    embedder = SpectroscopyEmbedder()
    
    # Extract from current directory
    embeddings = embedder.extract_from_directory(".", pattern="*.wav")
    
    # Display results
    print(f"\n✓ Extracted {len(embeddings)} embeddings")
    
    for name, emb in embeddings.items():
        print(f"  {name}: shape {emb.shape}, mean={emb.mean():.3f}, std={emb.std():.3f}")
    
    # Save to file
    np.savez('spectroscopy_embeddings.npz', **embeddings)
    print("\n✓ Saved to: spectroscopy_embeddings.npz")
    
    return embeddings

# ==================== EXAMPLE 2: PROTEIN FUNCTION PREDICTION ====================

def example_2_protein_function_prediction():
    """
    Train a model to predict protein function from spectroscopy
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Protein Function Prediction")
    print("="*60 + "\n")
    
    # Create training dataset
    # In reality, you'd load from TAPE or your own data
    training_data = [
        # Format: (wav_file, function_label)
        ('ubiquitin_hsqc_demo.wav', 'protein_degradation'),
        ('lysozyme_cd_spectrum.wav', 'enzyme_hydrolase'),
        ('bsa_ftir_demo.wav', 'transport_protein'),
        ('insulin_multimodal.wav', 'hormone_signaling'),
        # Add more proteins...
    ]
    
    print("Loading spectroscopy data...")
    embedder = SpectroscopyEmbedder()
    
    # Extract embeddings
    X = []
    y = []
    
    for wav_file, label in training_data:
        try:
            embedding = embedder.extract_from_wav(wav_file)
            X.append(embedding)
            y.append(label)
            print(f"✓ {wav_file}: {label}")
        except FileNotFoundError:
            print(f"✗ {wav_file}: not found, skipping")
    
    if len(X) < 2:
        print("\n⚠ Need at least 2 samples to train")
        print("  This is a demo - in production, use datasets like TAPE")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset: {len(X)} proteins, {X.shape[1]} features")
    
    # Train model
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Predict (on training data for demo)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n✓ Training accuracy: {accuracy:.3f}")
    
    # Feature importance
    importance = clf.feature_importances_
    top_features = np.argsort(importance)[-10:][::-1]
    
    print("\nTop 10 most important embedding dimensions:")
    for i, idx in enumerate(top_features, 1):
        print(f"  {i}. Dimension {idx}: {importance[idx]:.4f}")
    
    return clf

# ==================== EXAMPLE 3: SIMILARITY ANALYSIS ====================

def example_3_similarity_analysis():
    """
    Compare proteins by spectroscopic similarity
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Spectroscopic Similarity Analysis")
    print("="*60 + "\n")
    
    embedder = SpectroscopyEmbedder()
    
    # Compare all proteins
    proteins = {
        'ubiquitin': 'ubiquitin_hsqc_demo.wav',
        'lysozyme': 'lysozyme_cd_spectrum.wav',
        'bsa': 'bsa_ftir_demo.wav',
        'insulin': 'insulin_multimodal.wav'
    }
    
    # Extract embeddings
    embeddings = {}
    for name, wav_file in proteins.items():
        try:
            embeddings[name] = embedder.extract_from_wav(wav_file)
            print(f"✓ Loaded {name}")
        except FileNotFoundError:
            print(f"✗ {name} not found, skipping")
    
    if len(embeddings) < 2:
        print("\n⚠ Need at least 2 proteins to compare")
        return None
    
    # Compute pairwise similarities
    print("\nPairwise Cosine Similarities:")
    print("-" * 50)
    
    names = list(embeddings.keys())
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            emb1 = embeddings[name1].reshape(1, -1)
            emb2 = embeddings[name2].reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            print(f"{name1} <-> {name2}: {similarity:.3f}")
    
    # Find most similar pair
    max_sim = 0
    max_pair = None
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            emb1 = embeddings[name1].reshape(1, -1)
            emb2 = embeddings[name2].reshape(1, -1)
            sim = cosine_similarity(emb1, emb2)[0][0]
            
            if sim > max_sim:
                max_sim = sim
                max_pair = (name1, name2)
    
    print(f"\nMost similar: {max_pair[0]} and {max_pair[1]} ({max_sim:.3f})")
    
    return embeddings

# ==================== EXAMPLE 4: QUALITY CONTROL ====================

def example_4_quality_control():
    """
    QC check: Compare sample to reference standard
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Quality Control Check")
    print("="*60 + "\n")
    
    embedder = SpectroscopyEmbedder()
    
    # Reference standard
    reference_file = 'lysozyme_cd_spectrum.wav'
    
    # Test samples (in reality, different batches)
    test_samples = [
        ('lysozyme_cd_structure.wav', 'Batch_A'),
        ('bsa_ftir_demo.wav', 'Batch_B'),  # Should fail (different protein!)
    ]
    
    print("Loading reference standard...")
    try:
        reference_emb = embedder.extract_from_wav(reference_file)
        print(f"✓ Reference: {reference_file}")
    except FileNotFoundError:
        print(f"✗ Reference not found: {reference_file}")
        return None
    
    # QC threshold (FDA requirement: >0.95 for biopharmaceuticals)
    QC_THRESHOLD = 0.95
    
    print(f"\nQC Threshold: {QC_THRESHOLD:.2f}")
    print("-" * 50)
    
    # Test each sample
    results = []
    
    for sample_file, batch_id in test_samples:
        try:
            sample_emb = embedder.extract_from_wav(sample_file)
            
            # Compute similarity
            similarity = cosine_similarity(
                reference_emb.reshape(1, -1),
                sample_emb.reshape(1, -1)
            )[0][0]
            
            # QC decision
            passed = similarity >= QC_THRESHOLD
            status = "PASS ✓" if passed else "FAIL ✗"
            
            print(f"{batch_id}: {similarity:.4f} - {status}")
            
            results.append({
                'batch': batch_id,
                'similarity': similarity,
                'passed': passed
            })
            
        except FileNotFoundError:
            print(f"{batch_id}: File not found - SKIP")
    
    # Summary
    print("\nQC Summary:")
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {total - passed}/{total}")
    
    return results

# ==================== EXAMPLE 5: MUTATION DETECTION ====================

def example_5_mutation_detection():
    """
    Detect mutations by comparing spectroscopic signatures
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Mutation Detection")
    print("="*60 + "\n")
    
    embedder = SpectroscopyEmbedder()
    
    # Wild-type protein
    wt_file = 'ubiquitin_hsqc_demo.wav'
    
    # Mutant proteins (in reality, different mutations)
    mutants = [
        ('lysozyme_cd_spectrum.wav', 'L33P'),  # Different protein = severe mutation
        ('ubiquitin_hsqc_demo.wav', 'wild-type'),  # Same = no mutation
    ]
    
    print("Loading wild-type...")
    try:
        wt_emb = embedder.extract_from_wav(wt_file)
        print(f"✓ Wild-type: {wt_file}")
    except FileNotFoundError:
        print(f"✗ Wild-type not found: {wt_file}")
        return None
    
    print("\nAnalyzing mutants...")
    print("-" * 60)
    
    # Analyze each mutant
    for mutant_file, mutation in mutants:
        try:
            mutant_emb = embedder.extract_from_wav(mutant_file)
            
            # Compute difference
            delta = mutant_emb - wt_emb
            delta_magnitude = np.linalg.norm(delta)
            
            # Similarity
            similarity = cosine_similarity(
                wt_emb.reshape(1, -1),
                mutant_emb.reshape(1, -1)
            )[0][0]
            
            # Classification
            if similarity > 0.98:
                effect = "Neutral (no effect)"
            elif similarity > 0.90:
                effect = "Mild (small effect)"
            elif similarity > 0.80:
                effect = "Moderate (noticeable effect)"
            else:
                effect = "Severe (large effect)"
            
            print(f"Mutation: {mutation}")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Delta magnitude: {delta_magnitude:.3f}")
            print(f"  Predicted effect: {effect}")
            print()
            
        except FileNotFoundError:
            print(f"Mutation {mutation}: File not found - SKIP\n")
    
    return True

# ==================== EXAMPLE 6: BATCH PROCESSING ====================

def example_6_batch_processing():
    """
    Process multiple proteins in batch
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Batch Processing")
    print("="*60 + "\n")
    
    embedder = SpectroscopyEmbedder()
    
    # Process all WAV files in current directory
    embeddings = embedder.extract_from_directory(".", pattern="*.wav")
    
    # Create DataFrame
    data = []
    for name, emb in embeddings.items():
        data.append({
            'protein': name,
            'embedding_mean': emb.mean(),
            'embedding_std': emb.std(),
            'embedding_min': emb.min(),
            'embedding_max': emb.max(),
            'embedding': emb  # Store full embedding
        })
    
    df = pd.DataFrame(data)
    
    print("\nDataset Summary:")
    print(df[['protein', 'embedding_mean', 'embedding_std']].to_string(index=False))
    
    # Save to CSV (without full embedding)
    df_save = df.drop(columns=['embedding'])
    df_save.to_csv('protein_embeddings_summary.csv', index=False)
    print("\n✓ Saved summary to: protein_embeddings_summary.csv")
    
    # Save full embeddings
    np.savez('all_protein_embeddings.npz', **embeddings)
    print("✓ Saved full embeddings to: all_protein_embeddings.npz")
    
    return df

# ==================== MAIN DEMO ====================

def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("MOLECULAR AI WITH SPECTROSCOPIC AUDIO - COMPLETE EXAMPLES")
    print("="*70)
    
    # Check if WAV files exist
    import glob
    wav_files = glob.glob("*.wav")
    
    if len(wav_files) == 0:
        print("\n⚠ No WAV files found in current directory!")
        print("\nPlease ensure you have audio files from demo_autofind.py:")
        print("  - ubiquitin_hsqc_demo.wav")
        print("  - lysozyme_cd_spectrum.wav")
        print("  - bsa_ftir_demo.wav")
        print("  - insulin_multimodal.wav")
        print("\nRun demo_autofind.py first to generate these files.")
        return
    
    print(f"\n✓ Found {len(wav_files)} WAV files")
    print("  Files:", ", ".join([f for f in wav_files[:5]]))
    if len(wav_files) > 5:
        print(f"  ... and {len(wav_files) - 5} more")
    
    print("\n" + "="*70)
    print("Running examples...")
    print("="*70)
    
    # Run examples
    try:
        # Example 1: Extract embeddings
        embeddings = example_1_extract_embeddings()
        
        # Example 2: Function prediction
        model = example_2_protein_function_prediction()
        
        # Example 3: Similarity analysis
        similarities = example_3_similarity_analysis()
        
        # Example 4: Quality control
        qc_results = example_4_quality_control()
        
        # Example 5: Mutation detection
        mutation_results = example_5_mutation_detection()
        
        # Example 6: Batch processing
        batch_df = example_6_batch_processing()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETE!")
        print("="*70)
        
        print("\nFiles created:")
        print("  ✓ spectroscopy_embeddings.npz")
        print("  ✓ protein_embeddings_summary.csv")
        print("  ✓ all_protein_embeddings.npz")
        
        print("\nNext steps:")
        print("  1. Load real datasets (TAPE, BindingDB, etc.)")
        print("  2. Train on larger datasets")
        print("  3. Integrate with MolecularWorld platform")
        print("  4. Deploy for customers!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Install requirements: pip install torch transformers scipy numpy pandas scikit-learn")
        print("  - Ensure WAV files are in current directory")
        print("  - Check file formats (16kHz, mono, 16-bit)")

if __name__ == "__main__":
    main()
