import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import os
import joblib 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- GPU Configuration ---
def configure_gpu():
    """Configure TensorFlow to use GPU if available."""
    print("\n" + "="*60)
    print("GPU Configuration Check")
    print("="*60)
    
    # List all available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu.name}")
            
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
            
            # Set GPU as default if multiple devices available
            if len(gpus) > 1:
                print(f"✓ Using GPU: {gpus[0].name}")
            
            # Enable mixed precision for faster training (if supported)
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("✓ Mixed precision enabled (float16) for faster training")
            except:
                print("ℹ Mixed precision not available (using float32)")
            
            print("\nGPU training enabled - Training will be faster! 🚀")
            return True
            
        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
            print("Falling back to CPU")
            return False
    else:
        print("✗ No GPU detected - Training will use CPU")
        print("  (Install CUDA and cuDNN for GPU support)")
        return False
    
    print("="*60 + "\n")

# Check GPU at import time
GPU_AVAILABLE = configure_gpu()

# --- Constants ---
# Make sure this path is correct (matches actual folder name)
DATA_PATH = "data/audio_speech_actors_01-24/" 
MODEL_SAVE_PATH = "voice_model.h5"
ENCODER_SAVE_PATH = "voice_emotion_encoder.joblib"
SAMPLE_RATE = 22050
TARGET_EMOTIONS = ['anxious', 'calm', 'happy', 'neutral', 'sad', 'angry'] 

# RAVDESS file format: 03-01-EMOTION-01-01-01-ACTOR.wav
# Mapping RAVDESS emotions to our app's target emotions
EMOTION_MAP = {
    '01': 'neutral', # neutral
    '02': 'calm',    # calm
    '03': 'happy',   # happy
    '04': 'sad',     # sad
    '05': 'angry',   # angry
    '06': 'anxious', # fearful (using as 'anxious')
    '07': 'sad',     # disgust (mapping to 'sad' as a general negative valence)
    '08': 'happy'    # surprised (mapping to 'happy' as high arousal, positive)
}

from audio_features import extract_features, FEATURE_VECTOR_LENGTH

# --- Data Loading Function ---
def load_data(data_path):
    """
    Loads all RAVDESS data and extracts features.
    """
    features = []
    labels = []
    
    print(f"Loading audio files from: {data_path}")
    
    # Use glob to find all .wav files
    file_list = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)
    if not file_list:
        print(f"CRITICAL ERROR: No .wav files found in {data_path}")
        print("Please check that you have downloaded and unzipped the RAVDESS dataset correctly.")
        return None, None
        
    for file_path in file_list:
        try:
            # Parse emotion from filename
            basename = os.path.basename(file_path)
            emotion_code = basename.split('-')[2]
            emotion = EMOTION_MAP.get(emotion_code)
            
            # If it's one of the emotions we care about, process it
            if emotion in TARGET_EMOTIONS:
                # Load and extract
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=3.0) # Load 3 seconds
                feature_vector = extract_features(y, sr)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(emotion)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    print(f"Loaded {len(features)} audio samples.")
    return np.array(features), np.array(labels)

# --- Build the Model ---
def build_model(input_shape, num_classes):
    """
    Builds a 1D-CNN model for audio classification.
    Input shape must be (features, 1), e.g., (180, 1)
    Uses compatible API for all TensorFlow versions.
    """
    # Use functional API for better compatibility
    input_layer = tf.keras.Input(shape=input_shape, name='input_audio')
    
    # Layer 1
    x = Conv1D(128, 5, padding='same', activation='relu', name='conv1d_1')(input_layer)
    x = MaxPooling1D(2, name='maxpool_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)
    
    # Layer 2
    x = Conv1D(256, 5, padding='same', activation='relu', name='conv1d_2')(x)
    x = MaxPooling1D(2, name='maxpool_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)

    # Layer 3
    x = Conv1D(512, 5, padding='same', activation='relu', name='conv1d_3')(x)
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Dense Layer
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(0.3, name='dropout_3')(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model using functional API
    model = tf.keras.Model(inputs=input_layer, outputs=output, name='voice_emotion_model')
    
    # Compile model
    # Mixed precision is handled by policy, optimizer is standard
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )
    return model

# --- Main Training Function ---
def main():
    print("\n" + "="*60)
    print("VOICE EMOTION RECOGNITION MODEL TRAINING")
    print("="*60)
    
    if GPU_AVAILABLE:
        print("\n🚀 GPU TRAINING MODE - Faster training enabled")
    else:
        print("\n💻 CPU TRAINING MODE - Will take longer")
        print("   Tip: Install CUDA + cuDNN for GPU acceleration\n")
    
    # 1. Load data
    X, y = load_data(DATA_PATH)
    if X is None or X.shape[0] == 0:
        print("Training aborted due to data loading error.")
        return
        
    # 2. Encode labels
    # LabelEncoder: 'happy' -> 2
    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_EMOTIONS) # Fit on all possible targets
    y_encoded = label_encoder.transform(y)
    
    # OneHotEncoder: 2 -> [0, 0, 1, 0, 0]
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))
    
    # Save the label encoder
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)
    print(f"\nSaved label encoder to {ENCODER_SAVE_PATH}")
    print(f"Classes: {list(label_encoder.classes_)}\n")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
    )
    
    # 4. Reshape data for 1D-CNN
    # (samples, features) -> (samples, features, 1)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # 5. Build model
    input_shape = (X_train.shape[1], 1) # (180, 1)
    num_classes = len(label_encoder.classes_)
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # 6. Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)
    ]
    
    # 7. Train model
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    # Adjust batch size based on GPU availability
    batch_size = 64 if GPU_AVAILABLE else 32
    print(f"Batch size: {batch_size} ({'GPU' if GPU_AVAILABLE else 'CPU'} mode)")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150, # Set a high number, EarlyStopping will find the best
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. Evaluate model
    print("\n" + "="*60)
    print("Evaluating Model...")
    print("="*60)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # 9. Save model with compatibility settings
    print("\nSaving model files...")
    try:
        # Save in SavedModel format (most compatible)
        model.save(MODEL_SAVE_PATH, save_format='h5')
        print(f"✓ Saved trained model to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"⚠ Error saving in h5 format: {e}")
        print("Trying SavedModel format...")
        try:
            # Fallback to SavedModel format
            model.save(MODEL_SAVE_PATH.replace('.h5', '_savedmodel'), save_format='tf')
            print(f"✓ Saved trained model to: {MODEL_SAVE_PATH.replace('.h5', '_savedmodel')}")
        except Exception as e2:
            print(f"✗ Error saving model: {e2}")
    
    print(f"✓ Saved encoder to: {ENCODER_SAVE_PATH}")
    print("\n✅ Training complete! You can now run: streamlit run app.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

