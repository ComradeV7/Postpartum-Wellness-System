"""
Script to automatically fix model compatibility issues.
This will delete incompatible models and prepare for retraining.
"""
import os
import sys

MODEL_PATH = "voice_model.h5"
ENCODER_PATH = "voice_emotion_encoder.joblib"

def auto_fix_model_compatibility():
    """Automatically detect and fix incompatible models."""
    print("="*70)
    print("AUTOMATIC MODEL COMPATIBILITY FIX")
    print("="*70)
    
    files_to_delete = []
    incompatible_detected = False
    
    # Check if model file exists
    if os.path.exists(MODEL_PATH):
        print(f"✓ Found model file: {MODEL_PATH}")
        
        # Check if it's incompatible by examining error patterns
        # We'll try to load it to detect the issue
        try:
            import tensorflow as tf
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("✅ Model is compatible - no action needed!")
                return True
            except Exception as e:
                error_str = str(e).lower()
                if 'batch_shape' in error_str or 'unrecognized keyword' in error_str:
                    print("❌ INCOMPATIBLE MODEL DETECTED")
                    print(f"   Error: {str(e)[:150]}")
                    incompatible_detected = True
                    files_to_delete.append(MODEL_PATH)
        except ImportError:
            print("⚠ Cannot check compatibility - TensorFlow not available")
            # Assume incompatible if we can't check
            incompatible_detected = True
            files_to_delete.append(MODEL_PATH)
    else:
        print(f"ℹ Model file not found: {MODEL_PATH}")
        print("   (This is okay - you can train a new one)")
    
    # Check encoder
    if os.path.exists(ENCODER_PATH):
        print(f"✓ Found encoder file: {ENCODER_PATH}")
        if incompatible_detected:
            files_to_delete.append(ENCODER_PATH)
    
    # Auto-fix if incompatible
    if incompatible_detected and files_to_delete:
        print("\n" + "="*70)
        print("AUTO-FIXING INCOMPATIBLE FILES...")
        print("="*70)
        
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"✓ Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Failed to delete {file_path}: {e}")
        
        if deleted_count > 0:
            print("\n" + "="*70)
            print("✅ INCOMPATIBLE FILES REMOVED!")
            print("="*70)
            print("\n📋 NEXT STEPS:")
            print("1. Retrain the model with the updated script:")
            print("   python train_voice_model.py")
            print("\n2. The new model will be fully compatible")
            print("3. Training will use GPU if available (faster)")
            print("\n" + "="*70)
            return True
    else:
        if not files_to_delete:
            print("\n✅ No incompatible files detected - you're all set!")
        return False

if __name__ == "__main__":
    try:
        auto_fix_model_compatibility()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)

