"""
Quick script to test ONNX model with real images
"""

import sys
from pathlib import Path
from load_onnx_model import ONNXModelWrapper


def test_onnx_model(onnx_model_path='models/skin_model.onnx', test_image_path='/Users/ayaanfaisal/HackRUF25/backend/data/demo/IMG_8014.jpg'):
    """Test ONNX model with a real image."""
    
    print("="*70)
    print("TESTING ONNX MODEL")
    print("="*70)
    
    # Check if model exists
    model_path = Path(onnx_model_path)
    if not model_path.exists():
        print(f"\n❌ Error: ONNX model not found")
        print(f"   Looking for: {model_path.absolute()}")
        print("\nPlease export model first:")
        print("  python3 export_to_onnx.py")
        return False
    
    # Check if test image exists
    image_path = Path(test_image_path)
    if not image_path.exists():
        print(f"\n❌ Error: Test image cannot be read")
        print(f"   Looking for: {image_path.absolute()}")
        print("\nPlease provide a valid image path.")
        print("Update line 10 in test_onnx_model.py with the correct path.")
        return False
    
    # Load model
    print(f"\n📥 Loading ONNX model: {model_path}")
    try:
        model = ONNXModelWrapper(str(model_path))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with image
    print(f"\n🧪 Testing with image: {image_path}")
    
    try:
        result = model.predict(str(image_path))
        
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"\n✅ Prediction successful!")
        print(f"   Predicted class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        print(f"\n📊 Top 5 predictions:")
        for i, (cls_idx, prob) in enumerate(zip(result['top5_classes'], result['top5_probabilities']), 1):
            print(f"   {i}. Class {cls_idx:2d}: {prob:.2%}")
        
        # Show class names if available
        class_names_path = Path('class_names.txt')
        if class_names_path.exists():
            try:
                with open(class_names_path, 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
                
                if result['predicted_class'] < len(class_names):
                    print(f"\n📋 With class names:")
                    pred_name = class_names[result['predicted_class']]
                    print(f"   Predicted: {pred_name}")
                    print(f"   Confidence: {result['confidence']:.2%}")
                    
                    print(f"\n   Top 5:")
                    for i, (cls_idx, prob) in enumerate(zip(result['top5_classes'], result['top5_probabilities']), 1):
                        if cls_idx < len(class_names):
                            cls_name = class_names[cls_idx]
                            print(f"   {i}. {cls_name}: {prob:.2%}")
            except Exception as e:
                print(f"\n⚠️  Could not load class names: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ONNX model with a specific image')
    parser.add_argument('--model', type=str, default='models/skin_model.onnx',
                       help='Path to ONNX model (default: models/skin_model.onnx)')
    parser.add_argument('--image', type=str, default='data/Healthy Skin/1-1-59_6128b3f32ba4b.jpg',
                       help='Path to test image')
    
    args = parser.parse_args()
    
    # Run test
    success = test_onnx_model(args.model, args.image)
    
    if success:
        print("\n" + "="*70)
        print("✅ TEST COMPLETE")
        print("="*70)
        print("\nYour ONNX model is working correctly! 🎉")
        print("\nTo test with a different image:")
        print(f"  python3 test_onnx_model.py --image path/to/your/image.jpg")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        sys.exit(1)
