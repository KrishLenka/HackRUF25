"""
Complete ONNX Workflow Example

This script demonstrates the entire workflow from training to deployment
with ONNX model versioning.
"""

import os
from pathlib import Path


def workflow_example():
    """
    Complete workflow showing how to:
    1. Train a model
    2. Export to ONNX
    3. Store multiple versions
    4. Compare versions
    5. Deploy best version
    """
    
    print("="*70)
    print("COMPLETE ONNX WORKFLOW EXAMPLE")
    print("="*70)
    
    # STEP 1: Train initial model
    print("\n📚 STEP 1: Train Initial Model")
    print("-"*70)
    print("Commands:")
    print("  cd backend")
    print("  python train.py --num-classes 25 --epochs 50")
    print("\n→ Creates: models/pt_skin_model.pth")
    print("→ Accuracy: ~66%")
    
    # STEP 2: Export to ONNX (baseline)
    print("\n📦 STEP 2: Export Baseline Model to ONNX")
    print("-"*70)
    print("Commands:")
    print("  python export_to_onnx.py \\")
    print("    --model models/pt_skin_model.pth \\")
    print("    --output models/skin_baseline_v1.onnx \\")
    print("    --benchmark")
    print("\n→ Creates:")
    print("  • models/skin_baseline_v1.onnx (ONNX model)")
    print("  • models/skin_baseline_v1.json (metadata)")
    print("→ Speedup: ~2x faster than PyTorch")
    
    # STEP 3: Train improved model
    print("\n🚀 STEP 3: Train Improved Model")
    print("-"*70)
    print("Commands:")
    print("  # Add class weights and use EfficientNet-B3")
    print("  python train.py \\")
    print("    --num-classes 25 \\")
    print("    --model efficientnet_b3 \\")
    print("    --epochs 100")
    print("\n→ Creates: models/pt_skin_model.pth (overwrites)")
    print("→ Expected accuracy: ~75%")
    
    # STEP 4: Export improved model
    print("\n📦 STEP 4: Export Improved Model")
    print("-"*70)
    print("Commands:")
    print("  python export_to_onnx.py \\")
    print("    --model models/pt_skin_model.pth \\")
    print("    --model-name efficientnet_b3 \\")
    print("    --output models/skin_improved_v2.onnx")
    print("\n→ Creates:")
    print("  • models/skin_improved_v2.onnx")
    print("  • models/skin_improved_v2.json")
    
    # STEP 5: Compare versions
    print("\n📊 STEP 5: Compare Model Versions")
    print("-"*70)
    print("Python code:")
    print("""
from load_onnx_model import ONNXModelWrapper
from pathlib import Path

# Load both versions
baseline = ONNXModelWrapper('models/skin_baseline_v1.onnx')
improved = ONNXModelWrapper('models/skin_improved_v2.onnx')

# Test on validation images
test_images = list(Path('data/val').rglob('*.jpg'))[:100]

baseline_correct = 0
improved_correct = 0

for img in test_images:
    actual_class = img.parent.name
    
    pred_baseline = baseline.predict(str(img))['predicted_class']
    pred_improved = improved.predict(str(img))['predicted_class']
    
    # (Compare with actual class)
    # baseline_correct += (pred_baseline == actual_class_idx)
    # improved_correct += (pred_improved == actual_class_idx)

print(f"Baseline accuracy: {baseline_correct/100:.1%}")
print(f"Improved accuracy: {improved_correct/100:.1%}")
""")
    print("\n→ Results:")
    print("  Baseline: 66.5%")
    print("  Improved: 74.8%")
    print("  ✅ Improved model is better!")
    
    # STEP 6: Deploy to production
    print("\n🚀 STEP 6: Deploy Best Model to Production")
    print("-"*70)
    print("Commands:")
    print("  # Copy best model to production name")
    print("  cp models/skin_improved_v2.onnx models/onnx_skin_model.onnx")
    print("  cp models/skin_improved_v2.json models/onnx_skin_model.json")
    print()
    print("  # Archive old version")
    print("  mkdir -p models/archive")
    print("  mv models/skin_baseline_v1.* models/archive/")
    print()
    print("  # Restart Flask API")
    print("  python app.py")
    print("\n→ Flask API now uses improved ONNX model!")
    print("→ 2x faster inference + 8% better accuracy")
    
    # STEP 7: Continue iterating
    print("\n🔄 STEP 7: Continue Iterating")
    print("-"*70)
    print("Future improvements:")
    print("  1. Train with more data")
    print("  2. Try different architectures")
    print("  3. Export each as ONNX")
    print("  4. Compare all versions")
    print("  5. Deploy best one")
    print()
    print("Version history:")
    print("  models/")
    print("  ├── onnx_skin_model.onnx          # Current production")
    print("  ├── archive/")
    print("  │   ├── skin_baseline_v1.onnx     # 66% accuracy")
    print("  │   └── skin_improved_v2.onnx     # 75% accuracy")
    print("  └── experiments/")
    print("      ├── skin_resnet50.onnx        # Testing")
    print("      └── skin_ensemble.onnx         # Testing")
    
    print("\n" + "="*70)
    print("✅ COMPLETE WORKFLOW")
    print("="*70)
    print("\nYou now know how to:")
    print("  ✓ Export PyTorch models to ONNX")
    print("  ✓ Manage multiple model versions")
    print("  ✓ Compare model performance")
    print("  ✓ Deploy best model to production")
    print("  ✓ Iterate and improve over time")
    print("\n🎉 You're ready to use ONNX in production!")


def directory_structure_example():
    """Show recommended directory structure."""
    
    print("\n" + "="*70)
    print("RECOMMENDED DIRECTORY STRUCTURE")
    print("="*70 + "\n")
    
    structure = """
backend/
├── models/
│   ├── onnx_skin_model.onnx          ← PRODUCTION (current)
│   ├── onnx_skin_model.json          ← Metadata
│   │
│   ├── pt_skin_model.pth             ← Latest PyTorch
│   ├── model_config.json             ← Training config
│   │
│   ├── archive/                      ← Old versions
│   │   ├── skin_v1.0.onnx            (66% accuracy)
│   │   ├── skin_v1.0.json
│   │   ├── skin_v1.1.onnx            (68% accuracy)
│   │   ├── skin_v1.1.json
│   │   ├── skin_v2.0.onnx            (75% accuracy)
│   │   └── skin_v2.0.json
│   │
│   └── experiments/                  ← Testing new ideas
│       ├── skin_efficientb4.onnx
│       ├── skin_resnet50.onnx
│       ├── skin_ensemble.onnx
│       └── ...
│
├── export_to_onnx.py                 ← Export script
├── load_onnx_model.py                ← Loading utilities
├── compare_models.py                 ← Version comparison
└── app.py                            ← Flask API (auto-detects ONNX)
"""
    
    print(structure)
    
    print("\n📋 Naming Convention:")
    print("  • skin_<description>_v<version>.onnx")
    print("  • Examples:")
    print("    - skin_baseline_v1.onnx")
    print("    - skin_efficientb3_v1.onnx")
    print("    - skin_balanced_v2.onnx")
    print("    - skin_production.onnx")


if __name__ == "__main__":
    workflow_example()
    directory_structure_example()
    
    print("\n" + "="*70)
    print("QUICK START")
    print("="*70)
    print("\n1. Install ONNX:")
    print("   pip install onnx onnxruntime")
    print("\n2. Export your model:")
    print("   python export_to_onnx.py")
    print("\n3. Use it:")
    print("   from load_onnx_model import ONNXModelWrapper")
    print("   model = ONNXModelWrapper('models/skin_model.onnx')")
    print("   result = model.predict('image.jpg')")
    print("\n4. Deploy:")
    print("   cp models/skin_model.onnx models/onnx_skin_model.onnx")
    print("   python app.py")
    print("\n🚀 Done!")
