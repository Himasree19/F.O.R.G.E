from backend.services.fake_image_detector import model, WEIGHTS_PATH

print("Weights path:", WEIGHTS_PATH)

try:
    model.load_weights(WEIGHTS_PATH)
    print("✅ Full weights loaded correctly")
except Exception as e:
    print("❌ Full weights loading failed:")
    print(e)

model.summary()