import h5py
import json
import keras
import tensorflow as tf

OLD_MODEL = "backend/models/audio_fusion_cnn_bilstm_model.h5"
FIXED_MODEL = "backend/models/audio_fusion_cnn_bilstm_model_fixed.keras"

REMOVE_KEYS = {
    "renorm",
    "renorm_clipping",
    "renorm_momentum"
}

with h5py.File(OLD_MODEL, "r+") as f:
    config = f.attrs.get("model_config")

    if isinstance(config, bytes):
        config = config.decode("utf-8")

    model_config = json.loads(config)

    def clean(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "BatchNormalization":
                cfg = obj.get("config", {})
                for key in REMOVE_KEYS:
                    cfg.pop(key, None)

            for value in obj.values():
                clean(value)

        elif isinstance(obj, list):
            for item in obj:
                clean(item)

    clean(model_config)

    f.attrs.modify(
        "model_config",
        json.dumps(model_config).encode("utf-8")
    )

model = tf.keras.models.load_model(
    OLD_MODEL,
    compile=False
)

model.save(FIXED_MODEL)

print("Fixed model saved:", FIXED_MODEL)
