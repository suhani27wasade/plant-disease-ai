import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/validation"

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("Classes found:")
print(class_names)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

# Model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Create model folder
Path("model").mkdir(exist_ok=True)

# Save model
model.save("model/plant_disease_model.keras")

# Save class names
with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n==============================\n")
print("MODEL TRAINING COMPLETED!")
print("==============================")
print("Model saved at:")
print("model/plant_disease_model.keras")
