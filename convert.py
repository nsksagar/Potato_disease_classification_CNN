import tensorflow as tf

print("Loading native Keras model...")
model = tf.keras.models.load_model('models/1.keras')

print("Exporting to SavedModel format...")
# Keras built-in exporter correctly tracks all variables automatically
model.export('models/1')

print("Success! Your model has been converted to 'models/1'.")