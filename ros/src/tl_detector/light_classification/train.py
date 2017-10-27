from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from model import Model

# Start
print("Running on TensorFlow " + str(tf.__version__))

# Height and width to convert our images to before feeding in
img_width = 224
img_height = 224
input_shape = (img_width, img_height, 3)

# Set configuration
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 890
nb_validation_samples = 160
epochs = 50
batch_size = 16

# Compile
model = Model(input_shape)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Shear, zoom, and flip training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Just rescale color for test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Training data
print("Training:")
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Validation data
print("Validation:")
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Train
model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    epochs=epochs)

# Save
model.save_weights('model.h5')


