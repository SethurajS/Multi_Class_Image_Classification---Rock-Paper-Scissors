import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping



print("Tensorflow : {}".format(tf.__version__))
print("Numpy : {}".format(np.__version__))


BASE_DIR = 'D:\Multi_Class_Image_Classification\Rock-Paper-Scissors'

train_dir = os.path.join(BASE_DIR, 'train')  # DOWNLOAD DATA-SET
test_dir = os.path.join(BASE_DIR, 'test')  # DOWNLOAD DATA-SET
validation_dir = os.path.join(BASE_DIR, 'validation')  # DOWNLOAD DATA-SET

OUTPUT_DIR = 'D:\Multi_Class_Image_Classification\Rock-Paper-Scissors\Model'

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)






train_num_data = []
train_num_classes = []
test_num_data = []
test_num_classes = []

for (dirpath, dirnames, filenames) in os.walk(train_dir):
  train_num_data.append(len(filenames))
  for dir in dirnames:
     train_num_classes.append(dir)

for (dirpath, dirnames, filenames) in os.walk(test_dir):
  test_num_data.append(len(filenames))
  for dir in dirnames:
     test_num_classes.append(dir)

print("TRAIN_DATA : \n")
for i in range(len(train_num_classes)):
  print("Number of data in Train_class {} : {}".format(train_num_classes[i], train_num_data[i+1]))

print("\n")

print("TEST_DATA : \n")
for i in range(len(test_num_classes)):
  print("Number of data in Test_class {} : {}".format(test_num_classes[i], test_num_data[i+1]))

print("\n")

print("VALIDATION_DATA : \n")
print("Number of data for Validation : {}".format(len(validation_dir)))



paper_data_dir = os.path.join(train_dir, "paper")
rock_data_dir = os.path.join(train_dir, "rock")
scissor_data_dir = os.path.join(train_dir, "scissors")
labels = os.listdir(train_dir)



# dirs = [paper_data_dir, rock_data_dir, scissor_data_dir]
# classes = ["paper", "rock", "scissors"]
# for i in range(len(dirs)):
#   for dir, name, files in os.walk(dirs[i]):
#     for files in os.listdir(dirs[i])[:2]:
#       image = mpimg.imread(os.path.join(dir, files))
#       plt.imshow(image)
#       plt.title(classes[i])
#       plt.axis('Off')
#       plt.show()


def preprocess_Image(train_dir, test_dir, labels=None, image_size=(150, 150), batch_size=50):

  train_dataGen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.0,
    horizontal_flip=False,
    vertical_flip=False
  )

  test_dataGen = ImageDataGenerator(rescale=1. / 255)

  train_images = train_dataGen.flow_from_directory(train_dir, target_size=image_size, class_mode='categorical',
                                                    batch_size=batch_size, shuffle=True,
                                                    )

  test_images = test_dataGen.flow_from_directory(test_dir, target_size=image_size, class_mode='categorical',
                                                  batch_size=batch_size, shuffle=False, subset=None,
                                                  )

  return train_images, test_images, test_dataGen


def build_model(input_shape, num_classes):

  model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
  ])

  return model


def model_history(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label="Training Accuracy")
  plt.plot(val_acc, label="Validation Accuracy")
  plt.legend(loc='lower right')
  plt.ylabel("Accuracy")
  plt.ylim([min(plt.ylim()), 1])
  plt.title("Training and Validation Accuracy")

  plt.subplot(2, 1, 2)
  plt.plot(loss, label="Training Loss")
  plt.plot(val_loss, label="Validation Loss")
  plt.legend(loc='upper right')
  plt.ylabel("Loss")
  plt.ylim([min(plt.ylim()), 1])
  plt.title("Training and Validation Loss")

  plt.show()




lr_rate = 0.0001
min_lr = 0.00001
learning_rate_decay_factor = 0.5
patience = 4.0
verbose = 1.0


def train_and_evaluate_model(model, name="", epochs=25, batch_size=32, verbose=verbose, checkpt=False):
  model.summary()

  model_save_dir = os.path.join(OUTPUT_DIR, name)

  if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
  if checkpt:
    model.load_weights(model_save_dir + "/model.h5")

  train_images, test_images, test_dataGen = preprocess_Image(train_dir, test_dir, labels=labels, image_size=(150, 150),
                                                             batch_size=50)

  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])


  checkpoint = ModelCheckpoint("'D:\Multi_Class_Image_Classification\Rock-Paper-Scissors\Model\model.h5",
                               monitor="val_loss",
                               mode="min",
                               save_best_only=True,
                               verbose=1)

  earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            restore_best_weights=True)

  reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                factor=learning_rate_decay_factor,
                                patience=3,
                                verbose=1,
                                min_delta=min_lr)

  callbacks = [earlystop, checkpoint, reduce_lr]

  history = model.fit(train_images,
                      epochs=epochs,
                      callbacks=callbacks,
                      validation_data=test_images,
                      )

  model.save(model_save_dir + "/model.h5")

  model.load_weights(model_save_dir + "/model.h5")

  train_images.reset()

  train_loss, train_accuracy = model.evaluate(train_images, steps=(train_images.n // batch_size) + 1, verbose=verbose)
  test_loss, test_accuracy = model.evaluate(test_images, steps=(test_images.n // batch_size) + 1, verbose=verbose)

  model_history(history)

  print("Training :   Loss - {}, Accuracy - {}".format(train_loss, train_accuracy))
  print("Testing :   Loss - {}, Accuracy - {}".format(test_loss, test_accuracy))

  test_images.reset()

  prediction = model.predict(test_images, steps=25 + 1, verbose=verbose)




num_classes = 3
model_shape = (150, 150, 3)
model = build_model(model_shape, num_classes=num_classes)
train_and_evaluate_model(model, name="Multi_label_Classification_model", epochs=25, batch_size=32, verbose=verbose, checkpt=False )