from keras import layers
from keras import models
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


def multiple_choice():
    """Choose the right answers (one or more) from multiple choices.
    Note: if you think a is the right answer, return a;
            if your think a, b are the right answers, return a, b; ect.
    """
    # Here are 4 choices
    a = "Training accuracy increases over time and this is a problem."
    b = "Validation accuracy does not increase over time and this is a problem."
    c = "Training loss decreases over time and this is a problem."
    d = "Validation loss does not decrease over time and this is a problem."
    return a, b, c, d


def modified_cnn():
    """Return instance of keras.models.Sequential
    This is similar to build_cnn_architecture() in q1.py; however, you have to compile your model with
    a proper loss function and optimizer.
    """
    #######
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='sigmoid',input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='sigmoid'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='sigmoid'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='sigmoid'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(4, activation='softmax'))
    
    #######
    return model


def test_multiple_choice():
    choice = multiple_choice()
    assert isinstance(choice, str) or isinstance(choice, tuple)
    print("\nPass.")


def test_modified_cnn():
    model = modified_cnn()
    assert isinstance(model, models.Sequential)
    if hasattr(model, "loss") and hasattr(model, "optimizer"):
        print("\nPass.")


if __name__ == '__main__':
    test_multiple_choice()
    test_modified_cnn()
    
train_dir="../Datasets/cat_dog_car_bike/train"
val_dir="../Datasets/cat_dog_car_bike/val"

model = modified_cnn()
model.compile(loss='sparse_categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=0.1),
            metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255) #rescale the tensor values to [0,1]
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20) #we only have two classes

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=20)

history = model.fit(train_generator, epochs=10, validation_data = validation_generator)




acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
"""





    
print ("start..")

history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=50)



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

"""
