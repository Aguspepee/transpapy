import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# load the train and test datasets
train_data = []
train_labels = []
with open("train.txt", "r") as f:
    for line in f:
        sentence, label = line.strip().split(",")
        train_data.append(sentence)
        train_labels.append(int(label))

test_data = []
test_labels = []
with open("test.txt", "r") as f:
    for line in f:
        sentence, label = line.strip().split(",")
        test_data.append(sentence)
        test_labels.append(int(label))

# convert the data to numerical tensors
tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data)

x_train = tokenizer.texts_to_sequences(train_data)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=500)
y_train = train_labels

x_test = tokenizer.texts_to_sequences(test_data)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=500)
y_test = test_labels

# build the model
model = keras.Sequential()
model.add(layers.Embedding(10000, 16))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(5, activation="softmax"))

# compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# train the model
model.fit(x_train, y_train, epochs=5, batch_size=512)

# evaluate the model
_, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: {:.2f}".format(accuracy))
