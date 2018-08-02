from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import  sys

MAX_SEQUENCE_LENGTH=70
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2

text = np.load("data/text/all_text_2.npy")
text = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)

label = np.load("data/text/all_label_1.npy")
label = to_categorical(label)


# split the data into training set, validation set, and test set
p1 = int(len(text)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(text)*(1-TEST_SPLIT))
x_train = text[:p1]
y_train = label[:p1]
x_val = text[p1:p2]
y_val = label[p1:p2]
x_test = text[p2:]
y_test = label[p2:]
print('train docs: '+str(len(x_train)))
print('val docs: '+str(len(x_val)))
print('test docs: '+str(len(x_test)))

model = Sequential()
model.add(Dense(200, activation='relu',input_shape=(x_train.shape[1],200)))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(label.shape[1], activation='softmax'))
model.summary()
#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print(model.metrics_names)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
# model.save('cnn.h5')

print('(6) testing model...')
print(model.evaluate(x_test, y_test))



