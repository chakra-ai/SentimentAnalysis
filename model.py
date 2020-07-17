from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

# define the model
def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(30, input_shape=(n_words,), kernel_regularizer = l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(10, input_shape=(n_words,), kernel_regularizer = l2(0.001)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    
    return model