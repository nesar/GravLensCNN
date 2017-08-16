from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam, Adadelta


def basic_model(num_classes = 2, learning_rate = 0.01, decay_rate = 0.0, opti_id = 0,
                        loss_id = 0, image_size = 45):

	# input_shape=img_data[0].shape

	model = Sequential()

	model.add(Convolution2D(32, 3, 3,border_mode='same',input_shape=(image_size, image_size, 1)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])

	if opti_id == 0:
		sgd = SGD(lr=learning_rate, decay=decay_rate)
		# lr = 0.01, momentum = 0., decay = 0., nesterov = False
		model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
	elif opti_id == 1:
		adam = Adam(lr=learning_rate, decay=decay_rate)
		#adam = adam()
		# lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
		model.compile(loss= 'categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
	else:
		rmsprop = RMSprop(lr=learning_rate, decay=decay_rate)
		#rmsprop = RMSprop()
		# lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.
		model.compile(loss= 'categorical_crossentropy', optimizer=rmsprop, metrics=["accuracy"])

	return model
