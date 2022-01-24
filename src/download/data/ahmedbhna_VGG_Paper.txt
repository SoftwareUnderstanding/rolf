# VGG_Paper
Re implement oxford paper " VERY DEEP CONVOLUTIONAL Networks for large -scale image recognition " paper
https://arxiv.org/pdf/1409.1556.pdf

to Train the model use the command :

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
cnn_n = base_model()
cnn_n.summary()

to fit the model :
cnn = cnn_n.fit(x_train,y_train, batch_size=batch_size, epochs=epochs,validation_data=(x_test,y_test),shuffle=True)
