# Francisco Javier Blázquez Martínez
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:
# Python generator template

def generator(X_data, y_data, minibatch_size, number_of_batches):

    counter=0

    while 1:
        X_batch = np.array(X_data[minibatch_size*counter:minibatch_size*(counter+1)]).astype('float32')
        y_batch = np.array(y_data[minibatch_size*counter:minibatch_size*(counter+1)]).astype('float32')
        counter += 1
        yield X_batch,y_batch

    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0
