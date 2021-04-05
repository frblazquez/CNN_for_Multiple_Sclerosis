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
    
	
	
	
for i in range(54, options['max_epochs']):
	eddl.reset_loss(model[0])
	g = generator(X, Y, options['batch_size'], num_batches)
	print("Epoch %d/%d (%d batches)" % (i + 1, options['max_epochs'], num_batches))
	epoch_time = time.time()
	for j in range(num_batches):
		d = next(g)
		start_batch = time.time()
		eddl.train_batch(model[0], [Tensor.fromarray(d[0])], [Tensor.fromarray(d[1])])
		eddl.print_loss(model[0], j)
		print(f'Batch time: {time.time()-start_batch:.2f}')
	print(f'Epoch {i} time: {time.time()-epoch_time:.2f}')

