package nnfs

import (
	"fmt"
	u "nnfs/utilities"
)

func Run1() {
	fmt.Println("Saving model to disk")
	X, y := u.Create_fashion_data(false)
	u.Shuffle(X, y)
	u.Scale_fashion_data(X)
	X_reshape := u.Reshape_fashion_data(X)

	model := u.NewModel()
	model.Accuracy = u.NewAccuracy_Classification()
	model.Add(u.NewLayerDense(28*28, 128))
	model.Add(u.NewActivationRelu())
	model.Add(u.NewLayerDense(128, 128))
	model.Add(u.NewActivationRelu())
	model.Add(u.NewLayerDense(128, 10))
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	model.Add(loss_activation)
	model.Set(loss_activation.Loss, u.NewOptimizerAdam(0.001, 1e-3, 1e-7, 0.9, 0.999))
	model.Finalize()
	model.Train_image(X_reshape, y, 10, 128, 100)
	model.Save("ch20/mnist_model.geo")
}

func Run2() {
	fmt.Println("Loading the model")
	X_val, y_val := u.Create_fashion_data(true)
	X_val_reshape := u.Reshape_fashion_data(X_val)

	model := u.NewModel()
	model.Accuracy = u.NewAccuracy_Classification()
	model.Add(u.NewLayerDense(28*28, 128))
	model.Add(u.NewActivationRelu())
	model.Add(u.NewLayerDense(128, 128))
	model.Add(u.NewActivationRelu())
	model.Add(u.NewLayerDense(128, 10))
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	model.Add(loss_activation)
	model.Set(loss_activation.Loss, u.NewOptimizerAdam(0.001, 1e-3, 1e-7, 0.9, 0.999))
	model.Finalize()
	model.Load("ch20/mnist_model.geo")
	model.Evaluate(X_val_reshape, y_val, len(X_val))
}
