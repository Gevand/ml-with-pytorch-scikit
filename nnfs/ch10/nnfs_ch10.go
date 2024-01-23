package nnfs

import (
	"fmt"
	u "nnfs/utilities"

	"gonum.org/v1/gonum/mat"
)

func Run1() {
	fmt.Println("Using SGD")
	X, y := u.Create_spiral_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 64)
	activation_1 := u.NewActivationRelu()
	dense_2 := u.NewLayerDense(64, 3)
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	optimizer := u.NewOptimizerSGD(0.05, 0.001)

	max_epoch := 10001
	for i := 0; i < max_epoch; i++ {
		if i == 7 {
			fmt.Print("Problem epoch")
		}
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)

		loss := loss_activation.Forward(dense_2.Output, y_one_hot)

		if i%100 == 0 {
			fmt.Println("epoch", i, "loss -->", loss)
			accuracy := u.Accuracy(loss_activation.Activation.Output, y_one_hot)
			fmt.Println("epoch", i, "acc -->", accuracy)
		}

		//backward pass
		loss_activation.Backward(loss_activation.Activation.Output, y_one_hot)
		dense_2.Backward(loss_activation.Dinputs)
		activation_1.Backward(dense_2.Dinputs)
		dense_1.Backward(activation_1.Dinputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParameters(dense_1)
		optimizer.UpdateParameters(dense_2)
		optimizer.PostUpdateParams()
	}
}
