package nnfs

import (
	"fmt"
	u "nnfs/utilities"
)

func Run1() {
	fmt.Println("Building a model object")
	X, y := u.Create_data(10)
	model := u.NewModel()
	model.Accuracy = u.NewAccuracy_Regression(0, true, y)
	model.Add(u.NewLayerDense(1, 64))
	model.Add(u.NewActivationRelu())
	model.Add(u.NewLayerDense(64, 64))
	model.Add(u.NewActivationRelu())
	model.Add(u.NewLayerDense(64, 1))
	model.Add(u.NewActivationLinear())
	model.Set(u.NewLoss_MSE(), u.NewOptimizerAdam(0.005, 1e-3, 1e-7, 0.9, 0.999))
	model.Finalize()
	model.Train(X, y, 10000, 100)
	fmt.Println(model.Layers)
}
