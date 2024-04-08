package nnfs

import (
	"gonum.org/v1/gonum/mat"
)

type Model struct {
	Layers     []ILayer
	InputLayer ILayer
	Loss       ILoss
	Optimizer  IOptimizer
}

func NewModel() *Model {
	return &Model{Layers: []ILayer{}}
}

func (model *Model) Add(layer ILayer) {
	model.Layers = append(model.Layers, layer)
}

func (model *Model) Set(loss ILoss, optimizer IOptimizer) {
	model.Loss = loss
	model.Optimizer = optimizer
}

func (model *Model) Train(X, y *mat.Dense, epochs, print_every int) {
	if print_every <= 0 {
		panic("print_every can't be 0 or negative, make it 1 bro")
	}
	for i := 1; i <= epochs; i++ {
		output := model.Forward(X)
		model.Backward(output, y)

		model.Optimizer.PreUpdateParams()

		for _, layer := range model.Layers {
			//TODO: make this better
			switch v := layer.(type) {
			case *LayerDense:
				model.Optimizer.UpdateParameters(v)
			}
		}
		//TODO: Handle print every stuff

		model.Optimizer.PostUpdateParams()
	}
}

func (model *Model) Forward(X *mat.Dense) *mat.Dense {

	model.InputLayer.Forward(X)

	for _, layer := range model.Layers {
		layer.Forward(layer.GetPrevious().GetOutput())
	}

	return model.Layers[len(model.Layers)-1].GetOutput()
}

func (model *Model) Backward(output, y *mat.Dense) {
	CalculateLoss(model.Loss, output, y)
	model.Loss.Backward(output, y)
	for i := len(model.Layers) - 1; i >= 0; i-- {
		layer := model.Layers[i]
		if i == len(model.Layers)-1 {
			layer.Backward(model.Loss.GetDInputs())
		} else {
			layer.Backward(layer.GetNext().GetDInputs())
		}
	}
}

func (model *Model) Finalize() {
	model.InputLayer = NewInputLayer()
	layer_count := len(model.Layers)
	for index, layer := range model.Layers {
		if index == 0 {
			layer.SetPrevious(model.InputLayer)
			layer.SetNext(model.Layers[index+1])
		} else if index < layer_count-1 {
			layer.SetPrevious(model.Layers[index-1])
			layer.SetNext(model.Layers[index+1])
		} else {
			layer.SetPrevious(model.Layers[index-1])
			layer.SetNext(nil)
		}
	}
}
