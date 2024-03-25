package nnfs

import "gonum.org/v1/gonum/mat"

type Model struct {
	Layers    []Layer
	Loss      Loss
	Optimizer Optimizer
}

func NewModel() *Model {
	return &Model{Layers: []Layer{}}
}

func (model *Model) Add(layer Layer) {
	model.Layers = append(model.Layers, layer)
}

func (model *Model) Set(loss Loss, optimizer Optimizer) {
	model.Loss = loss
	model.Optimizer = optimizer
}

func (model *Model) Train(X, y *mat.Dense, epochs, print_every int) {
	if print_every <= 0 {
		panic("print_every can't be 0 or negative, make it 1 bro")
	}
	for i := 0; i < epochs; i++ {
		//TODO: for now do nothing
	}
}
