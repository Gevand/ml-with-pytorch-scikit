package nnfs

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type LayerDense struct {
	n_inputs, n_neurons int
	Weights             *mat.Dense
	Biases              *mat.Dense
	Output              *mat.Dense
}

func NewLayerDense(n_inputs int, n_neurons int) *LayerDense {
	output := &LayerDense{n_inputs: n_inputs, n_neurons: n_neurons}
	data := make([]float64, n_inputs*n_neurons)
	for i := range data {
		data[i] = rand.NormFloat64() * 0.01
	}
	output.Weights = mat.NewDense(n_inputs, n_neurons, data)
	output.Biases = mat.NewDense(1, n_neurons, nil)
	return output
}

func (layer *LayerDense) Forward(input *mat.Dense) {
	layer.Output = mat.NewDense(input.RawMatrix().Rows, layer.n_neurons, nil)

	var rawBias = make([]float64, input.RawMatrix().Rows*layer.n_neurons)
	for i := range rawBias {
		rawBias[i] = layer.Biases.At(0, i%layer.n_neurons)
	}
	var biases1 = mat.NewDense(input.RawMatrix().Rows, layer.n_neurons, rawBias)

	layer.Output.Mul(input, layer.Weights)
	layer.Output.Add(layer.Output, biases1)
}
