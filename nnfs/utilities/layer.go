package nnfs

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type LayerDense struct {
	n_inputs, n_neurons                     int
	Dweights, Dbiases, Dinputs              *mat.Dense
	Weights, Weight_Momentums, Weight_Cache *mat.Dense
	Biases, Bias_Momentums, Bias_Cache      *mat.Dense
	Output                                  *mat.Dense
	Inputs                                  *mat.Dense
	Name                                    string
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
	layer.Inputs = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, nil)
	layer.Inputs.Copy(input)

	layer.Output = mat.NewDense(input.RawMatrix().Rows, layer.n_neurons, nil)

	layer.Output.Mul(input, layer.Weights)
	layer.Output.Apply(func(r, c int, v float64) float64 {
		return v + layer.Biases.At(0, c)
	}, layer.Output)
}

func (layer *LayerDense) Backward(dvalues *mat.Dense) {
	layer.Dweights = mat.NewDense(layer.Inputs.RawMatrix().Cols, dvalues.RawMatrix().Cols, nil)
	layer.Dweights.Mul(layer.Inputs.T(), dvalues)

	layer.Dbiases = SumAxis0KeepDimsTrue(dvalues)
	layer.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, layer.Weights.RawMatrix().Rows, nil)
	layer.Dinputs.Mul(dvalues, layer.Weights.T())
}
