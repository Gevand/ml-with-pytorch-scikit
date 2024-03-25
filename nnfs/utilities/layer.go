package nnfs

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Forward(input *mat.Dense)
	Backward(dvalues *mat.Dense)
}

type LayerDense struct {
	n_inputs, n_neurons                                                            int
	Dweights, Dbiases, Dinputs                                                     *mat.Dense
	Weights, Weight_Momentums, Weight_Cache                                        *mat.Dense
	Biases, Bias_Momentums, Bias_Cache                                             *mat.Dense
	Output                                                                         *mat.Dense
	Inputs                                                                         *mat.Dense
	Name                                                                           string
	Weight_Regulizer_L1, Weight_Regulizer_L2, Bias_Regulizer_L1, Bias_Regulizer_L2 float64 //used for regulazition
}

func NewLayerDense(n_inputs int, n_neurons int) *LayerDense {
	output := &LayerDense{n_inputs: n_inputs, n_neurons: n_neurons}
	data := make([]float64, n_inputs*n_neurons)
	for i := range data {
		data[i] = rand.NormFloat64() * 0.01
	}
	output.Weights = mat.NewDense(n_inputs, n_neurons, data)
	output.Biases = mat.NewDense(1, n_neurons, nil)
	//no regularization by default
	output.Weight_Regulizer_L1, output.Weight_Regulizer_L2, output.Bias_Regulizer_L1, output.Bias_Regulizer_L2 = 0.0, 0.0, 0.0, 0.0
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
	if layer.Weight_Regulizer_L1 > 0 {
		//derivative of l1 is either 1 or -1, depending of the slope of the absolute value
		layer.Dweights.Apply(func(r, c int, v float64) float64 {
			if layer.Weight_Regulizer_L1 >= 0 {
				return v + (layer.Weights.At(r, c) * 1)
			}
			return v + (layer.Weights.At(r, c) * -1)
		}, layer.Dweights)
	}
	if layer.Weight_Regulizer_L2 > 0 {
		//derivative of l2 is 2 * weight_regulizer_l2 * Wij
		//we add it to the current dweights
		layer.Dweights.Apply(func(r, c int, v float64) float64 {
			return v + (layer.Weights.At(r, c) * 2 * layer.Weight_Regulizer_L2)
		}, layer.Dweights)
	}

	layer.Dbiases = SumAxis0KeepDimsTrue(dvalues)
	if layer.Bias_Regulizer_L1 > 0 {
		layer.Dbiases.Apply(func(r, c int, v float64) float64 {
			if layer.Bias_Regulizer_L1 >= 0 {
				return v + (layer.Biases.At(r, c) * 1)
			}
			return v + (layer.Biases.At(r, c) * -1)
		}, layer.Dbiases)
	}
	if layer.Bias_Regulizer_L2 > 0 {
		layer.Dbiases.Apply(func(r, c int, v float64) float64 {
			return v + (layer.Biases.At(r, c) * 2 * layer.Bias_Regulizer_L2)
		}, layer.Dbiases)
	}

	layer.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, layer.Weights.RawMatrix().Rows, nil)
	layer.Dinputs.Mul(dvalues, layer.Weights.T())
}

type LayerDropout struct {
	rate                                float64
	BinaryMask, Output, Inputs, Dinputs *mat.Dense
}

func (layer *LayerDropout) Forward(input *mat.Dense) {
	layer.Inputs = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, nil)
	layer.Inputs.Copy(input)

	layer.BinaryMask = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, Binomial(1, layer.rate, input.RawMatrix().Rows*input.RawMatrix().Cols))
	//scale the binary mask
	layer.BinaryMask.Apply(func(r, c int, v float64) float64 {
		return v / (1.0 - layer.rate)
	}, layer.BinaryMask)

	layer.Output = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, nil)
	layer.Output.MulElem(layer.Inputs, layer.BinaryMask)
}

func (layer *LayerDropout) Backward(dvalues *mat.Dense) {
	layer.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	layer.Dinputs.MulElem(dvalues, layer.BinaryMask)
}

func NewLayerDropout(rate float64) *LayerDropout {
	output := &LayerDropout{rate: rate}
	return output
}
