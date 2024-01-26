package nnfs

import "gonum.org/v1/gonum/mat"

type OptimizerSGD struct {
	LearningRate, Decay, Momentum float64 //hyper parameters
	Iterations                    int
	CurrentLearningRate           float64
}

func NewOptimizerSGD(lr, decay, momentum float64) *OptimizerSGD {
	output := &OptimizerSGD{LearningRate: lr, Decay: decay, CurrentLearningRate: lr, Momentum: momentum}
	return output
}

func (optimizer *OptimizerSGD) PreUpdateParams() {
	if optimizer.Decay > 0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1. / (1 + optimizer.Decay*float64(optimizer.Iterations)))
	}
}
func (optimizer *OptimizerSGD) UpdateParameters(layer *LayerDense) {
	if optimizer.Momentum > 0 {
		if layer.Weight_Momentums == nil {
			layer.Weight_Momentums = mat.NewDense(layer.Weights.RawMatrix().Rows, layer.Weights.RawMatrix().Cols, nil)
		}
		if layer.Bias_Momentums == nil {
			layer.Bias_Momentums = mat.NewDense(layer.Biases.RawMatrix().Rows, layer.Biases.RawMatrix().Cols, nil)
		}

		//save the momentum for the next run
		layer.Weight_Momentums.Apply(func(r, c int, v float64) float64 {
			return (v * optimizer.Momentum) - (optimizer.CurrentLearningRate * layer.Dweights.At(r, c))
		}, layer.Weight_Momentums)
		layer.Bias_Momentums.Apply(func(r, c int, v float64) float64 {
			return (v * optimizer.Momentum) - (optimizer.CurrentLearningRate * layer.Dbiases.At(r, c))
		}, layer.Bias_Momentums)

		//weight update is simply weights + momentum
		layer.Weights.Apply(func(r, c int, v float64) float64 {
			return v + layer.Weight_Momentums.At(r, c)
		}, layer.Weights)
		layer.Biases.Apply(func(r, c int, v float64) float64 {
			return v + layer.Bias_Momentums.At(r, c)
		}, layer.Biases)
	} else {
		layer.Weights.Apply(func(r, c int, v float64) float64 {
			return v + (-1 * optimizer.CurrentLearningRate * layer.Dweights.At(r, c))
		}, layer.Weights)

		layer.Biases.Apply(func(r, c int, v float64) float64 {
			return v + (-1 * optimizer.CurrentLearningRate * layer.Dbiases.At(r, c))
		}, layer.Biases)
	}

}
func (optimizer *OptimizerSGD) PostUpdateParams() {
	optimizer.Iterations++
}
