package nnfs

type OptimizerSGD struct {
	LearningRate, Decay, CurrentLearningRate float64
	Iterations                               int
}

func NewOptimizerSGD(lr float64, decay float64) *OptimizerSGD {
	output := &OptimizerSGD{LearningRate: lr, Decay: decay, CurrentLearningRate: lr}
	return output
}

func (optimizer *OptimizerSGD) PreUpdateParams() {
	if optimizer.Decay > 0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1. / (1 + optimizer.Decay*float64(optimizer.Iterations)))
	}
}
func (optimizer *OptimizerSGD) UpdateParameters(layer *LayerDense) {
	layer.Weights.Apply(func(r, c int, v float64) float64 {
		return v + (-1 * optimizer.CurrentLearningRate * layer.Dweights.At(r, c))
	}, layer.Weights)

	layer.Biases.Apply(func(r, c int, v float64) float64 {
		return v + (-1 * optimizer.CurrentLearningRate * layer.Dbiases.At(r, c))
	}, layer.Biases)
}
func (optimizer *OptimizerSGD) PostUpdateParams() {
	optimizer.Iterations++
}
