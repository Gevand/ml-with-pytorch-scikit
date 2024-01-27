package nnfs

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

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

type OptimizerAdaGrad struct {
	LearningRate, Decay, Epsilon float64 //hyper parameters
	Iterations                   int
	CurrentLearningRate          float64
}

func NewOptimizerAdaGrad(lr, decay, eps float64) *OptimizerAdaGrad {
	output := &OptimizerAdaGrad{LearningRate: lr, Decay: decay, CurrentLearningRate: lr, Epsilon: eps}
	return output
}

func (optimizer *OptimizerAdaGrad) PreUpdateParams() {
	if optimizer.Decay > 0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1. / (1 + optimizer.Decay*float64(optimizer.Iterations)))
	}
}
func (optimizer *OptimizerAdaGrad) UpdateParameters(layer *LayerDense) {
	if layer.Weight_Cache == nil {
		layer.Weight_Cache = mat.NewDense(layer.Weights.RawMatrix().Rows, layer.Weights.RawMatrix().Cols, nil)
		layer.Bias_Cache = mat.NewDense(layer.Biases.RawMatrix().Rows, layer.Biases.RawMatrix().Cols, nil)
	}

	//cache is the square of all the gradients added tot he previous iteration
	layer.Weight_Cache.Apply(func(r, c int, v float64) float64 {
		return v + math.Pow(layer.Dweights.At(r, c), 2)
	}, layer.Weight_Cache)
	layer.Bias_Cache.Apply(func(r, c int, v float64) float64 {
		return v + math.Pow(layer.Dbiases.At(r, c), 2)
	}, layer.Bias_Cache)

	//update is the normal sgd, but you also divide by the sqrt of the cache + epsilon
	//this is not "nothing" for example 1 + 3 = 4; math.sqrt(1^2 + 3^2) = math.sqrt(10) = 2.16
	layer.Weights.Apply(func(r, c int, v float64) float64 {
		if v == math.NaN() || layer.Dweights.At(r, c) == math.NaN() {
			fmt.Println(optimizer.Iterations)
		}
		sgd := (-1 * optimizer.CurrentLearningRate * layer.Dweights.At(r, c))
		ada := math.Sqrt(layer.Weight_Cache.At(r, c)) + optimizer.Epsilon
		return v + (sgd / ada)
	}, layer.Weights)

	layer.Biases.Apply(func(r, c int, v float64) float64 {
		sgd := (-1 * optimizer.CurrentLearningRate * layer.Dbiases.At(r, c))
		ada := math.Sqrt(layer.Bias_Cache.At(r, c)) + optimizer.Epsilon
		return v + (sgd / ada)
	}, layer.Biases)
}

func (optimizer *OptimizerAdaGrad) PostUpdateParams() {
	optimizer.Iterations++
}

type OptimizerRmsProp struct {
	LearningRate, Decay, Epsilon, Rho float64 //hyper parameters
	Iterations                        int
	CurrentLearningRate               float64
}

func NewOptimizerRmsProp(lr, decay, eps, rho float64) *OptimizerRmsProp {
	output := &OptimizerRmsProp{LearningRate: lr, Decay: decay, CurrentLearningRate: lr, Epsilon: eps, Rho: rho}
	return output
}

func (optimizer *OptimizerRmsProp) PreUpdateParams() {
	if optimizer.Decay > 0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1. / (1 + optimizer.Decay*float64(optimizer.Iterations)))
	}
}
func (optimizer *OptimizerRmsProp) UpdateParameters(layer *LayerDense) {
	if layer.Weight_Cache == nil {
		layer.Weight_Cache = mat.NewDense(layer.Weights.RawMatrix().Rows, layer.Weights.RawMatrix().Cols, nil)
		layer.Bias_Cache = mat.NewDense(layer.Biases.RawMatrix().Rows, layer.Biases.RawMatrix().Cols, nil)
	}

	//cache is the square of all the gradients added tot he previous iteration
	layer.Weight_Cache.Apply(func(r, c int, v float64) float64 {
		rho_calc := optimizer.Rho * v
		dweight_calc := (1 - optimizer.Rho) * math.Pow(layer.Dweights.At(r, c), 2)
		return rho_calc + dweight_calc
	}, layer.Weight_Cache)
	layer.Bias_Cache.Apply(func(r, c int, v float64) float64 {
		rho_calc := optimizer.Rho * v
		dweight_calc := (1 - optimizer.Rho) * math.Pow(layer.Dbiases.At(r, c), 2)
		return rho_calc + dweight_calc
	}, layer.Bias_Cache)

	layer.Weights.Apply(func(r, c int, v float64) float64 {
		if v == math.NaN() || layer.Dweights.At(r, c) == math.NaN() {
			fmt.Println(optimizer.Iterations)
		}
		sgd := (-1 * optimizer.CurrentLearningRate * layer.Dweights.At(r, c))
		sqrt := math.Sqrt(layer.Weight_Cache.At(r, c)) + optimizer.Epsilon
		return v + (sgd / sqrt)
	}, layer.Weights)

	layer.Biases.Apply(func(r, c int, v float64) float64 {
		sgd := (-1 * optimizer.CurrentLearningRate * layer.Dbiases.At(r, c))
		sqrt := math.Sqrt(layer.Bias_Cache.At(r, c)) + optimizer.Epsilon
		return v + (sgd / sqrt)
	}, layer.Biases)
}

func (optimizer *OptimizerRmsProp) PostUpdateParams() {
	optimizer.Iterations++
}
