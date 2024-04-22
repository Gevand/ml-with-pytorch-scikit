package nnfs

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type IOptimizer interface {
	PreUpdateParams()
	UpdateParameters(layer *LayerDense)
	PostUpdateParams()
	GetLearningRate() float64
}

type OptimizerSGD struct {
	LearningRate, Decay, Momentum float64 //hyper parameters
	Iterations                    int
	CurrentLearningRate           float64
}

func NewOptimizerSGD(lr, decay, momentum float64) *OptimizerSGD {
	output := &OptimizerSGD{LearningRate: lr, Decay: decay, CurrentLearningRate: lr, Momentum: momentum}
	return output
}

func (optimizer *OptimizerSGD) GetLearningRate() float64 { return optimizer.CurrentLearningRate }

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
func (optimizer *OptimizerAdaGrad) GetLearningRate() float64 { return optimizer.CurrentLearningRate }
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
func (optimizer *OptimizerRmsProp) GetLearningRate() float64 { return optimizer.CurrentLearningRate }

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

type OptimizerAdam struct {
	LearningRate, Decay, Epsilon, Beta1, Beta2 float64 //hyper parameters
	Iterations                                 int
	CurrentLearningRate                        float64
}

func NewOptimizerAdam(lr, decay, eps, beta1, beta2 float64) *OptimizerAdam {
	output := &OptimizerAdam{LearningRate: lr, Decay: decay, CurrentLearningRate: lr, Epsilon: eps, Beta1: beta1, Beta2: beta2}
	return output
}

func (optimizer *OptimizerAdam) PreUpdateParams() {
	if optimizer.Decay > 0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate * (1. / (1 + optimizer.Decay*float64(optimizer.Iterations)))
	}
}

func (optimizer *OptimizerAdam) GetLearningRate() float64 { return optimizer.CurrentLearningRate }

func (optimizer *OptimizerAdam) UpdateParameters(layer *LayerDense) {
	if layer.Weight_Cache == nil {
		layer.Weight_Cache = mat.NewDense(layer.Weights.RawMatrix().Rows, layer.Weights.RawMatrix().Cols, nil)
		layer.Bias_Cache = mat.NewDense(layer.Biases.RawMatrix().Rows, layer.Biases.RawMatrix().Cols, nil)
		layer.Bias_Momentums = mat.NewDense(layer.Biases.RawMatrix().Rows, layer.Biases.RawMatrix().Cols, nil)
		layer.Weight_Momentums = mat.NewDense(layer.Weights.RawMatrix().Rows, layer.Weights.RawMatrix().Cols, nil)
	}

	layer.Weight_Momentums.Apply(func(r, c int, v float64) float64 {
		return optimizer.Beta1*v + ((1 - optimizer.Beta1) * layer.Dweights.At(r, c))
	}, layer.Weight_Momentums)

	layer.Bias_Momentums.Apply(func(r, c int, v float64) float64 {
		bias_momentum := optimizer.Beta1*v + ((1 - optimizer.Beta1) * layer.Dbiases.At(r, c))
		return bias_momentum
	}, layer.Bias_Momentums)

	weight_momentums_corrected := mat.NewDense(layer.Weight_Momentums.RawMatrix().Rows, layer.Weight_Momentums.RawMatrix().Cols, nil)
	correction := 1 - math.Pow(optimizer.Beta1, float64(optimizer.Iterations+1))
	weight_momentums_corrected.Apply(func(r, c int, v float64) float64 {
		return layer.Weight_Momentums.At(r, c) / correction
	}, weight_momentums_corrected)

	bias_momentums_corrected := mat.NewDense(layer.Bias_Momentums.RawMatrix().Rows, layer.Bias_Momentums.RawMatrix().Cols, nil)
	bias_momentums_corrected.Apply(func(r, c int, v float64) float64 {
		return layer.Bias_Momentums.At(r, c) / correction
	}, bias_momentums_corrected)

	//beta2 is Rho from RmsProp
	layer.Weight_Cache.Apply(func(r, c int, v float64) float64 {
		rho_calc := optimizer.Beta2 * v
		dweight_calc := (1 - optimizer.Beta2) * math.Pow(layer.Dweights.At(r, c), 2)
		return rho_calc + dweight_calc
	}, layer.Weight_Cache)
	layer.Bias_Cache.Apply(func(r, c int, v float64) float64 {
		rho_calc := optimizer.Beta2 * v
		dweight_calc := (1 - optimizer.Beta2) * math.Pow(layer.Dbiases.At(r, c), 2)
		return rho_calc + dweight_calc
	}, layer.Bias_Cache)

	//correct the cache
	weight_cache_corrected := mat.NewDense(layer.Weight_Cache.RawMatrix().Rows, layer.Weight_Cache.RawMatrix().Cols, nil)
	bias_cache_corrected := mat.NewDense(layer.Bias_Cache.RawMatrix().Rows, layer.Bias_Cache.RawMatrix().Cols, nil)

	correction_2 := 1 - math.Pow(optimizer.Beta2, float64(optimizer.Iterations+1))
	weight_cache_corrected.Apply(func(r, c int, v float64) float64 {
		return layer.Weight_Cache.At(r, c) / correction_2
	}, weight_cache_corrected)
	bias_cache_corrected.Apply(func(r, c int, v float64) float64 {
		return layer.Bias_Cache.At(r, c) / correction_2
	}, bias_cache_corrected)

	layer.Weights.Apply(func(r, c int, v float64) float64 {
		sgd := (-1 * optimizer.CurrentLearningRate * weight_momentums_corrected.At(r, c))
		sqrt := math.Sqrt(weight_cache_corrected.At(r, c)) + optimizer.Epsilon
		return v + (sgd / sqrt)
	}, layer.Weights)

	// layer.biases += -self.current_learning_rate * \
	//         bias_momentums_corrected / \
	//         (np.sqrt(bias_cache_corrected) +
	//          self.epsilon)

	layer.Biases.Apply(func(r, c int, v float64) float64 {

		sgd := (-1 * optimizer.CurrentLearningRate * bias_momentums_corrected.At(r, c))
		sqrt := math.Sqrt(bias_cache_corrected.At(r, c)) + optimizer.Epsilon
		return v + (sgd / sqrt)
	}, layer.Biases)
	if layer.Biases.At(0, 0) != 0 {
		fmt.Println("Bias", layer.Biases)
		fmt.Println("DBiases", layer.Dbiases)
		fmt.Println("Bias Momentum", layer.Bias_Momentums)
		fmt.Println("Bias Momentum Corrected", bias_momentums_corrected)
	}
}

func (optimizer *OptimizerAdam) PostUpdateParams() {
	optimizer.Iterations++
}
