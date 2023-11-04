package nnfs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ActivationRelu struct {
	Output *mat.Dense
}

func NewActivationRelu() *ActivationRelu {
	output := &ActivationRelu{}
	return output
}

func (activation *ActivationRelu) Forward(input *mat.Dense) {
	raw := input.RawMatrix().Data
	for i := range raw {
		if raw[i] < 0 {
			raw[i] = 0
		}
	}
	activation.Output = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, raw)
}

type ActivationSoftMax struct {
	Output *mat.Dense
}

func NewActivationSoftMax() *ActivationSoftMax {
	output := &ActivationSoftMax{}
	return output
}
func (activation *ActivationSoftMax) Forward(input *mat.Dense) {
	activation.Output = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, nil)
	activation.Output.Copy(input)
	for i := 0; i < activation.Output.RawMatrix().Rows; i++ {
		var soft_maxed = make([]float64, activation.Output.RawMatrix().Cols)
		soft_maxed_sum := 0.0
		for j := 0; j < activation.Output.RawMatrix().Cols; j++ {
			soft_maxed[j] = math.Pow(math.E, input.At(i, j))
			soft_maxed_sum += soft_maxed[j]
		}
		for i := range soft_maxed {
			soft_maxed[i] = soft_maxed[i] / soft_maxed_sum
		}
		activation.Output.SetRow(i, soft_maxed)
	}
}
