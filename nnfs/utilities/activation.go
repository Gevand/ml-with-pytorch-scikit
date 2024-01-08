package nnfs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ActivationRelu struct {
	Output, Inputs *mat.Dense
	Dinputs        *mat.Dense
}

func NewActivationRelu() *ActivationRelu {
	output := &ActivationRelu{}
	return output
}

func (activation *ActivationRelu) Forward(input *mat.Dense) {
	activation.Inputs = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, nil)
	activation.Inputs.Copy(input)

	raw := input.RawMatrix().Data
	for i := range raw {
		if raw[i] < 0 {
			raw[i] = 0
		}
	}
	activation.Output = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, raw)
}

func (activation *ActivationRelu) Backward(dvalues *mat.Dense) {
	activation.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	activation.Dinputs.Copy(dvalues)

	activation.Dinputs.Apply(func(r, c int, v float64) float64 {
		if activation.Inputs.At(r, c) <= 0 {
			return 0
		}
		return v
	}, activation.Dinputs)
}

type ActivationSoftMax struct {
	Output  *mat.Dense
	Dinputs *mat.Dense
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

func (activation *ActivationSoftMax) Backward(dvalues *mat.Dense) {
	activation.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	activation.Dinputs.Copy(dvalues)

	/*
		Python code I'm trying to copy
		for ​index, (single_output, single_dvalues) ​in ​​enumerate​(​zip​(self.output, dvalues)):
			​# Flatten output array
			​single_output ​= ​single_output.reshape(​-​1​, ​1​)
			​# Calculate Jacobian matrix of the output and
			​jacobian_matrix ​= ​np.diagflat(single_output) ​- ​np.dot(single_output, single_output.T)
			​# Calculate sample-wise gradient
			# and add it to the array of sample gradients
			​self.dinputs[index] ​= ​np.dot(jacobian_matrix,
			single_dvalues)
	*/
	for i := 0; i < activation.Output.RawMatrix().Rows; i++ {
		single_dvalues := mat.NewDense(1, activation.Output.RawMatrix().Cols, dvalues.RawRowView(i))
		single_output := mat.NewDense(1, activation.Output.RawMatrix().Cols, activation.Output.RawRowView(i))
		single_output_dot := mat.NewDense(3, 3, nil)
		single_output_dot.Mul(single_output.T(), single_output)

		//intialized as a diagnal matrix with the single_output being the diagnal
		jacobian_matrix := mat.NewDense(activation.Output.RawMatrix().Cols, activation.Output.RawMatrix().Cols, nil)
		jacobian_matrix.Apply(func(r, c int, v float64) float64 {
			if r != c {
				return 0 - single_output_dot.At(r, c)
			}
			return single_output.At(0, r) - single_output_dot.At(r, c)
		}, jacobian_matrix)
		dinputs_temp := mat.NewDense(jacobian_matrix.RawMatrix().Cols, single_dvalues.RawMatrix().Rows, nil)
		dinputs_temp.Mul(jacobian_matrix, single_dvalues.T())
		activation.Dinputs.SetRow(i, dinputs_temp.RawMatrix().Data)
	}
}
