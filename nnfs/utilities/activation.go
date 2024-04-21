package nnfs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ActivationRelu struct {
	Output, Inputs *mat.Dense
	Dinputs        *mat.Dense
	Prev, Next     ILayer
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

func (activation *ActivationRelu) SetPrevious(layer_or_loss ILayer) {
	activation.Prev = layer_or_loss
}

func (activation *ActivationRelu) SetNext(layer_or_loss ILayer) {
	activation.Next = layer_or_loss
}

func (activation *ActivationRelu) GetOutput() *mat.Dense {
	return activation.Output
}

func (activation *ActivationRelu) GetPrevious() ILayer {
	return activation.Prev
}

func (activation *ActivationRelu) GetNext() ILayer {
	return activation.Next
}

func (activation *ActivationRelu) GetDInputs() *mat.Dense {
	return activation.Dinputs
}

type ActivationSoftMax struct {
	Output     *mat.Dense
	Dinputs    *mat.Dense
	Prev, Next ILayer
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

		//had some stability issues, decided to subtract the max as suggested here
		//https://stackoverflow.com/questions/42599498/numerically-stable-softmax
		max_in_row := input.At(i, 0)
		for j := 0; j < activation.Output.RawMatrix().Cols; j++ {
			max_in_row = math.Max(max_in_row, input.At(i, j))
		}
		for j := 0; j < activation.Output.RawMatrix().Cols; j++ {
			soft_maxed[j] = math.Pow(math.E, input.At(i, j)-max_in_row)
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
		single_output_dot := mat.NewDense(single_output.RawMatrix().Cols, single_output.RawMatrix().Cols, nil)
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
		data := []float64{}
		data = append(data, dinputs_temp.RawMatrix().Data...)
		activation.Dinputs.SetRow(i, data)
	}
}

func (activation *ActivationSoftMax) GetDInputs() *mat.Dense {
	return activation.Dinputs
}

func (activation *ActivationSoftMax) GetNext() ILayer {
	return activation.Next
}

func (activation *ActivationSoftMax) GetPrevious() ILayer {
	return activation.Prev
}

func (activation *ActivationSoftMax) SetPrevious(layer_or_loss ILayer) {
	activation.Prev = layer_or_loss
}

func (activation *ActivationSoftMax) SetNext(layer_or_loss ILayer) {
	activation.Next = layer_or_loss
}

func (activation *ActivationSoftMax) GetOutput() *mat.Dense {
	return activation.Output
}

type ActivationSoftMaxLossCategoricalCrossEntropy struct {
	Activation *ActivationSoftMax
	Loss       *Loss_CategoricalCrossentropy
	Dinputs    *mat.Dense
	Prev, Next ILayer
}

func NewActivationSoftMaxLossCategoricalCrossEntropy() *ActivationSoftMaxLossCategoricalCrossEntropy {
	output := &ActivationSoftMaxLossCategoricalCrossEntropy{Activation: NewActivationSoftMax(), Loss: NewLoss_CategoricalCrossentropy()}
	return output
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) Forward(input *mat.Dense) {
	panic("Never call this")
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) ForwardCombined(input *mat.Dense, y_true *mat.Dense) float64 {
	combine.Activation.Forward(input)
	return CalculateLoss(combine.Loss, combine.Activation.Output, y_true)
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) Backward(dvalues *mat.Dense) {
	panic("Never call this")
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) BackwardCombined(dvalues *mat.Dense, y_true *mat.Dense) {
	//y prediction - y true
	//since y true is one hot encoded, we subract 1 or 0
	combine.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	combine.Dinputs.Sub(dvalues, y_true)
	//normalize
	samples := float64(y_true.RawMatrix().Rows)
	combine.Dinputs.Apply(func(r, c int, v float64) float64 {
		return v / samples
	}, combine.Dinputs)
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) GetDInputs() *mat.Dense {
	return combine.Dinputs
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) GetNext() ILayer {
	return combine.Next
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) GetPrevious() ILayer {
	return combine.Prev
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) SetPrevious(layer_or_loss ILayer) {
	combine.Prev = layer_or_loss
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) SetNext(layer_or_loss ILayer) {
	combine.Next = layer_or_loss
}

func (combine *ActivationSoftMaxLossCategoricalCrossEntropy) GetOutput() *mat.Dense {
	return combine.Activation.Output
}

type ActivationSigmoid struct {
	Inputs     *mat.Dense
	Output     *mat.Dense
	Dinputs    *mat.Dense
	Prev, Next ILayer
}

func NewActivationSigmoid() *ActivationSigmoid {
	output := &ActivationSigmoid{}
	return output
}
func (activation *ActivationSigmoid) Forward(input *mat.Dense) {
	activation.Inputs = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, nil)
	activation.Inputs.Copy(input)

	activation.Output = mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, nil)
	activation.Output.Apply(func(r, c int, v float64) float64 {
		exp := 1 + math.Exp((-1 * activation.Inputs.At(r, c)))
		return 1 / exp
	}, activation.Output)
}
func (activation *ActivationSigmoid) Backward(dvalues *mat.Dense) {
	activation.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	activation.Dinputs.Apply(func(r, c int, v float64) float64 {
		output := activation.Output.At(r, c)
		return dvalues.At(r, c) * (1 - output) * output
	}, activation.Dinputs)
}

type ActivationLinear struct {
	Inputs     *mat.Dense
	Output     *mat.Dense
	Dinputs    *mat.Dense
	Prev, Next ILayer
}

func NewActivationLinear() *ActivationLinear {
	output := &ActivationLinear{}
	return output
}

func (activation *ActivationLinear) Forward(input *mat.Dense) {
	activation.Inputs = input
	activation.Output = input
}
func (activation *ActivationLinear) Backward(dvalues *mat.Dense) {
	activation.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	activation.Dinputs.Copy(dvalues)
}

func (activation *ActivationLinear) SetPrevious(layer_or_loss ILayer) {
	activation.Prev = layer_or_loss
}

func (activation *ActivationLinear) SetNext(layer_or_loss ILayer) {
	activation.Next = layer_or_loss
}

func (activation *ActivationLinear) GetOutput() *mat.Dense {
	return activation.Output
}

func (activation *ActivationLinear) GetPrevious() ILayer {
	return activation.Prev
}

func (activation *ActivationLinear) GetNext() ILayer {
	return activation.Next
}

func (activation *ActivationLinear) GetDInputs() *mat.Dense {
	return activation.Dinputs
}
