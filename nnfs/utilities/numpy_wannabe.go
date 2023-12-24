package nnfs

import "gonum.org/v1/gonum/mat"

func Linspace(start, end float64, num int) []float64 {
	if num < 0 {
		panic("number of samples, %d, must be non-negative.")
	}
	result := make([]float64, num)
	step := (end - start) / float64(num-1)
	for i := range result {
		result[i] = start + float64(i)*step
	}
	return result
}

func Argmax(input []float64) int {
	if input == nil || len(input) == 0 {
		panic("input must have at least one item")
	}
	return_value := 0
	max := input[0]
	for index, value := range input {
		if value > max {
			return_value = index
			max = value
		}
	}
	return return_value
}

func Compare(first_input, second_input []float64) []float64 {
	if first_input == nil || len(first_input) == 0 {
		panic("input 1 must have at least one item")
	}

	if second_input == nil || len(second_input) == 0 {
		panic("input must have at least one item")
	}
	min := max(len(first_input), len(second_input))
	output := make([]float64, max(len(first_input), len(second_input)))
	for index := range output {
		if index == min {
			break
		}
		if first_input[index] == second_input[index] {
			output[index] = 1
		}
	}
	return output
}

//​np.sum(dvalues, ​axis​=​0​, ​keepdims=​True​)
func SumAxis0KeepDimsTrue(input *mat.Dense) *mat.Dense {
	ouput := mat.NewDense(1, input.RawMatrix().Cols, nil)
	for col := 0; col < ouput.RawMatrix().Cols; col++ {
		sum := 0.0
		for row := 0; row < input.RawMatrix().Rows; row++ {
			sum += input.At(row, col)
		}
		ouput.Set(0, col, sum)
	}
	return ouput
}
func Accuracy(predictions, targets *mat.Dense) float64 {
	//TODO: panic if predictions and targets arent' the same shape

	//TODO: better way to do this? But I can't have this function destroy the inputs
	target_copy := mat.DenseCopyOf(targets)
	predictions_copy := mat.DenseCopyOf(predictions)

	argmax_predictions := mat.NewDense(predictions_copy.RawMatrix().Rows, predictions_copy.RawMatrix().Cols, nil)
	argmax_targets := mat.NewDense(target_copy.RawMatrix().Rows, target_copy.RawMatrix().Cols, nil)

	argmax_predictions.Copy(predictions_copy)
	target_copy.Copy(argmax_targets)

	for i := 0; i < 3; i++ {
		argmax_predictions.Set(0, i, float64(Argmax(predictions_copy.RawRowView(i))))
		argmax_targets.Set(0, i, float64(Argmax(target_copy.RawRowView(i))))
	}

	comparison := mat.NewDense(argmax_predictions.RawMatrix().Rows, argmax_predictions.RawMatrix().Cols, Compare(argmax_predictions.RawMatrix().Data, argmax_targets.RawMatrix().Data))
	mean := mat.Sum(comparison) / float64(argmax_predictions.RawMatrix().Cols)

	return mean
}
