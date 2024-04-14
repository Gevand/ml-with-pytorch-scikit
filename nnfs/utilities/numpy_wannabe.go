package nnfs

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

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

func Arange(size int) []float64 {
	result := make([]float64, size)
	for i := range result {
		result[i] = float64(i)
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

// ​np.sum(dvalues, ​axis​=​0​, ​keepdims=​True​)
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

func Shuffle(X []*mat.Dense, y *mat.Dense) {
	if y.RawMatrix().Rows != len(X) {
		panic("Mismatch of sizes, X's length and y's row count should match")
	}

	rand.Shuffle(len(X), func(i, j int) {
		X[i], X[j] = X[j], X[i]
		temp := y.At(i, 0)
		y.Set(i, 0, y.At(j, 0))
		y.Set(j, 0, temp)
	})
}

func Binomial(number_of_experiments int, probability float64, size int) []float64 {
	output := make([]float64, size)
	experiment := 0
	for experiment < number_of_experiments {
		for index, _ := range output {
			output[index] += float64(rand.Intn(2))
		}
		experiment += 1
	}
	return output
}

func Accuracy(predictions, targets *mat.Dense) float64 {
	//TODO: panic if predictions and targets arent' the same shape

	//TODO: better way to do this? But I can't have this function destroy the inputs
	target_copy := mat.DenseCopyOf(targets)
	predictions_copy := mat.DenseCopyOf(predictions)

	argmax_predictions := mat.NewDense(predictions_copy.RawMatrix().Rows, 1, nil)
	argmax_targets := mat.NewDense(target_copy.RawMatrix().Rows, 1, nil)

	argmax_predictions.Copy(predictions_copy)
	target_copy.Copy(argmax_targets)

	for i := 0; i < argmax_predictions.RawMatrix().Rows; i++ {
		argmax_predictions.Set(i, 0, float64(Argmax(predictions_copy.RawRowView(i))))
		argmax_targets.Set(i, 0, float64(Argmax(target_copy.RawRowView(i))))
	}

	comparison := mat.NewDense(argmax_predictions.RawMatrix().Rows, argmax_predictions.RawMatrix().Cols, Compare(argmax_predictions.RawMatrix().Data, argmax_targets.RawMatrix().Data))

	sum := mat.Sum(comparison)
	mean := sum / float64(argmax_predictions.RawMatrix().Cols*argmax_predictions.RawMatrix().Rows)

	return mean
}
