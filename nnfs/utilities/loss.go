package nnfs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Loss interface {
	Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.Dense
}

type Loss_CategoricalCrossentropy struct {
	Dinputs *mat.Dense
}

func NewLoss_CategoricalCrossentropy() *Loss_CategoricalCrossentropy {
	output := &Loss_CategoricalCrossentropy{}
	return output

}

func (loss *Loss_CategoricalCrossentropy) Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.Dense {

	y_pred_clipped := mat.NewDense(y_pred.RawMatrix().Rows, y_pred.RawMatrix().Cols, nil)
	y_pred_clipped.Copy(y_pred)
	y_pred_clipped.Apply(func(r, c int, v float64) float64 {
		return clip(v, 1e-7, 1-1e-7)
	}, y_pred_clipped)

	y_pred_clipped.MulElem(y_pred_clipped, y_true)
	y_pred_clipped.Apply(func(r, c int, v float64) float64 {
		return -math.Log(mat.Sum(y_pred_clipped.RowView(r)))
	}, y_pred_clipped)
	return y_pred_clipped
}
func (loss *Loss_CategoricalCrossentropy) Backward(dvalues *mat.Dense, y_true *mat.Dense) {
	//expect the y_true to be one hot encoded, the book has some code here to transform this into a one hot encoded thing if its not
	loss.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	loss.Dinputs.Copy(dvalues)
	samples := float64(y_true.RawMatrix().Cols)
	loss.Dinputs.Apply(func(r, c int, v float64) float64 {
		return ((-1) * y_true.At(r, c)) / v / samples
	}, loss.Dinputs)
}

func CalculateLoss(loss Loss, y_pred *mat.Dense, y_true *mat.Dense) float64 {
	data_loss := 0.0
	sample_losses := loss.Forward(y_pred, y_true)
	sample_losses.Apply(func(r, c int, v float64) float64 {
		data_loss = data_loss + v
		return v
	}, sample_losses)
	data_loss = data_loss / float64(sample_losses.RawMatrix().Rows*sample_losses.RawMatrix().Cols)
	return data_loss
}

func RegularizationLoss(loss Loss, layer *LayerDense) float64 {
	regularization_loss := 0.0
	//l1 stuff
	if layer.Weight_Regulizer_L1 > 0 {
		var sum float64 = 0.0
		for _, value := range layer.Weights.RawMatrix().Data {
			sum += math.Abs(value)
		}
		regularization_loss += sum * layer.Weight_Regulizer_L1
	}

	if layer.Bias_Regulizer_L1 > 0 {
		var sum float64 = 0.0
		for _, value := range layer.Biases.RawMatrix().Data {
			sum += math.Abs(value)
		}
		regularization_loss += sum * layer.Bias_Regulizer_L1
	}

	//l2 stuff
	if layer.Weight_Regulizer_L2 > 0 {
		var sum float64 = 0.0
		for _, value := range layer.Weights.RawMatrix().Data {
			sum += math.Pow(value, 2)
		}
		regularization_loss += sum * layer.Weight_Regulizer_L2
	}

	if layer.Bias_Regulizer_L2 > 0 {
		var sum float64 = 0.0
		for _, value := range layer.Biases.RawMatrix().Data {
			sum += math.Pow(value, 2)
		}
		regularization_loss += sum * layer.Bias_Regulizer_L2
	}
	return regularization_loss
}

type Loss_BinaryCrossentropy struct {
	Dinputs *mat.Dense
}

func NewLoss_BinaryCrossentropy() *Loss_BinaryCrossentropy {
	output := &Loss_BinaryCrossentropy{}
	return output
}

func (loss *Loss_BinaryCrossentropy) Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.Dense {
	y_pred_clipped := mat.NewDense(y_pred.RawMatrix().Rows, y_pred.RawMatrix().Cols, nil)
	y_pred_clipped.Copy(y_pred)
	y_pred_clipped.Apply(func(r, c int, v float64) float64 {
		return clip(v, 1e-7, 1-1e-7)
	}, y_pred_clipped)

	sample_losses := mat.NewDense(y_pred.RawMatrix().Rows, y_pred.RawMatrix().Cols, nil)
	sample_losses.Apply(func(r, c int, v float64) float64 {
		y_true_r_c := y_true.At(r, c)
		y_pred_clipped_r_c := y_pred_clipped.At(r, c)

		return -1*(y_true_r_c*math.Log(y_pred_clipped_r_c)) + (1-y_true_r_c)*math.Log(1-y_pred_clipped_r_c)
	}, sample_losses)

	sample_losses_mean := mat.NewDense(y_pred.RawMatrix().Rows, 1, nil)
	column_count := float64(y_pred.RawMatrix().Cols)
	sample_losses_mean.Apply(func(r, c int, v float64) float64 {
		row_sum := mat.Sum(sample_losses.RowView(r))
		return row_sum / column_count
	}, sample_losses_mean)

	return sample_losses_mean
}
func (loss *Loss_BinaryCrossentropy) Backward(dvalues *mat.Dense, y_true *mat.Dense) {
	samples := float64(dvalues.RawMatrix().Rows)
	outputs := float64(dvalues.RawMatrix().Cols)

	clipped_dvalues := mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	clipped_dvalues.Copy(dvalues)
	clipped_dvalues.Apply(func(r, c int, v float64) float64 {
		return clip(v, 1e-7, 1-1e-7)
	}, clipped_dvalues)

	loss.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	loss.Dinputs.Apply(func(r, c int, v float64) float64 {
		clipped_dvalues_r_c := clipped_dvalues.At(r, c)
		y_true_value_r_c := y_true.At(r, c)
		return -(y_true_value_r_c/clipped_dvalues_r_c - (1-y_true_value_r_c)/(1-clipped_dvalues_r_c)) / outputs
	}, loss.Dinputs)

	loss.Dinputs.Apply(func(r, c int, v float64) float64 {
		return v / samples
	}, loss.Dinputs)

}

type Loss_MSE struct {
	Dinputs *mat.Dense
}

func NewLoss_MSE() *Loss_MSE {
	output := &Loss_MSE{}
	return output
}

func (loss *Loss_MSE) Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.Dense {
	sample_losses := mat.NewDense(y_pred.RawMatrix().Rows, y_pred.RawMatrix().Cols, nil)
	sample_losses.Apply(func(r, c int, v float64) float64 {
		return math.Pow((y_true.At(r, c) - y_pred.At(r, c)), 2.0)
	}, sample_losses)

	sample_losses_mean := mat.NewDense(y_pred.RawMatrix().Rows, 1, nil)
	column_count := float64(y_pred.RawMatrix().Cols)
	sample_losses_mean.Apply(func(r, c int, v float64) float64 {
		row_sum := mat.Sum(sample_losses.RowView(r))
		return row_sum / column_count
	}, sample_losses_mean)

	return sample_losses_mean
}

func (loss *Loss_MSE) Backward(dvalues *mat.Dense, y_true *mat.Dense) {
	samples := float64(dvalues.RawMatrix().Rows)
	outputs := float64(dvalues.RawMatrix().Cols)

	loss.Dinputs = mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)
	loss.Dinputs.Apply(func(r, c int, v float64) float64 {

		return -2 * (y_true.At(r, c) - dvalues.At(r, c)) / outputs
	}, loss.Dinputs)

	loss.Dinputs.Apply(func(r, c int, v float64) float64 {
		return v / samples
	}, loss.Dinputs)
}
func clip(value, left, right float64) float64 {
	if value >= left && value <= right {
		return value
	}
	if value <= left {
		return left
	}
	return right
}
