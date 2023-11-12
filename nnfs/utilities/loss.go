package nnfs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Loss interface {
	Forward(y_pred *mat.Dense, y_true *mat.Dense) *mat.Dense
}

type Loss_CategoricalCrossentropy struct {
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

func clip(value, left, right float64) float64 {
	if value >= left && value <= right {
		return value
	}
	if value <= left {
		return left
	}
	return right
}
