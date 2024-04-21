package nnfs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type IAccuracy interface {
	Compare(predictions, y *mat.Dense) float64
}
type Accuracy_Regression struct {
	precision float64
	reinit    bool
	y         *mat.Dense
}

func NewAccuracy_Regression(precision float64, reinit bool, y *mat.Dense) *Accuracy_Regression {
	accuracy := &Accuracy_Regression{precision: precision, reinit: reinit, y: y}
	if reinit {
		accuracy.precision = Calculate_std(y) / 25 //Being within 1/25th of a standard deviation is good enough :)
	}
	return accuracy
}

func (accuracy *Accuracy_Regression) Compare(predictions, y *mat.Dense) float64 {
	predictions.Apply(func(r, c int, v float64) float64 {
		real := y.At(r, 0)
		if math.Abs(v-real) < accuracy.precision {
			return 1
		}
		return 0
	}, predictions)
	pred_sum := mat.Sum(predictions)
	row_cnt := float64(y.RawMatrix().Rows)
	return pred_sum / row_cnt * 100
}

type Accuracy_Classification struct {
}

func NewAccuracy_Classification() *Accuracy_Classification {
	accuracy := &Accuracy_Classification{}
	return accuracy
}

func (accuracy *Accuracy_Classification) Compare(predictions, targets *mat.Dense) float64 {
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
