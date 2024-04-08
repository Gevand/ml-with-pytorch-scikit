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
