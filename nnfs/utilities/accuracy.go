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
	threshold float64
	y         *mat.Dense
}

func NewAccuracy_Classification(threshold float64, y *mat.Dense) *Accuracy_Classification {
	accuracy := &Accuracy_Classification{threshold: threshold, y: y}
	return accuracy
}

func (accuracy *Accuracy_Classification) Compare(predictions, y *mat.Dense) float64 {
	ac := 0.0
	//turn all the flots into 0s or 1s
	predictions.Apply(func(r, c int, v float64) float64 {
		if v >= accuracy.threshold {
			return 1
		}
		return 0
	}, predictions)
	//compare to the truth and add up the 1s
	predictions.Apply(func(r, c int, v float64) float64 {
		if v == y.At(r, c) {
			ac += 1
		}
		return v
	}, predictions)
	ac = ac / float64(predictions.RawMatrix().Rows)
	return ac
}
