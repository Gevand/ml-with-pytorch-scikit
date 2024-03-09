package nnfs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Create_data(samples int) (*mat.Dense, *mat.Dense) {
	var data = Arange(samples)
	var X = mat.NewDense(samples, 1, data)
	var y = mat.NewDense(samples, 1, nil)
	X.Apply(func(r, c int, v float64) float64 {

		return v / float64(samples)
	}, X)
	y.Apply(func(r, c int, v float64) float64 {

		return math.Sin(2 * math.Pi * X.At(r, c))
	}, y)
	return X, y
}
