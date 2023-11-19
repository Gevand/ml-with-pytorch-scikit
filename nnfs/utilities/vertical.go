package nnfs

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func Create_vertical_data(samples int, classes int) (*mat.Dense, *mat.Dense) {
	//X = np.zeros((samples*classes, 2))
	var X = mat.NewDense(samples*classes, 2, nil)
	//y = np.zeros(samples*classes, dtype='uint8')
	var y = mat.NewDense(samples*classes, 1, nil)

	class_distance := 1.0 / float64(classes)
	for class_number := 0; class_number < classes; class_number++ {
		for ix := 0; ix < samples; ix++ {
			offset := samples * class_number
			r := Linspace((class_distance * float64(class_number)), 0.0+(class_distance*float64(class_number+1)), samples)
			X.Set(ix+offset, 0, r[ix]+rand.NormFloat64()*.05)
			X.Set(ix+offset, 1, rand.NormFloat64()*0.7)
			y.Set(ix+offset, 0, float64(class_number))
		}
	}
	return X, y
}
