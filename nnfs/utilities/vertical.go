package nnfs

import "gonum.org/v1/gonum/mat"

func Create_vertical_data(samples int, classes int) (*mat.Dense, *mat.Dense) {
	//X = np.zeros((samples*classes, 2))
	var X = mat.NewDense(samples*classes, 2, nil)
	//y = np.zeros(samples*classes, dtype='uint8')
	var y = mat.NewDense(samples*classes, 1, nil)

	return X, y
}
