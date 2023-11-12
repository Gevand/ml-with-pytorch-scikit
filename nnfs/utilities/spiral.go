package nnfs

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Copyright (c) 2015 Andrej Karpathy
// License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
// Source: https://cs231n.github.io/neural-networks-case-study/
func Create_spiral_data(samples int, classes int) (*mat.Dense, *mat.Dense) {
	//X = np.zeros((samples*classes, 2))
	var X = mat.NewDense(samples*classes, 2, nil)
	//y = np.zeros(samples*classes, dtype='uint8')
	var y = mat.NewDense(samples*classes, 1, nil)

	//for class_number in range(classes):
	for class_number := 0; class_number < classes; class_number++ {
		//ix = range(samples*class_number, samples*(class_number+1))
		for ix := 0; ix < samples; ix++ {
			offset := samples * class_number
			// r = np.linspace(0.0, 1, samples)
			r := Linspace(0.0, 1, samples)
			// t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
			t := Linspace(float64(class_number*4.0), float64((class_number+1)*4), samples)
			for i := range t {
				noise := rand.NormFloat64() * .2
				t[i] += noise
			}
			//X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
			X.Set(ix+offset, 0, r[ix]*math.Sin(t[ix]*2.5))
			X.Set(ix+offset, 1, r[ix]*math.Cos(t[ix]*2.5))
			//y[ix] = class_number
			y.Set(ix+offset, 0, float64(class_number))
		}

	}

	return X, y
}
