package nnfs

import (
	"fmt"
	"math"
	"strconv"
)

func Run1() {
	fmt.Println("Starting ch9 -- Forward Pass")

	x := [3]float64{1.0, -2.0, 3.0}
	w := [3]float64{-3.0, -1.0, 2.0}
	b := 1.0

	z := 0.0

	for i, v := range x {
		z += v * w[i]
		fmt.Println("x["+strconv.Itoa(i)+"] + y["+strconv.Itoa(i)+"] = ", v*w[i])
	}
	z = z + b
	fmt.Println("z = ", z)
	fmt.Println("Now need to push this through a RELU activation funciton")
	y := math.Max(z, 0)
	fmt.Println(y)

	//lets assumet he gradient is 1.0 and lets propogate it back
	gradient := 1.0

	// (dRelu()/dSum()) * (dSum()/dMul(x0,w0)) * (dMul(x0,w0)/dx0)
	d_z := 0.0
	if z > 0 {
		d_z = 1.0
	}
	d_relu_d_sum := gradient * d_z
	print(d_relu_d_sum)
}
