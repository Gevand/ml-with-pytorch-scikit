package nnfs

import (
	"fmt"
	"math"
	nnfs "nnfs/utilities"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func Run1() {
	fmt.Println("Starting ch9 -- Forward Pass + Back Propogation for a single neuron")

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
	d_relu_dz := gradient * d_z
	fmt.Println(d_relu_dz)

	//partial derivatives
	dsum_dxw0 := 1.0
	dsum_dxw1 := 1.0
	dsum_dxw2 := 1.0
	dsum_db := 1.0
	drelu_dxw0 := d_relu_dz * dsum_dxw0
	drelu_dxw1 := d_relu_dz * dsum_dxw1
	drelu_dxw2 := d_relu_dz * dsum_dxw2
	drelu_db := d_relu_dz * dsum_db

	fmt.Println(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)
	dmul_dx0 := w[0]
	dmul_dx1 := w[1]
	dmul_dx2 := w[2]
	dmul_dw0 := x[0]
	dmul_dw1 := x[1]
	dmul_dw2 := x[2]

	drel_dx0 := drelu_dxw0 * dmul_dx0
	drel_dw0 := drelu_dxw0 * dmul_dw0
	drel_dx1 := drelu_dxw1 * dmul_dx1
	drel_dw1 := drelu_dxw1 * dmul_dw1
	drel_dx2 := drelu_dxw2 * dmul_dx2
	drel_dw2 := drelu_dxw2 * dmul_dw2
	fmt.Println(drel_dx0, drel_dw0, drel_dx1, drel_dw1, drel_dx2, drel_dw2)
}

func Run2() {
	fmt.Println("Back Propogation for a three neurons - Gradient w.r.t Inputs")

	//d values passed formt he gradients of the preivous layer, set to 1 for the purpose of this example
	//we got three gradients
	dvalues := [][]float64{{1.0, 1.0, 1.0}}

	//4 inputs -> 3 neurons, so there are 12 weights
	weights := [][]float64{
		{0.2, 0.8, -0.5, 1},
		{0.5, -0.91, .26, -.5},
		{-0.26, -0.27, 0.17, 0.87}}
	dx0 := (weights[0][0] + weights[1][0] + weights[2][0]) * dvalues[0][0]
	dx1 := (weights[0][1] + weights[1][1] + weights[2][1]) * dvalues[0][0]
	dx2 := (weights[0][2] + weights[1][2] + weights[2][2]) * dvalues[0][0]
	dx3 := (weights[0][3] + weights[1][3] + weights[2][3]) * dvalues[0][0]
	fmt.Println(dx0, dx1, dx2, dx3)
}

func Run3() {
	fmt.Println("Back Propogation for a three neurons - Gradient w.r.t Inputs using dot product")

	dvalues := mat.NewDense(1, 3, []float64{1.0, 1.0, 1.0})
	weights := mat.NewDense(3, 4, []float64{
		.2, .8, -.5, 1,
		.5, -.91, 0.26, -.5,
		-.26, -.27, .17, .87})

	dinputs := mat.NewDense(1, 4, nil)
	dinputs.Mul(dvalues, weights)
	fmt.Println(dinputs.RowView(0))
}

func Run4() {
	fmt.Println("Back Propogation for a three neurons - Gradient w.r.t Inputs dotproduct with batch")

	dvalues := mat.NewDense(3, 3, []float64{
		1.0, 1.0, 1.0,
		2.0, 2.0, 2.0,
		3.0, 3.0, 3.0})
	weights := mat.NewDense(3, 4, []float64{
		.2, .8, -.5, 1,
		.5, -.91, 0.26, -.5,
		-.26, -.27, .17, .87})

	dinputs := mat.NewDense(3, 4, nil)
	dinputs.Mul(dvalues, weights)
	fmt.Println(dinputs)
}

func Run5() {
	fmt.Println("Back Propogation for a three neurons - Gradient w.r.t Weights ")
	dvalues := mat.NewDense(3, 3, []float64{
		1.0, 1.0, 1.0,
		2.0, 2.0, 2.0,
		3.0, 3.0, 3.0})
	inputs := mat.NewDense(3, 4, []float64{
		1, 2, 3, 2.5,
		2., 5., -1., 2,
		-1.5, 2.7, 3.3, -.8})
	inputsT := inputs.T()
	dweights := mat.NewDense(4, 3, nil)
	dweights.Mul(inputsT, dvalues)
	fmt.Println(dweights)
}

func Run6() {
	fmt.Println("Back Propogation for a three neurons - Gradient w.r.t Biases ")
	dvalues := mat.NewDense(3, 3, []float64{
		1.0, 1.0, 1.0,
		2.0, 2.0, 2.0,
		3.0, 3.0, 3.0})
	biases := mat.NewDense(1, 3, []float64{2, 3, .5})
	fmt.Println("Biases will not be used in the calculation because the derivative is always 1 for them", biases)
	dbiases := nnfs.SumAxis0KeepDimsTrue(dvalues)
	fmt.Println(dbiases)
}
