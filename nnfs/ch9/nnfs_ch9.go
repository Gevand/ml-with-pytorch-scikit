package nnfs

import (
	"fmt"
	"math"
	nnfs "nnfs/utilities"
	u "nnfs/utilities"
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

func Run7() {
	fmt.Println("Back propogation for three neurons - Gradient wrt to activation relu function")
	dvalues := mat.NewDense(3, 4, []float64{
		1.0, 3.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0})
	zvalues := mat.NewDense(3, 4, []float64{
		1.0, 2.0, -3.0, -4,
		2.0, 7.0, -1.0, 3,
		-1.0, 2.0, 5.0, -1})

	drelu := mat.NewDense(3, 4, nil)
	for i := 0; i < drelu.RawMatrix().Rows; i++ {
		for j := 0; j < drelu.RawMatrix().Cols; j++ {
			v := 0.0
			if zvalues.At(i, j) > 0 {
				v = 1.0
			}
			drelu.Set(i, j, v)
		}
	}
	fmt.Println("Drelu", drelu)
	drelu.MulElem(drelu, dvalues)
	fmt.Println("Drelu post multiply", drelu)
}
func Run8() {
	fmt.Println("Back propogation for three neurons - Gradient wrt to activation relu function - simpler version")
	dvalues := mat.NewDense(3, 4, []float64{
		1.0, 3.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0})
	zvalues := mat.NewDense(3, 4, []float64{
		1.0, 2.0, -3.0, -4,
		2.0, 7.0, -1.0, 3,
		-1.0, 2.0, 5.0, -1})
	drelu := mat.NewDense(3, 4, nil)
	drelu.Copy(dvalues)
	for i := 0; i < drelu.RawMatrix().Rows; i++ {
		for j := 0; j < drelu.RawMatrix().Cols; j++ {
			v := drelu.At(i, j)
			if zvalues.At(i, j) < 0 {
				v = 0.0
			}
			drelu.Set(i, j, v)
		}
	}
	fmt.Println("Drelu on run 8, should be same as run 7, just simpler", drelu)
}

func Run9() {
	fmt.Println("Back propogation for three neurons - Putting it all together")
	// dvalues := mat.NewDense(3, 3, []float64{
	// 	1.0, 1.0, 1.0,
	// 	2.0, 2.0, 2.0,
	// 	3.0, 3.0, 3.0})
	inputs := mat.NewDense(3, 4, []float64{
		1, 2, 3, 2.5,
		2., 5., -1., 2,
		-1.5, 2.7, 3.3, -.8})
	weights := mat.NewDense(3, 4, []float64{
		.2, .8, -.5, 1,
		.5, -.91, 0.26, -.5,
		-.26, -.27, .17, .87})
	weights_t := mat.DenseCopyOf(weights.T())
	biases := mat.NewDense(1, 3, []float64{2, 3, .5})

	//calculate forward -- (multiply * weights + bias)
	layer_outputs := mat.NewDense(3, 3, nil)
	layer_outputs.Mul(inputs, weights_t)
	layer_outputs.Apply(func(r, c int, v float64) float64 {
		return v + biases.At(0, c)
	}, layer_outputs)

	//calculate forward -- activation function (relu)
	fmt.Println("Layer outputs pre relu", layer_outputs)
	relu_outputs := mat.NewDense(3, 3, nil)
	relu_outputs.Copy(layer_outputs)
	relu_outputs.Apply(func(r, c int, v float64) float64 {
		if v > 1 {
			return v
		}
		return 0
	}, relu_outputs)
	fmt.Println("Layer outputs post relu", relu_outputs)

	//for this example use relu_outputs as drelu
	d_relu := mat.NewDense(3, 3, nil)
	d_relu.Copy(relu_outputs)
	d_relu.Apply(func(r, c int, v float64) float64 {
		if layer_outputs.At(r, c) <= 0 {
			return 0
		}
		return v
	}, d_relu)
	fmt.Println("Drelu -", d_relu)

	dinputs := mat.NewDense(3, 4, nil)
	dinputs.Mul(d_relu, weights)
	fmt.Println("Dinputs -", dinputs)

	dweights := mat.NewDense(4, 3, nil)
	dweights.Mul(inputs.T(), d_relu)
	fmt.Println("Dweights -", dweights)

	dbiases := nnfs.SumAxis0KeepDimsTrue(d_relu)
	fmt.Println("Dbiases -", dbiases)

	dweights.Apply(func(r, c int, v float64) float64 {
		return v * -0.001
	}, dweights)
	dbiases.Apply(func(r, c int, v float64) float64 {
		return v * -0.001
	}, dbiases)

	weights_t.Apply(func(r, c int, v float64) float64 {
		return v + dweights.At(r, c)
	}, weights_t)

	biases.Apply(func(r, c int, v float64) float64 {
		return v + dbiases.At(r, c)
	}, biases)

	fmt.Print("Weights post update", weights_t)
	fmt.Print("Biases post update", biases)
}

func Run10() {
	fmt.Println("Back propogation softmax testing")

	softmax_outputs := mat.NewDense(3, 3, []float64{
		.7, .1, .2,
		.1, .5, .4,
		.02, .9, .08})
	class_targets := mat.NewDense(3, 1, []float64{0, 1, 1})
	class_targets_one_hot := mat.NewDense(3, 3, nil)
	for i := 0; i < class_targets.RawMatrix().Rows; i++ {
		class_targets_one_hot.Set(i, int(class_targets.At(i, 0)), 1.0)
	}

	soft_max := u.NewActivationSoftMax()
	loss_function := u.NewLoss_CategoricalCrossentropy()
	loss_function.Backward(softmax_outputs, class_targets_one_hot)

	fmt.Println(loss_function.Dinputs)
	soft_max.Output = softmax_outputs //pretend this is what Forward calculated
	soft_max.Backward(loss_function.Dinputs)
	fmt.Println(soft_max.Dinputs)
}

func Run11() {
	fmt.Println("Back propogation softmax testing combined, should match run 10")

	softmax_outputs := mat.NewDense(3, 3, []float64{
		.7, .1, .2,
		.1, .5, .4,
		.02, .9, .08})
	class_targets := mat.NewDense(3, 1, []float64{0, 1, 1})
	class_targets_one_hot := mat.NewDense(3, 3, nil)
	for i := 0; i < class_targets.RawMatrix().Rows; i++ {
		class_targets_one_hot.Set(i, int(class_targets.At(i, 0)), 1.0)
	}

	soft_max_and_loss := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	soft_max_and_loss.Backward(softmax_outputs, class_targets_one_hot)
	fmt.Println(soft_max_and_loss.Dinputs)
}

func Run12() {
	fmt.Println("Combining everything + accuracy")
	X, y := u.Create_spiral_data(100, 3)

	//My loss function only works with one_hot_encoded, the book does some python stuff to handle
	//1-d arrays, but MatDense is always 2d, so its just easier to 1 hot encode
	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 3)
	activation_1 := u.NewActivationRelu()
	dense_1.Forward(X)
	activation_1.Forward(dense_1.Output)

	dense_2 := u.NewLayerDense(3, 3)
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	dense_2.Forward(activation_1.Output)
	loss := loss_activation.Forward(dense_2.Output, y_one_hot)

	accuracy := u.Accuracy(loss_activation.Activation.Output, y_one_hot)
	fmt.Println("loss:", loss, "accuracy:", accuracy)

	loss_activation.Backward(loss_activation.Activation.Output, y_one_hot)
	dense_2.Backward(loss_activation.Dinputs)
	activation_1.Backward(dense_2.Dinputs)
	dense_1.Backward(activation_1.Dinputs)

	fmt.Println("DENSE 1:", dense_1.Dweights, dense_1.Dbiases)
	fmt.Println("DENSE 2:", dense_2.Dweights, dense_2.Dbiases)
}
