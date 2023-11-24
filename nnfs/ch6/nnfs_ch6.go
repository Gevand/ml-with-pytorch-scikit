package nnfs

import (
	"fmt"
	"math/rand"
	u "nnfs/utilities"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func Run1() {
	fmt.Println("Creating vertical data")

	X, y := u.Create_vertical_data(100, 3)
	fmt.Println("X:", X, "y:", y)
	p := plot.New()
	p.Title.Text = "Vertical"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := make(plotter.XYs, 100)
	pts2 := make(plotter.XYs, 100)
	pts3 := make(plotter.XYs, 100)
	for i := range pts1 {
		pts1[i].X = X.At(i, 0)
		pts1[i].Y = X.At(i, 1)

		pts2[i].X = X.At(i+100, 0)
		pts2[i].Y = X.At(i+100, 1)

		pts3[i].X = X.At(i+200, 0)
		pts3[i].Y = X.At(i+200, 1)
	}

	err := plotutil.AddScatters(p,
		"First", pts1,
		"Second", pts2,
		"Third", pts3)
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch6/vertical.png"); err != nil {
		panic(err)
	}
}

func Run2() {
	fmt.Println("Guessing the weights completely randomly")
	X, y := u.Create_vertical_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 3)
	activation_1 := u.NewActivationRelu()

	dense_2 := u.NewLayerDense(3, 3)
	activation_2 := u.NewActivationSoftMax()

	loss_function := u.NewLoss_CategoricalCrossentropy()

	best_loss := 999999.0 //book has this, but could be infinitie

	best_dense1_weights := mat.NewDense(2, 3, nil)
	best_dense1_biases := mat.NewDense(1, 3, nil)
	best_dense2_weights := mat.NewDense(3, 3, nil)
	best_dense2_biases := mat.NewDense(1, 3, nil)

	for iteration := 0; iteration < 1000; iteration++ {

		//apply randomness to weights
		dense_1.Weights.Apply(func(r, c int, v float64) float64 {
			return rand.NormFloat64() * .05
		}, dense_1.Weights)
		dense_1.Biases.Apply(func(r, c int, v float64) float64 {
			return rand.NormFloat64() * .05
		}, dense_1.Biases)
		dense_2.Weights.Apply(func(r, c int, v float64) float64 {
			return rand.NormFloat64() * .05
		}, dense_2.Weights)
		dense_2.Biases.Apply(func(r, c int, v float64) float64 {
			return rand.NormFloat64() * .05
		}, dense_2.Biases)

		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)
		activation_2.Forward(dense_2.Output)

		loss := u.CalculateLoss(loss_function, activation_2.Output, y_one_hot)
		accuracy := u.Accuracy(activation_2.Output, y_one_hot)

		if loss < best_loss {
			fmt.Println("New set of weights found, iteration:", iteration, "loss:", loss, "acc:", accuracy)
			best_loss = loss
			best_dense1_weights.Copy(dense_1.Weights)
			best_dense1_biases.Copy(dense_1.Biases)
			best_dense2_weights.Copy(dense_2.Weights)
			best_dense2_biases.Copy(dense_2.Biases)
		}
	}
}
func Run3() {
	fmt.Println("Adjusting the weights randomly, but keeping them for future iterations")
	X, y := u.Create_vertical_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 3)
	activation_1 := u.NewActivationRelu()

	dense_2 := u.NewLayerDense(3, 3)
	activation_2 := u.NewActivationSoftMax()

	loss_function := u.NewLoss_CategoricalCrossentropy()

	best_loss := 999999.0 //book has this, but could be infinitie

	best_dense1_weights := mat.NewDense(2, 3, nil)
	best_dense1_biases := mat.NewDense(1, 3, nil)
	best_dense2_weights := mat.NewDense(3, 3, nil)
	best_dense2_biases := mat.NewDense(1, 3, nil)

	for iteration := 0; iteration < 1000; iteration++ {

		//Adjust them now
		dense_1.Weights.Apply(func(r, c int, v float64) float64 {
			return v + (rand.NormFloat64() * .05)
		}, dense_1.Weights)
		dense_1.Biases.Apply(func(r, c int, v float64) float64 {
			return v + (rand.NormFloat64() * .05)
		}, dense_1.Biases)
		dense_2.Weights.Apply(func(r, c int, v float64) float64 {
			return v + (rand.NormFloat64() * .05)
		}, dense_2.Weights)
		dense_2.Biases.Apply(func(r, c int, v float64) float64 {
			return v + (rand.NormFloat64() * .05)
		}, dense_2.Biases)

		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)
		activation_2.Forward(dense_2.Output)

		loss := u.CalculateLoss(loss_function, activation_2.Output, y_one_hot)
		accuracy := u.Accuracy(activation_2.Output, y_one_hot)

		if loss < best_loss {
			fmt.Println("New set of weights found, iteration:", iteration, "loss:", loss, "acc:", accuracy)
			best_loss = loss
			best_dense1_weights.Copy(dense_1.Weights)
			best_dense1_biases.Copy(dense_1.Biases)
			best_dense2_weights.Copy(dense_2.Weights)
			best_dense2_biases.Copy(dense_2.Biases)
		} else {
			//keep what we currently have
			dense_1.Weights.Copy(best_dense1_weights)
			dense_1.Biases.Copy(best_dense1_biases)
			dense_2.Weights.Copy(best_dense2_weights)
			dense_2.Biases.Copy(best_dense2_biases)
		}
	}
}

func Run4() {
	fmt.Println("Same as run4 but with spiral")
	X, y := u.Create_spiral_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 3)
	activation_1 := u.NewActivationRelu()

	dense_2 := u.NewLayerDense(3, 3)
	activation_2 := u.NewActivationSoftMax()

	loss_function := u.NewLoss_CategoricalCrossentropy()

	best_loss := 999999.0 //book has this, but could be infinitie

	best_dense1_weights := mat.NewDense(2, 3, nil)
	best_dense1_biases := mat.NewDense(1, 3, nil)
	best_dense2_weights := mat.NewDense(3, 3, nil)
	best_dense2_biases := mat.NewDense(1, 3, nil)

	for iteration := 0; iteration < 1000; iteration++ {

		//Adjust them now
		dense_1.Weights.Apply(func(r, c int, v float64) float64 {
			return v + (rand.NormFloat64() * .05)
		}, dense_1.Weights)
		dense_1.Biases.Apply(func(r, c int, v float64) float64 {
			return v + (rand.NormFloat64() * .05)
		}, dense_1.Biases)
		dense_2.Weights.Apply(func(r, c int, v float64) float64 {
			return v + (rand.NormFloat64() * .05)
		}, dense_2.Weights)
		dense_2.Biases.Apply(func(r, c int, v float64) float64 {
			return v + (rand.NormFloat64() * .05)
		}, dense_2.Biases)

		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)
		activation_2.Forward(dense_2.Output)

		loss := u.CalculateLoss(loss_function, activation_2.Output, y_one_hot)
		accuracy := u.Accuracy(activation_2.Output, y_one_hot)

		if loss < best_loss {
			fmt.Println("New set of weights found, iteration:", iteration, "loss:", loss, "acc:", accuracy)
			best_loss = loss
			best_dense1_weights.Copy(dense_1.Weights)
			best_dense1_biases.Copy(dense_1.Biases)
			best_dense2_weights.Copy(dense_2.Weights)
			best_dense2_biases.Copy(dense_2.Biases)
		} else {
			//keep what we currently have
			dense_1.Weights.Copy(best_dense1_weights)
			dense_1.Biases.Copy(best_dense1_biases)
			dense_2.Weights.Copy(best_dense2_weights)
			dense_2.Biases.Copy(best_dense2_biases)
		}
	}
}
