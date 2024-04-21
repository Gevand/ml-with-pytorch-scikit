package nnfs

import (
	"fmt"
	u "nnfs/utilities"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func Run1() {
	fmt.Println("Using SGD")
	X, y := u.Create_spiral_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 64)
	activation_1 := u.NewActivationRelu()
	dense_2 := u.NewLayerDense(64, 3)
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	optimizer := u.NewOptimizerSGD(0.05, 0.001, 0)

	max_epoch := 1
	for i := 0; i < max_epoch; i++ {
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)

		loss := loss_activation.ForwardCombined(dense_2.Output, y_one_hot)

		if i%100 == 0 {
			fmt.Println("epoch", i, "loss -->", loss)
			accuracy := u.Accuracy(loss_activation.Activation.Output, y_one_hot)
			fmt.Println("epoch", i, "acc -->", accuracy)
		}

		//backward pass
		loss_activation.BackwardCombined(loss_activation.Activation.Output, y_one_hot)
		dense_2.Backward(loss_activation.Dinputs)
		activation_1.Backward(dense_2.Dinputs)
		dense_1.Backward(activation_1.Dinputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParameters(dense_1)
		optimizer.UpdateParameters(dense_2)
		optimizer.PostUpdateParams()
	}
}

func Run2() {
	fmt.Println("Using SGD With Momentum")
	X, y := u.Create_spiral_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 64)
	activation_1 := u.NewActivationRelu()
	dense_2 := u.NewLayerDense(64, 3)
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	optimizer := u.NewOptimizerSGD(0.05, 0.001, 0.3)

	max_epoch := 10001
	for i := 0; i < max_epoch; i++ {
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)

		loss := loss_activation.ForwardCombined(dense_2.Output, y_one_hot)

		if i%100 == 0 {
			fmt.Println("epoch", i, "loss -->", loss)
			accuracy := u.Accuracy(loss_activation.Activation.Output, y_one_hot)
			fmt.Println("epoch", i, "acc -->", accuracy)
		}

		//backward pass
		loss_activation.BackwardCombined(loss_activation.Activation.Output, y_one_hot)
		dense_2.Backward(loss_activation.Dinputs)
		activation_1.Backward(dense_2.Dinputs)
		dense_1.Backward(activation_1.Dinputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParameters(dense_1)
		optimizer.UpdateParameters(dense_2)
		optimizer.PostUpdateParams()
	}

	graph_x_y := mat.NewDense(1000*1000, 2, nil)

	//build an x, y array for the graph that's a 1000, by 1000 box in the region -1-1 on both x and ys

	graph_x_y.Apply(func(r, c int, v float64) float64 {
		if c == 1 {
			return float64(int(r/1000))*.001*2 - 1
		}
		return float64((r%1000))/1000*2 - 1
	}, graph_x_y)

	//now figure out
	var graph_color = mat.NewDense(1000*1000, 1, nil)
	dense_1.Forward(graph_x_y)
	activation_1.Forward(dense_1.Output)
	dense_2.Forward(activation_1.Output)
	loss_activation.Activation.Forward(dense_2.Output)

	graph_color.Apply(func(r, c int, v float64) float64 {
		row := loss_activation.Activation.Output.RawRowView(r)
		return float64(u.Argmax(row))
	}, graph_color)

	p := plot.New()
	p.Title.Text = "Spiral"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := plotter.XYs{}
	pts2 := plotter.XYs{}
	pts3 := plotter.XYs{}

	for i := 0; i < graph_x_y.RawMatrix().Rows; i++ {
		x_val := graph_x_y.At(i, 0)
		y_val := graph_x_y.At(i, 1)
		switch graph_color.At(i, 0) {
		case 0:
			pts1 = append(pts1, plotter.XY{X: x_val, Y: y_val})
		case 1:
			pts2 = append(pts2, plotter.XY{X: x_val, Y: y_val})
		case 2:
			pts3 = append(pts3, plotter.XY{X: x_val, Y: y_val})
		}
	}

	plotutil.AddScatters(p,
		pts1,
		pts2,
		pts3)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch10/spiral_output_with_momentum.png"); err != nil {
		panic(err)
	}
}

func Run3() {
	fmt.Println("Using ADAGRAD ")
	X, y := u.Create_spiral_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 64)
	activation_1 := u.NewActivationRelu()
	dense_2 := u.NewLayerDense(64, 3)
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	optimizer := u.NewOptimizerAdaGrad(0.5, 0.001, .0001)

	max_epoch := 10001
	for i := 0; i < max_epoch; i++ {
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)

		loss := loss_activation.ForwardCombined(dense_2.Output, y_one_hot)

		if i%100 == 0 {
			fmt.Println("epoch", i, "loss -->", loss)
			accuracy := u.Accuracy(loss_activation.Activation.Output, y_one_hot)
			fmt.Println("epoch", i, "acc -->", accuracy)
		}

		//backward pass
		loss_activation.BackwardCombined(loss_activation.Activation.Output, y_one_hot)
		dense_2.Backward(loss_activation.Dinputs)
		activation_1.Backward(dense_2.Dinputs)
		dense_1.Backward(activation_1.Dinputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParameters(dense_1)
		optimizer.UpdateParameters(dense_2)
		optimizer.PostUpdateParams()
	}
}

func Run4() {
	fmt.Println("Using RMSProp")
	X, y := u.Create_spiral_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 64)
	activation_1 := u.NewActivationRelu()
	dense_2 := u.NewLayerDense(64, 3)
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	optimizer := u.NewOptimizerRmsProp(0.02, 1e-5, .00001, 0.999)

	max_epoch := 10001
	for i := 0; i < max_epoch; i++ {
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)

		loss := loss_activation.ForwardCombined(dense_2.Output, y_one_hot)

		if i%100 == 0 {
			fmt.Println("epoch", i, "loss -->", loss)
			accuracy := u.Accuracy(loss_activation.Activation.Output, y_one_hot)
			fmt.Println("epoch", i, "acc -->", accuracy)
		}

		//backward pass
		loss_activation.BackwardCombined(loss_activation.Activation.Output, y_one_hot)
		dense_2.Backward(loss_activation.Dinputs)
		activation_1.Backward(dense_2.Dinputs)
		dense_1.Backward(activation_1.Dinputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParameters(dense_1)
		optimizer.UpdateParameters(dense_2)
		optimizer.PostUpdateParams()
	}

	graph_x_y := mat.NewDense(1000*1000, 2, nil)
	//build an x, y array for the graph that's a 1000, by 1000 box in the region -1-1 on both x and ys
	graph_x_y.Apply(func(r, c int, v float64) float64 {
		if c == 1 {
			return float64(int(r/1000))*.001*2 - 1
		}
		return float64((r%1000))/1000*2 - 1
	}, graph_x_y)

	//now figure out
	var graph_color = mat.NewDense(1000*1000, 1, nil)
	dense_1.Forward(graph_x_y)
	activation_1.Forward(dense_1.Output)
	dense_2.Forward(activation_1.Output)
	loss_activation.Activation.Forward(dense_2.Output)

	graph_color.Apply(func(r, c int, v float64) float64 {
		row := loss_activation.Activation.Output.RawRowView(r)
		return float64(u.Argmax(row))
	}, graph_color)

	p := plot.New()
	p.Title.Text = "Spiral"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := plotter.XYs{}
	pts2 := plotter.XYs{}
	pts3 := plotter.XYs{}

	for i := 0; i < graph_x_y.RawMatrix().Rows; i++ {
		x_val := graph_x_y.At(i, 0)
		y_val := graph_x_y.At(i, 1)
		switch graph_color.At(i, 0) {
		case 0:
			pts1 = append(pts1, plotter.XY{X: x_val, Y: y_val})
		case 1:
			pts2 = append(pts2, plotter.XY{X: x_val, Y: y_val})
		case 2:
			pts3 = append(pts3, plotter.XY{X: x_val, Y: y_val})
		}
	}

	plotutil.AddScatters(p,
		pts1,
		pts2,
		pts3)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch10/spiral_output_with_rmsprop.png"); err != nil {
		panic(err)
	}
}

func Run5() {
	fmt.Println("Using Adam")
	X, y := u.Create_spiral_data(100, 3)

	y_one_hot := mat.NewDense(300, 3, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 64)
	dense_1.Name = "Dense 1"
	activation_1 := u.NewActivationRelu()
	dense_2 := u.NewLayerDense(64, 3)
	dense_2.Name = "Dense 2"
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	optimizer := u.NewOptimizerAdam(0.02, 5e-7, 1e-7, 0.9, 0.999)

	max_epoch := 10001
	for i := 0; i < max_epoch; i++ {
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)

		loss := loss_activation.ForwardCombined(dense_2.Output, y_one_hot)

		if i%100 == 0 {
			fmt.Println("epoch", i, "loss -->", loss)
			accuracy := u.Accuracy(loss_activation.Activation.Output, y_one_hot)
			fmt.Println("epoch", i, "acc -->", accuracy)
		}

		//backward pass
		loss_activation.BackwardCombined(loss_activation.Activation.Output, y_one_hot)
		dense_2.Backward(loss_activation.Dinputs)
		activation_1.Backward(dense_2.Dinputs)
		dense_1.Backward(activation_1.Dinputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParameters(dense_1)
		optimizer.UpdateParameters(dense_2)
		optimizer.PostUpdateParams()
	}

	graph_x_y := mat.NewDense(1000*1000, 2, nil)
	//build an x, y array for the graph that's a 1000, by 1000 box in the region -1-1 on both x and ys
	graph_x_y.Apply(func(r, c int, v float64) float64 {
		if c == 1 {
			return float64(int(r/1000))*.001*2 - 1
		}
		return float64((r%1000))/1000*2 - 1
	}, graph_x_y)

	//now figure out
	var graph_color = mat.NewDense(1000*1000, 1, nil)
	dense_1.Forward(graph_x_y)
	activation_1.Forward(dense_1.Output)
	dense_2.Forward(activation_1.Output)
	loss_activation.Activation.Forward(dense_2.Output)

	graph_color.Apply(func(r, c int, v float64) float64 {
		row := loss_activation.Activation.Output.RawRowView(r)
		return float64(u.Argmax(row))
	}, graph_color)

	p := plot.New()
	p.Title.Text = "Spiral"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := plotter.XYs{}
	pts2 := plotter.XYs{}
	pts3 := plotter.XYs{}

	for i := 0; i < graph_x_y.RawMatrix().Rows; i++ {
		x_val := graph_x_y.At(i, 0)
		y_val := graph_x_y.At(i, 1)
		switch graph_color.At(i, 0) {
		case 0:
			pts1 = append(pts1, plotter.XY{X: x_val, Y: y_val})
		case 1:
			pts2 = append(pts2, plotter.XY{X: x_val, Y: y_val})
		case 2:
			pts3 = append(pts3, plotter.XY{X: x_val, Y: y_val})
		}
	}

	plotutil.AddScatters(p,
		pts1,
		pts2,
		pts3)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch10/spiral_output_with_adam.png"); err != nil {
		panic(err)
	}
}
