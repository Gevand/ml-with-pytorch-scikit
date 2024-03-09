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
	fmt.Print("Binary classifier")
	sample_size := 1000
	class_size := 2
	X, y := u.Create_spiral_data(sample_size, class_size)

	dense_1 := u.NewLayerDense(2, 64)
	dense_1.Name = "Dense 1"
	dense_1.Bias_Regulizer_L2 = 5e-4
	dense_1.Weight_Regulizer_L2 = 5e-4
	activation_1 := u.NewActivationRelu()

	dense_2 := u.NewLayerDense(64, 1)
	dense_2.Name = "Dense 2"
	activation_2 := u.NewActivationSigmoid()

	loss := u.NewLoss_BinaryCrossentropy()
	optimizer := u.NewOptimizerAdam(0.01, 5e-7, 1e-7, 0.9, 0.999)

	max_epoch := 10001
	for i := 0; i < max_epoch; i++ {
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)
		activation_2.Forward(dense_2.Output)
		data_loss := u.CalculateLoss(loss, activation_2.Output, y)

		if i%100 == 0 {
			regularization_loss := u.RegularizationLoss(loss, dense_1) + u.RegularizationLoss(loss, dense_2)
			loss := data_loss + regularization_loss
			predictions := mat.NewDense(activation_2.Output.RawMatrix().Rows, activation_2.Output.RawMatrix().Cols, nil)
			predictions.Apply(func(r, c int, v float64) float64 {
				if activation_2.Output.At(r, c) >= .5 {
					return 1
				}
				return 0
			}, predictions)
			accuracy := 0.0
			predictions.Apply(func(r, c int, v float64) float64 {
				if v == y.At(r, c) {
					accuracy += 1
				}
				return v
			}, predictions)
			accuracy = accuracy / float64(predictions.RawMatrix().Rows)
			fmt.Println("epoch", i, "data loss -->", data_loss, "regularization_loss -->", regularization_loss, "loss -->", loss, "accuracy -->", accuracy)
		}

		//backward pass
		loss.Backward(activation_2.Output, y)
		activation_2.Backward(loss.Dinputs)
		dense_2.Backward(activation_2.Dinputs)
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
	activation_2.Forward(dense_2.Output)

	graph_color.Apply(func(r, c int, v float64) float64 {
		if activation_2.Output.At(r, c) >= .5 {
			return 1
		}
		return 0
	}, graph_color)

	p := plot.New()
	p.Title.Text = "Spiral"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := plotter.XYs{}
	pts2 := plotter.XYs{}

	for i := 0; i < graph_x_y.RawMatrix().Rows; i++ {
		x_val := graph_x_y.At(i, 0)
		y_val := graph_x_y.At(i, 1)
		switch graph_color.At(i, 0) {
		case 0:
			pts1 = append(pts1, plotter.XY{X: x_val, Y: y_val})
		case 1:
			pts2 = append(pts2, plotter.XY{X: x_val, Y: y_val})
		}
	}

	plotutil.AddScatters(p,
		pts1,
		pts2)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch16/spiral_output.png"); err != nil {
		panic(err)
	}
}
