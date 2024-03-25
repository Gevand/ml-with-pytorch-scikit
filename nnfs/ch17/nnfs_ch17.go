package nnfs

import (
	"fmt"
	"math"
	u "nnfs/utilities"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func Run1() {
	fmt.Println("Testing sine dataset")
	X, y := u.Create_data(1000)

	p := plot.New()
	p.Title.Text = "Sin"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := plotter.XYs{}
	for i := 0; i < X.RawMatrix().Rows; i++ {
		pts1 = append(pts1, plotter.XY{X: X.At(i, 0), Y: y.At(i, 0)})
	}

	plotutil.AddScatters(p, pts1)
	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch17/sine_output.png"); err != nil {
		panic(err)
	}
}

func Run2() {
	fmt.Println("Training a model to predict sine dataset as regression")

	X, y := u.Create_data(1000)
	std := u.Calculate_std(y)
	std = std / 25 //book uses 250, I made it 25
	dense_1 := u.NewLayerDense(1, 64)
	activation_1 := u.NewActivationRelu()
	dense_2 := u.NewLayerDense(64, 1)
	activation_2 := u.NewActivationLinear()
	loss_function := u.NewLoss_MSE()
	optimizer := u.NewOptimizerAdam(0.005, 1e-3, 1e-7, 0.9, 0.999)

	max_epoch := 10001
	for i := 0; i < max_epoch; i++ {
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dense_2.Forward(activation_1.Output)
		activation_2.Forward(dense_2.Output)

		data_loss := u.CalculateLoss(loss_function, activation_2.Output, y)
		if i%100 == 0 {
			regularization_loss := u.RegularizationLoss(loss_function, dense_1) + u.RegularizationLoss(loss_function, dense_2)
			loss := data_loss + regularization_loss

			predictions := mat.NewDense(activation_2.Output.RawMatrix().Rows, activation_2.Output.RawMatrix().Cols, nil)
			predictions.Apply(func(r, c int, v float64) float64 {
				pred := activation_2.Output.At(r, 0)
				real := y.At(r, 0)
				if math.Abs(pred-real) < std {
					return 1
				}
				return 0
			}, predictions)
			//all the 1s gets summed up
			pred_sum := mat.Sum(predictions)
			row_cnt := float64(y.RawMatrix().Rows)
			accuracy := pred_sum / row_cnt * 100
			fmt.Println("epoch", i, "data loss -->", data_loss, "regularization_loss -->", regularization_loss, "loss -->", loss, "accuracy -->", accuracy, "%")
		}

		loss_function.Backward(activation_2.Output, y)
		activation_2.Backward(loss_function.Dinputs)
		dense_2.Backward(activation_2.Dinputs)
		activation_1.Backward(dense_2.Dinputs)
		dense_1.Backward(activation_1.Dinputs)

		optimizer.PreUpdateParams()
		optimizer.UpdateParameters(dense_1)
		optimizer.UpdateParameters(dense_2)
		optimizer.PostUpdateParams()

	}
	p := plot.New()
	p.Title.Text = "Sin"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := plotter.XYs{}
	for i := 0; i < X.RawMatrix().Rows; i++ {
		pts1 = append(pts1, plotter.XY{X: X.At(i, 0), Y: activation_2.Output.At(i, 0)})
	}

	plotutil.AddScatters(p, pts1)
	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch17/sine_output_prediction.png"); err != nil {
		panic(err)
	}
}
