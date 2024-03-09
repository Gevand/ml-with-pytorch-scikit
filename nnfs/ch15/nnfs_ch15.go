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
	fmt.Println("Dropout")
	dropout_rate := 0.5
	example_output := []float64{
		0.27, -1.03, .67, .99, 0.05,
		-0.37, -2.01, 1.13, -0.07, 0.73}

	for {
		index := rand.Intn(len(example_output))
		example_output[index] = 0

		dropped_out := 0.0
		for _, value := range example_output {
			if value == 0 {
				dropped_out += 1
			}
		}

		if dropped_out/float64(len(example_output)) >= dropout_rate {
			break
		}
	}
	fmt.Println(example_output)

}

func Run2() {
	fmt.Println("Dropout with my own Binomial wannabe")
	dropout_rate := 0.5
	example_output := mat.NewDense(1, 10, []float64{
		0.27, -1.03, .67, .99, 0.05,
		-0.37, -2.01, 1.13, -0.07, 0.73})
	binomial_mask := mat.NewDense(1, 10, u.Binomial(1, 1-dropout_rate, 10))
	example_output.MulElem(example_output, binomial_mask)
	fmt.Println("Before scaling", example_output)
	example_output.Apply(func(r, c int, v float64) float64 {
		return v / (1.0 - dropout_rate)
	}, example_output)
	fmt.Println("After scaling", example_output)
}

func Run3() {
	fmt.Print("Full run with Dropout")
	sample_size := 1000
	class_size := 3
	X, y := u.Create_spiral_data(sample_size, class_size)

	y_one_hot := mat.NewDense(sample_size*class_size, class_size, nil)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		y_one_hot.Set(i, int(y.At(i, 0)), 1.0)
	}

	dense_1 := u.NewLayerDense(2, 512)
	dense_1.Name = "Dense 1"
	dense_1.Bias_Regulizer_L2 = 5e-4
	dense_1.Weight_Regulizer_L2 = 5e-4
	activation_1 := u.NewActivationRelu()
	dropout_1 := u.NewLayerDropout(.1)

	dense_2 := u.NewLayerDense(512, class_size)
	dense_2.Name = "Dense 2"
	loss_activation := u.NewActivationSoftMaxLossCategoricalCrossEntropy()
	optimizer := u.NewOptimizerAdam(0.02, 5e-7, 1e-7, 0.9, 0.999)

	max_epoch := 20001
	for i := 0; i < max_epoch; i++ {
		dense_1.Forward(X)
		activation_1.Forward(dense_1.Output)
		dropout_1.Forward(activation_1.Output)
		dense_2.Forward(dropout_1.Output)

		data_loss := loss_activation.Forward(dense_2.Output, y_one_hot)

		if i%100 == 0 {
			regularization_loss := u.RegularizationLoss(loss_activation.Loss, dense_1) + u.RegularizationLoss(loss_activation.Loss, dense_2)
			loss := data_loss + regularization_loss
			fmt.Println("epoch", i, "data loss -->", data_loss, "regularization_loss -->", regularization_loss, "loss -->", loss)
			accuracy := u.Accuracy(loss_activation.Activation.Output, y_one_hot)
			fmt.Println("epoch", i, "acc -->", accuracy)
		}

		//backward pass
		loss_activation.Backward(loss_activation.Activation.Output, y_one_hot)
		dense_2.Backward(loss_activation.Dinputs)
		dropout_1.Backward(dense_2.Dinputs)
		activation_1.Backward(dropout_1.Dinputs)
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
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch14/spiral_output.png"); err != nil {
		panic(err)
	}
}
