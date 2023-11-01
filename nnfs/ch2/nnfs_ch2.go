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
	fmt.Println("Starting with the basics")
	inputs := [3]float32{1, 2, 3}
	weights := [3]float32{.2, .8, -.5}
	var bias float32 = 2.0
	output := inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
	fmt.Println("Output ->", output)
}

func Run2() {
	fmt.Println("Full layer")
	inputs := []float32{1, 2, 3, 2.5}

	weights1 := []float32{.2, .8, -.5, 1}
	weights2 := []float32{.5, -.91, 0.26, -.5}
	weights3 := []float32{-.26, -.27, .17, .87}

	var bias1 float32 = 2
	var bias2 float32 = 3
	var bias3 float32 = .5

	outputs := []float32{
		//neuraon 1:
		inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
		//neuron 2:
		inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
		//neuron 3:
		inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3,
	}

	fmt.Println("Outputs ->", outputs)
}

func Run3() {
	fmt.Println("Full layer with a loop")
	inputs := []float32{1, 2, 3, 2.5}

	weights := [][]float32{
		{.2, .8, -.5, 1},
		{.5, -.91, 0.25, -.5},
		{-.26, -.27, .17, .87}}

	var biases = []float32{2, 3, .5}

	outputs := [3]float32{0, 0, 0}

	for neuron_count, _ := range biases {
		//multiple weight * input and sum them up with a for loop
		for input_index, _ := range inputs {
			outputs[neuron_count] += inputs[input_index] * weights[neuron_count][input_index]
		}
		//add a bias
		outputs[neuron_count] += biases[neuron_count]
	}

	fmt.Println("Outputs ->", outputs)
}

func Run4() {
	fmt.Println("Dot product")

	a := []float64{1, 2, 3}
	b := []float64{2, 3, 4}

	dot_product := a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
	fmt.Println("Manual dot", dot_product)

	var c float64 = mat.Dot(mat.NewVecDense(3, a), mat.NewVecDense(3, b))
	fmt.Println("Dot with library:", c)

	inputs := []float64{1, 2, 3}
	weights := []float64{.2, .8, -.5}
	var bias float64 = 2.0
	output := mat.Dot(mat.NewVecDense(3, inputs), mat.NewVecDense(3, weights)) + bias
	fmt.Println("Output ->", output)
}

func Run5() {
	fmt.Println("Full layer with library and dot product")
	inputs := mat.NewDense(1, 4, []float64{1, 2, 3, 2.5})

	weights := mat.NewDense(3, 4, []float64{
		.2, .8, -.5, 1,
		.5, -.91, 0.26, -.5,
		-.26, -.27, .17, .87})
	weightsT := weights.T()

	var biases = mat.NewVecDense(3, []float64{2, 3, .5})
	biasesT := biases.T()

	outputs_matrix := mat.NewDense(1, 3, nil)
	outputs_matrix.Mul(inputs, weightsT)
	outputs_matrix.Add(outputs_matrix, biasesT)
	outputs_vector := outputs_matrix.RowView(0)
	fmt.Println("Output with library: ", outputs_vector)
}

func Run6() {
	fmt.Println("Full layer with a batch of inputs")
	inputs := mat.NewDense(3, 4,
		[]float64{
			1, 2, 3, 2.5,
			3, 5, -1, 2,
			-1.5, 2.7, 3.3, -0.8})

	weights := mat.NewDense(3, 4, []float64{
		.2, .8, -.5, 1,
		.5, -.91, 0.26, -.5,
		-.26, -.27, .17, .87})
	weightsT := weights.T()

	var rawB = []float64{2, 3, .5}
	stackedB := append(rawB, rawB...)
	stackedB = append(stackedB, rawB...)
	var biases = mat.NewDense(3, 3, stackedB)

	outputs_matrix := mat.NewDense(3, 3, nil)
	outputs_matrix.Mul(inputs, weightsT)
	outputs_matrix.Add(outputs_matrix, biases)
	fmt.Println("Output with library: ", outputs_matrix)
}

func Run7() {
	fmt.Println("Fully connected 2 layers with a batch of inputs")
	inputs := mat.NewDense(3, 4,
		[]float64{
			1, 2, 3, 2.5,
			3, 5, -1, 2,
			-1.5, 2.7, 3.3, -0.8})

	weights1 := mat.NewDense(3, 4, []float64{
		.2, .8, -.5, 1,
		.5, -.91, 0.26, -.5,
		-.26, -.27, .17, .87})
	weightsT1 := weights1.T()

	weights2 := mat.NewDense(3, 3, []float64{
		.1, -.14, .5,
		-.5, .12, -.33,
		-.44, .73, -.13})
	weightsT2 := weights2.T()

	var rawB1 = []float64{2, 3, .5}
	stackedB1 := append(rawB1, rawB1...)
	stackedB1 = append(stackedB1, rawB1...)
	var biases1 = mat.NewDense(3, 3, stackedB1)

	var rawB2 = []float64{-1, 2, -.5}
	stackedB2 := append(rawB2, rawB2...)
	stackedB2 = append(stackedB2, rawB2...)
	var biases2 = mat.NewDense(3, 3, stackedB2)

	outputs_matrixInputToL1 := mat.NewDense(3, 3, nil)
	outputs_matrixInputToL1.Mul(inputs, weightsT1)
	outputs_matrixInputToL1.Add(outputs_matrixInputToL1, biases1)

	outputs_MatrixL1ToL2 := mat.NewDense(3, 3, nil)
	outputs_MatrixL1ToL2.Mul(outputs_matrixInputToL1, weightsT2)
	outputs_MatrixL1ToL2.Add(outputs_MatrixL1ToL2, biases2)

	fmt.Println("Output with library: ", outputs_MatrixL1ToL2)
}

func Run8() {
	X, y := u.Create_spiral_data(100, 3)
	fmt.Println("X:", X, "y:", y)
	p := plot.New()
	p.Title.Text = "Spiral"
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
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "ch2/spiral.png"); err != nil {
		panic(err)
	}
}
